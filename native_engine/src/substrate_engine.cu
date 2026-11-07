// Substrate Logistics CUDA kernels.
//
// Physics:
//   g_Omega      = N_j * K_Omega * Theta(rho_N_j - 4.0) / R^2
//   1PN factor   = 1 + 3 * N_j * K_Omega / (c^2 * R)
//   a_i          = sum_j g_Omega * (1PN factor) * (r_j - r_i) / R,  R >= 1 m
// Integration: 2nd-order symplectic Velocity Verlet; heavy bodies sub-stepped in shared memory.

#include "substrate_engine.h"
#include "substrate_constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                   \
    do {                                                                                   \
        cudaError_t _e = (expr);                                                           \
        if (_e != cudaSuccess)                                                             \
            throw std::runtime_error(std::string(#expr " failed: ") + cudaGetErrorString(_e)); \
    } while (0)

namespace substrate {
namespace {

constexpr int PARTICLE_BLOCK = 256;
constexpr double INV_AU = 1.0 / AU;
constexpr float MIN_D2_AU = static_cast<float>((1.0 / AU) * (1.0 / AU));  // 1 m clamp, in AU^2
constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

// Total integer bins: MAIN_BINS + KUIPER_BINS + ECC_BINS + 5 regime counters
constexpr int TOTAL_INT_BINS = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS + 5;

// ---------------------------------------------------------------------------
// Heavy bodies: fully coupled FP64, one block, one thread per body.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void heavyAccel(int i, double x, double y, double z,
                                           const double* sx, const double* sy, const double* sz,
                                           const double* smu, const double* src, int M,
                                           double& ax, double& ay, double& az) {
    ax = ay = az = 0.0;
    for (int j = 0; j < M; ++j) {
        if (j == i) continue;
        const double dx = sx[j] - x, dy = sy[j] - y, dz = sz[j] - z;
        const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
        const double inv = rsqrt(fmax(d2, 1.0));
        const double g = smu[j] * inv * inv * (1.0 + src[j] * inv);
        const double s = g * inv;
        ax = fma(s, dx, ax);
        ay = fma(s, dy, ay);
        az = fma(s, dz, az);
    }
}

__global__ void heavyInitAccKernel(double* __restrict__ st, const double* __restrict__ mu,
                                   const double* __restrict__ rc, int M, bool rel) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY], src[MAX_HEAVY];
    const int i = threadIdx.x;
    if (i < M) {
        sx[i] = st[i];
        sy[i] = st[M + i];
        sz[i] = st[2 * M + i];
        smu[i] = mu[i];
        src[i] = (rel && mu[i] > 0.0) ? rc[i] : 0.0;
    }
    __syncthreads();
    if (i < M) {
        double ax, ay, az;
        heavyAccel(i, sx[i], sy[i], sz[i], sx, sy, sz, smu, src, M, ax, ay, az);
        st[6 * M + i] = ax;
        st[7 * M + i] = ay;
        st[8 * M + i] = az;
    }
}

__global__ void heavySubstepKernel(double* __restrict__ st, const double* __restrict__ mu,
                                   const double* __restrict__ rc, int M, bool rel,
                                   double dt, int substeps) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY], src[MAX_HEAVY];
    const int i = threadIdx.x;
    const bool active = i < M;

    double x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0, ax = 0, ay = 0, az = 0;
    if (active) {
        x = st[i];          y = st[M + i];      z = st[2 * M + i];
        vx = st[3 * M + i]; vy = st[4 * M + i]; vz = st[5 * M + i];
        ax = st[6 * M + i]; ay = st[7 * M + i]; az = st[8 * M + i];
        smu[i] = mu[i];
        src[i] = (rel && mu[i] > 0.0) ? rc[i] : 0.0;
    }

    const double h = 0.5 * dt;
    for (int k = 0; k < substeps; ++k) {
        if (active) {
            vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);
            x = fma(vx, dt, x);  y = fma(vy, dt, y);  z = fma(vz, dt, z);
        }
        __syncthreads();
        if (active) { sx[i] = x; sy[i] = y; sz[i] = z; }
        __syncthreads();
        if (active) {
            heavyAccel(i, x, y, z, sx, sy, sz, smu, src, M, ax, ay, az);
            vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);
        }
    }

    if (active) {
        st[i] = x;          st[M + i] = y;      st[2 * M + i] = z;
        st[3 * M + i] = vx; st[4 * M + i] = vy; st[5 * M + i] = vz;
        st[6 * M + i] = ax; st[7 * M + i] = ay; st[8 * M + i] = az;
    }
}

// ---------------------------------------------------------------------------
// Test particles: O(N*M), heavy bodies broadcast from shared memory.
// ---------------------------------------------------------------------------
struct HeavyShared {
    double x[MAX_HEAVY], y[MAX_HEAVY], z[MAX_HEAVY];
    double mu[MAX_HEAVY], rc[MAX_HEAVY];
    float muAU[MAX_HEAVY];
    float rcAU[MAX_HEAVY];
};

__device__ __forceinline__ void loadHeavyShared(HeavyShared& s, const double* __restrict__ hs,
                                                const double* __restrict__ mu,
                                                const double* __restrict__ rc, int M, bool rel) {
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        s.x[j] = hs[j];
        s.y[j] = hs[M + j];
        s.z[j] = hs[2 * M + j];
        const double m = mu[j];
        const double r = (rel && m > 0.0) ? rc[j] : 0.0;
        s.mu[j] = m;
        s.rc[j] = r;
        s.muAU[j] = static_cast<float>(m * (INV_AU * INV_AU * INV_AU));
        s.rcAU[j] = static_cast<float>(r * INV_AU);
    }
}

template <bool MIXED>
__device__ __forceinline__ void particleAccel(double x, double y, double z, const HeavyShared& s,
                                              int M, double& ax, double& ay, double& az) {
    if constexpr (!MIXED) {
        ax = ay = az = 0.0;
        for (int j = 0; j < M; ++j) {
            const double dx = s.x[j] - x, dy = s.y[j] - y, dz = s.z[j] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = rsqrt(fmax(d2, 1.0));
            const double g = s.mu[j] * inv * inv * (1.0 + s.rc[j] * inv);
            const double sc = g * inv;
            ax = fma(sc, dx, ax);
            ay = fma(sc, dy, ay);
            az = fma(sc, dz, az);
        }
    } else {
        float axN = 0.f, ayN = 0.f, azN = 0.f;
        float axR = 0.f, ayR = 0.f, azR = 0.f;
        for (int j = 0; j < M; ++j) {
            const float dx = static_cast<float>((s.x[j] - x) * INV_AU);
            const float dy = static_cast<float>((s.y[j] - y) * INV_AU);
            const float dz = static_cast<float>((s.z[j] - z) * INV_AU);
            const float d2 = fmaxf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)), MIN_D2_AU);
            const float inv = rsqrtf(d2);
            const float s1 = s.muAU[j] * inv * inv * inv;
            const float s2 = s1 * s.rcAU[j] * inv;
            axN = fmaf(s1, dx, axN); ayN = fmaf(s1, dy, ayN); azN = fmaf(s1, dz, azN);
            axR = fmaf(s2, dx, axR); ayR = fmaf(s2, dy, ayR); azR = fmaf(s2, dz, azR);
        }
        ax = (static_cast<double>(axN) + static_cast<double>(axR)) * AU;
        ay = (static_cast<double>(ayN) + static_cast<double>(ayR)) * AU;
        az = (static_cast<double>(azN) + static_cast<double>(azR)) * AU;
    }
}

template <bool MIXED>
__global__ void particleInitAccKernel(double* __restrict__ p, size_t N,
                                      const double* __restrict__ hs, const double* __restrict__ mu,
                                      const double* __restrict__ rc, int M, bool rel) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, rc, M, rel);
    __syncthreads();
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    double ax, ay, az;
    particleAccel<MIXED>(p[i], p[N + i], p[2 * N + i], s, M, ax, ay, az);
    p[6 * N + i] = ax;
    p[7 * N + i] = ay;
    p[8 * N + i] = az;
}

template <bool MIXED>
__global__ void particleStepKernel(double* __restrict__ p, size_t N,
                                   const double* __restrict__ hs, const double* __restrict__ mu,
                                   const double* __restrict__ rc, int M, bool rel, double dt) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, rc, M, rel);
    __syncthreads();
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;

    double x = p[i], y = p[N + i], z = p[2 * N + i];
    double vx = p[3 * N + i], vy = p[4 * N + i], vz = p[5 * N + i];
    double ax = p[6 * N + i], ay = p[7 * N + i], az = p[8 * N + i];

    const double h = 0.5 * dt;
    vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);
    x = fma(vx, dt, x);  y = fma(vy, dt, y);  z = fma(vz, dt, z);
    particleAccel<MIXED>(x, y, z, s, M, ax, ay, az);
    vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);

    p[i] = x;          p[N + i] = y;      p[2 * N + i] = z;
    p[3 * N + i] = vx; p[4 * N + i] = vy; p[5 * N + i] = vz;
    p[6 * N + i] = ax; p[7 * N + i] = ay; p[8 * N + i] = az;
}

__global__ void writeRenderKernel(const double* __restrict__ st, size_t count,
                                  const double* __restrict__ hs, int M, int origin,
                                  float4* __restrict__ out) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    double ox = 0.0, oy = 0.0, oz = 0.0;
    if (origin >= 0) {
        ox = hs[origin];
        oy = hs[M + origin];
        oz = hs[2 * M + origin];
    }
    out[i] = make_float4(static_cast<float>((st[i] - ox) * INV_AU),
                         static_cast<float>((st[count + i] - oy) * INV_AU),
                         static_cast<float>((st[2 * count + i] - oz) * INV_AU), 0.0f);
}

// ---------------------------------------------------------------------------
// Zero-readback GPU Belt Statistical Mechanics Kernel (a, e, i, T_J histograms)
// ---------------------------------------------------------------------------
__global__ void beltStatsKernel(const double* __restrict__ p, size_t N,
                                const double* __restrict__ hs, const double* __restrict__ mu,
                                int M, double aJupAU,
                                uint32_t* __restrict__ outBins, float* __restrict__ outSums) {
    __shared__ uint32_t sBins[TOTAL_INT_BINS];
    __shared__ float sSumEcc;
    __shared__ float sSumInc;

    for (int b = threadIdx.x; b < TOTAL_INT_BINS; b += blockDim.x) sBins[b] = 0u;
    if (threadIdx.x == 0) {
        sSumEcc = 0.0f;
        sSumInc = 0.0f;
    }
    __syncthreads();

    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < N) {
        const double muSun = mu[0];
        if (muSun <= 0.0) {
            // Sun is decompiled (Theta = 0): all orbits are dynamically unbound!
            atomicAdd(&sBins[BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS + 3], 1u);
        } else {
            const double rx = p[i] - hs[0];
            const double ry = p[N + i] - hs[M];
            const double rz = p[2 * N + i] - hs[2 * M];
            const double vx = p[3 * N + i] - hs[3 * M];
            const double vy = p[4 * N + i] - hs[4 * M];
            const double vz = p[5 * N + i] - hs[5 * M];

            const double r = sqrt(fmax(rx * rx + ry * ry + rz * rz, 1.0));
            const double v2 = vx * vx + vy * vy + vz * vz;
            const double eps = 0.5 * v2 - muSun / r;

            const int baseCounter = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS;
            if (eps >= -1e-12) {
                atomicAdd(&sBins[baseCounter + 3], 1u);  // unbound
            } else {
                const double a_m = -muSun / (2.0 * eps);
                const double a_AU = a_m * INV_AU;

                const double hx = ry * vz - rz * vy;
                const double hy = rz * vx - rx * vz;
                const double hz = rx * vy - ry * vx;
                const double h2 = hx * hx + hy * hy + hz * hz;
                const double hMag = sqrt(fmax(h2, 1e-30));

                const double eccTerm = 1.0 + (2.0 * eps * h2) / (muSun * muSun);
                const double ecc = sqrt(fmax(eccTerm, 0.0));
                const double cosInc = fmin(fmax(hz / hMag, -1.0), 1.0);
                const double incDeg = acos(cosInc) * RAD2DEG;

                // Main Belt histogram [2.00, 3.50] AU
                if (a_AU >= 2.00 && a_AU < 3.50) {
                    int bin = static_cast<int>((a_AU - 2.00) / 1.50 * BeltStats::MAIN_BINS);
                    bin = min(max(bin, 0), BeltStats::MAIN_BINS - 1);
                    atomicAdd(&sBins[bin], 1u);
                }
                // Kuiper Belt & Scattered Disk histogram [30.0, 72.0] AU
                if (a_AU >= BeltStats::KUIPER_MIN_AU && a_AU < BeltStats::KUIPER_MAX_AU) {
                    constexpr double span = BeltStats::KUIPER_MAX_AU - BeltStats::KUIPER_MIN_AU;
                    int bin = static_cast<int>((a_AU - BeltStats::KUIPER_MIN_AU) / span * BeltStats::KUIPER_BINS);
                    bin = min(max(bin, 0), BeltStats::KUIPER_BINS - 1);
                    atomicAdd(&sBins[BeltStats::MAIN_BINS + bin], 1u);
                }
                // Eccentricity histogram [0.00, 0.50]
                if (ecc < 0.50) {
                    int bin = static_cast<int>((ecc / 0.50) * BeltStats::ECC_BINS);
                    bin = min(max(bin, 0), BeltStats::ECC_BINS - 1);
                    atomicAdd(&sBins[BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + bin], 1u);
                }

                // Jupiter Tisserand parameter T_J
                const double tj = (aJupAU / a_AU) + 2.0 * sqrt(fmax((a_AU / aJupAU) * (1.0 - ecc * ecc), 0.0)) * cosInc;
                if (tj > 3.0) {
                    atomicAdd(&sBins[baseCounter + 0], 1u);  // Asteroidal
                } else if (tj > 2.0) {
                    atomicAdd(&sBins[baseCounter + 1], 1u);  // JFC
                } else {
                    atomicAdd(&sBins[baseCounter + 2], 1u);  // Halley / Damocloid
                }
                atomicAdd(&sBins[baseCounter + 4], 1u);      // Bound total
                atomicAdd(&sSumEcc, static_cast<float>(ecc));
                atomicAdd(&sSumInc, static_cast<float>(incDeg));
            }
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < TOTAL_INT_BINS; b += blockDim.x) {
        if (sBins[b] > 0u) atomicAdd(&outBins[b], sBins[b]);
    }
    if (threadIdx.x == 0) {
        if (sSumEcc > 0.0f) atomicAdd(&outSums[0], sSumEcc);
        if (sSumInc > 0.0f) atomicAdd(&outSums[1], sSumInc);
    }
}

unsigned int blocksFor(size_t n, int block) {
    return static_cast<unsigned int>((n + block - 1) / block);
}

} // namespace

// ---------------------------------------------------------------------------
// Host engine
// ---------------------------------------------------------------------------
SubstrateEngine::SubstrateEngine() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_snapEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&m_timerStart));
    CUDA_CHECK(cudaEventCreate(&m_timerStop));
}

SubstrateEngine::~SubstrateEngine() {
    release();
    cudaEventDestroy(m_timerStop);
    cudaEventDestroy(m_timerStart);
    cudaEventDestroy(m_snapEvent);
    cudaStreamDestroy(m_stream);
}

void SubstrateEngine::release() {
    if (m_stream) cudaStreamSynchronize(m_stream);
    cudaFree(d_heavy);      d_heavy = nullptr;
    cudaFree(d_mu);         d_mu = nullptr;
    cudaFree(d_rc);         d_rc = nullptr;
    cudaFree(d_particles);  d_particles = nullptr;
    cudaFree(d_beltBins);   d_beltBins = nullptr;
    cudaFree(d_beltSums);   d_beltSums = nullptr;
    cudaFreeHost(h_snapPinned); h_snapPinned = nullptr;
    m_deviceBytes = 0;
    m_snapPending = false;
    m_timerPending = false;
}

void SubstrateEngine::refreshCachedAccelerations(bool relativistic) {
    if (m_numHeavy <= 0) return;
    heavyInitAccKernel<<<1, MAX_HEAVY, 0, m_stream>>>(d_heavy, d_mu, d_rc, m_numHeavy, relativistic);
    if (m_numParticles > 0) {
        particleInitAccKernel<false><<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_particles, m_numParticles, d_heavy, d_mu, d_rc, m_numHeavy, relativistic);
    }
    CUDA_CHECK(cudaGetLastError());
}

void SubstrateEngine::load(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles,
                           bool correctBarycenter) {
    release();

    const int M = static_cast<int>(heavy.size());
    if (M < 1 || M > MAX_HEAVY)
        throw std::runtime_error("Heavy body count must be in [1, " + std::to_string(MAX_HEAVY) + "]");
    m_numHeavy = M;

    m_mass.assign(M, 0.0);
    m_radius.assign(M, 0.0);
    m_nodes.assign(M, 0.0);
    m_baseDensity.assign(M, 0.0);
    m_density.assign(M, 0.0);
    m_mu.assign(M, 0.0);
    m_rc.assign(M, 0.0);
    m_overrideActive.assign(M, false);

    std::vector<double> hs(9 * static_cast<size_t>(M), 0.0);
    for (int i = 0; i < M; ++i) {
        const auto& b = heavy[i];
        m_mass[i] = b.mass;
        m_radius[i] = b.radiusM;
        m_nodes[i] = b.mass / M_P;
        m_baseDensity[i] = m_nodes[i] / b.vol;
        m_density[i] = m_baseDensity[i];
        const double theta = (m_density[i] >= RHO_GC) ? 1.0 : 0.0;
        m_mu[i] = m_nodes[i] * K_OMEGA * theta;
        m_rc[i] = 3.0 * m_nodes[i] * K_OMEGA / (C_LIGHT * C_LIGHT);
        for (int k = 0; k < 3; ++k) {
            hs[k * M + i] = b.pos[k];
            hs[(3 + k) * M + i] = b.vel[k];
        }
    }

    if (correctBarycenter && M > 1) {
        double P[3] = {}, mTot = 0.0;
        for (int i = 0; i < M; ++i) {
            mTot += m_mass[i];
            for (int k = 0; k < 3; ++k) P[k] += m_mass[i] * hs[(3 + k) * M + i];
        }
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < 3; ++k) hs[(3 + k) * M + i] -= P[k] / mTot;
    }

    CUDA_CHECK(cudaMalloc(&d_heavy, hs.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rc, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beltBins, TOTAL_INT_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_beltSums, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_heavy, hs.data(), hs.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, m_mu.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc, m_rc.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&h_snapPinned, 6 * M * sizeof(double)));
    m_deviceBytes = (hs.size() + 2 * M) * sizeof(double);

    const size_t N = particles.count();
    m_numParticles = N;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_particles, 9 * N * sizeof(double)));
        m_deviceBytes += 9 * N * sizeof(double);
        const std::vector<double>* src[6] = {&particles.px, &particles.py, &particles.pz,
                                             &particles.vx, &particles.vy, &particles.vz};
        for (int k = 0; k < 6; ++k)
            CUDA_CHECK(cudaMemcpy(d_particles + k * N, src[k]->data(), N * sizeof(double),
                                  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_particles + 6 * N, 0, 3 * N * sizeof(double)));
    }

    refreshCachedAccelerations(true);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    m_simTime = 0.0;
}

void SubstrateEngine::setDensityOverride(int bodyIdx, bool enabled, double overrideRhoNodes, bool relativistic) {
    if (bodyIdx < 0 || bodyIdx >= m_numHeavy) return;
    const double prevMu = m_mu[bodyIdx];
    m_overrideActive[bodyIdx] = enabled;
    m_density[bodyIdx] = enabled ? std::max(overrideRhoNodes, 0.0) : m_baseDensity[bodyIdx];
    const double theta = (m_density[bodyIdx] >= RHO_GC) ? 1.0 : 0.0;
    m_mu[bodyIdx] = m_nodes[bodyIdx] * K_OMEGA * theta;

    if (m_mu[bodyIdx] != prevMu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu + bodyIdx, &m_mu[bodyIdx], sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        refreshCachedAccelerations(relativistic);
    }
}

void SubstrateEngine::resetDensityOverrides(bool relativistic) {
    bool anyChanged = false;
    for (int i = 0; i < m_numHeavy; ++i) {
        m_overrideActive[i] = false;
        m_density[i] = m_baseDensity[i];
        const double newMu = m_nodes[i] * K_OMEGA * ((m_density[i] >= RHO_GC) ? 1.0 : 0.0);
        if (newMu != m_mu[i]) {
            m_mu[i] = newMu;
            anyChanged = true;
        }
    }
    if (anyChanged && d_mu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu, m_mu.data(), m_numHeavy * sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        refreshCachedAccelerations(relativistic);
    }
}

bool SubstrateEngine::isDensityOverridden(int bodyIdx) const {
    return (bodyIdx >= 0 && bodyIdx < m_numHeavy) ? m_overrideActive[bodyIdx] : false;
}

double SubstrateEngine::baseDensity(int bodyIdx) const {
    return (bodyIdx >= 0 && bodyIdx < m_numHeavy) ? m_baseDensity[bodyIdx] : 0.0;
}

void SubstrateEngine::computeBeltStats(BeltStats& out, double jupiterSemiMajorAU) {
    out = BeltStats{};
    if (m_numParticles == 0 || !d_particles || !d_beltBins) return;

    CUDA_CHECK(cudaMemsetAsync(d_beltBins, 0, TOTAL_INT_BINS * sizeof(uint32_t), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_beltSums, 0, 2 * sizeof(float), m_stream));

    beltStatsKernel<<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
        d_particles, m_numParticles, d_heavy, d_mu, m_numHeavy, jupiterSemiMajorAU,
        d_beltBins, d_beltSums);
    CUDA_CHECK(cudaGetLastError());

    uint32_t hBins[TOTAL_INT_BINS] = {};
    float hSums[2] = {};
    CUDA_CHECK(cudaMemcpyAsync(hBins, d_beltBins, sizeof(hBins), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(hSums, d_beltSums, sizeof(hSums), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    for (int i = 0; i < BeltStats::MAIN_BINS; ++i)
        out.mainBeltHist[i] = static_cast<float>(hBins[i]);
    for (int i = 0; i < BeltStats::KUIPER_BINS; ++i)
        out.kuiperBeltHist[i] = static_cast<float>(hBins[BeltStats::MAIN_BINS + i]);
    for (int i = 0; i < BeltStats::ECC_BINS; ++i)
        out.eccHist[i] = static_cast<float>(hBins[BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + i]);

    const int base = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS;
    out.asteroidalCount = hBins[base + 0];
    out.jfcCount        = hBins[base + 1];
    out.halleyCount     = hBins[base + 2];
    out.unboundCount    = hBins[base + 3];
    out.boundCount      = hBins[base + 4];
    if (out.boundCount > 0) {
        out.meanEcc    = hSums[0] / static_cast<float>(out.boundCount);
        out.meanIncDeg = hSums[1] / static_cast<float>(out.boundCount);
    }
}

void SubstrateEngine::step(const StepParams& p, bool captureSnapshot) {
    const int M = m_numHeavy;
    const int sub = std::max(1, p.substeps);

    heavySubstepKernel<<<1, MAX_HEAVY, 0, m_stream>>>(d_heavy, d_mu, d_rc, M, p.relativistic,
                                                      p.dtSeconds / sub, sub);
    m_simTime += p.dtSeconds;
    if (captureSnapshot) requestSnapshot();

    if (m_numParticles > 0) {
        const unsigned int blocks = blocksFor(m_numParticles, PARTICLE_BLOCK);
        if (p.precision == Precision::Mixed)
            particleStepKernel<true><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_rc, M, p.relativistic, p.dtSeconds);
        else
            particleStepKernel<false><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_rc, M, p.relativistic, p.dtSeconds);
    }
    CUDA_CHECK(cudaGetLastError());
}

void SubstrateEngine::requestSnapshot() {
    CUDA_CHECK(cudaMemcpyAsync(h_snapPinned, d_heavy, 6 * m_numHeavy * sizeof(double),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaEventRecord(m_snapEvent, m_stream));
    m_snapTime = m_simTime;
    m_snapPending = true;
}

const HeavySnapshot& SubstrateEngine::snapshot() {
    if (m_snapPending) {
        CUDA_CHECK(cudaEventSynchronize(m_snapEvent));
        const int M = m_numHeavy;
        m_snapshot.pos.resize(3 * M);
        m_snapshot.vel.resize(3 * M);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < 3; ++k) {
                m_snapshot.pos[3 * i + k] = h_snapPinned[k * M + i];
                m_snapshot.vel[3 * i + k] = h_snapPinned[(3 + k) * M + i];
            }
        m_snapshot.simTimeSeconds = m_snapTime;
        m_snapPending = false;
    }
    return m_snapshot;
}

void SubstrateEngine::writeRenderPositions(float4* heavyOut, float4* particleOut, int originIdx) {
    const int M = m_numHeavy;
    if (originIdx >= M) originIdx = -1;
    if (heavyOut)
        writeRenderKernel<<<1, MAX_HEAVY, 0, m_stream>>>(d_heavy, M, d_heavy, M, originIdx, heavyOut);
    if (particleOut && m_numParticles > 0)
        writeRenderKernel<<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_particles, m_numParticles, d_heavy, M, originIdx, particleOut);
    CUDA_CHECK(cudaGetLastError());
}

void SubstrateEngine::beginTimer() {
    if (m_timerPending) {
        if (cudaEventQuery(m_timerStop) != cudaSuccess) return;
        cudaEventElapsedTime(&m_lastTimedMs, m_timerStart, m_timerStop);
        m_timerPending = false;
    }
    CUDA_CHECK(cudaEventRecord(m_timerStart, m_stream));
}

void SubstrateEngine::endTimer() {
    if (m_timerPending) return;
    CUDA_CHECK(cudaEventRecord(m_timerStop, m_stream));
    m_timerPending = true;
}

float SubstrateEngine::lastTimedMs() {
    if (m_timerPending && cudaEventQuery(m_timerStop) == cudaSuccess) {
        cudaEventElapsedTime(&m_lastTimedMs, m_timerStart, m_timerStop);
        m_timerPending = false;
    }
    return m_lastTimedMs;
}

void SubstrateEngine::copyParticlePositions(std::vector<double>& xyzOut) {
    const size_t N = m_numParticles;
    std::vector<double> soa(3 * N);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    if (N > 0)
        CUDA_CHECK(cudaMemcpy(soa.data(), d_particles, 3 * N * sizeof(double), cudaMemcpyDeviceToHost));
    xyzOut.resize(3 * N);
    for (size_t i = 0; i < N; ++i)
        for (int k = 0; k < 3; ++k) xyzOut[3 * i + k] = soa[k * N + i];
}

void SubstrateEngine::synchronize() { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }

double SubstrateEngine::gravitationalFloorAU() const {
    if (m_numHeavy <= 0) return 0.0;
    const double compiledNodes = (m_density[0] >= RHO_GC) ? m_nodes[0] : 0.0;
    return gravitationalFloorMetersFromNodes(compiledNodes) / AU;
}

double SubstrateEngine::heliopauseRadiusAU() const {
    // 20-vertex dodecahedral astropause boundary: R_HP = R_g / 20 (~121.54 AU for 1.00 M_sun)
    return gravitationalFloorAU() / V_DODEC;
}

double SubstrateEngine::heliopauseTopologicalDragAU(double swProtonCm3At1AU,
                                                    double swSpeedKms,
                                                    double heliosheathStagnationFactor) {
    // Solar wind dynamic pressure at 1 AU: P_SW(1 AU) = m_p * n_SW * v_SW^2 (Pa)
    // Stagnation balance against Substrate Topological Drag floor P_delta = C_delta * Sigma_{0,Ledger}:
    //   R_TS = 1 AU * sqrt(P_SW(1 AU) / P_DRAG_FLOOR),  R_HP = factor * R_TS
    const double n_m3 = std::max(swProtonCm3At1AU, 0.01) * 1.0e6;
    const double v_ms = std::max(swSpeedKms, 1.0) * 1000.0;
    const double pSW_1AU = M_P * n_m3 * v_ms * v_ms;
    const double rTS_AU = std::sqrt(pSW_1AU / P_DRAG_FLOOR);
    return heliosheathStagnationFactor * rTS_AU;
}

double SubstrateEngine::newtonianEnergy(const HeavySnapshot& s) const {
    const int M = m_numHeavy;
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* v = &s.vel[3 * i];
        ke += 0.5 * m_mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const double dx = s.pos[3 * j] - s.pos[3 * i];
            const double dy = s.pos[3 * j + 1] - s.pos[3 * i + 1];
            const double dz = s.pos[3 * j + 2] - s.pos[3 * i + 2];
            pe -= 0.5 * m_mass[i] * m_mu[j] / std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
        }
    }
    return ke + pe;
}

double SubstrateEngine::angularMomentumMag(const HeavySnapshot& s) const {
    const int M = m_numHeavy;
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* r = &s.pos[3 * i];
        const double* v = &s.vel[3 * i];
        const double m = m_mass[i];
        Lx += m * (r[1] * v[2] - r[2] * v[1]);
        Ly += m * (r[2] * v[0] - r[0] * v[2]);
        Lz += m * (r[0] * v[1] - r[1] * v[0]);
    }
    return std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz);
}

double SubstrateEngine::barycenterDriftSpeed(const HeavySnapshot& s) const {
    const int M = m_numHeavy;
    double Px = 0.0, Py = 0.0, Pz = 0.0, mTot = 0.0;
    for (int i = 0; i < M; ++i) {
        const double m = m_mass[i];
        mTot += m;
        Px += m * s.vel[3 * i + 0];
        Py += m * s.vel[3 * i + 1];
        Pz += m * s.vel[3 * i + 2];
    }
    if (mTot <= 0.0) return 0.0;
    return std::sqrt(Px * Px + Py * Py + Pz * Pz) / mTot;
}

} // namespace substrate
