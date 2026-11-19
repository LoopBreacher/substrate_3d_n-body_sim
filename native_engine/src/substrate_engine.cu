// Substrate Logistics CUDA kernels.
//
// Physics:
//   g_Omega      = N_j * K_Omega * Theta(rho_N_j - 4.0) / R^2
//   1PN factor   = 1 + 3 * N_j * K_Omega / (c^2 * R)
//   a_i          = sum_j g_Omega * (1PN factor) * (r_j - r_i) / R,  R >= 1 m
// Integration:
//   - Heavy bodies: 4th-order Omelyan PEFRL symplectic composition with Kahan-Babuska-Neumaier
//     compensated summation in shared memory.
//   - Test particles: 2nd-order symplectic Velocity Verlet (FP64 or Dual-Accumulator Mixed).

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

constexpr int PARTICLE_BLOCK         = 256;
constexpr int HEAVY_THREADS_PER_BODY = 8;
constexpr double INV_AU = 1.0 / AU;
constexpr float MIN_D2_AU = static_cast<float>((1.0 / AU) * (1.0 / AU));  // 1 m clamp, in AU^2
constexpr double J2_GATE_D2_M2 = (0.01 * AU) * (0.01 * AU);               // 0.01 AU (~1.5M km) sub-mantissa particle J2 gate
constexpr float J2_GATE_D2_AU2 = 1.0e-4f;                                 // (0.01 AU)^2 in AU^2
constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

// Total integer bins: MAIN_BINS + KUIPER_BINS + ECC_BINS + 5 regime counters
constexpr int TOTAL_INT_BINS = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS + 5;

// Omelyan, Mryglod & Folk (2002) 4th-Order PEFRL Symplectic Composition Weights:
// Palindromic 5-Drift, 4-Kick sequence:
//   Drift(a1) -> Kick(b1) -> Drift(a2) -> Kick(b2) -> Drift(a3) -> Kick(b2) -> Drift(a2) -> Kick(b1) -> Drift(a1)
// Satisfies 2*PEFRL_XI + 2*PEFRL_CHI + PEFRL_A3 == 1.0 and 2*PEFRL_B1 + 2*PEFRL_LAMBDA == 1.0 exactly.
constexpr double PEFRL_XI     = +0.1786178958448091e0;
constexpr double PEFRL_LAMBDA = -0.2123418310626054e0;
constexpr double PEFRL_CHI    = -0.06626458266981849e0;
constexpr double PEFRL_A3     = 1.0 - 2.0 * (PEFRL_CHI + PEFRL_XI); // +0.7752933736500188
constexpr double PEFRL_B1     = 0.5 * (1.0 - 2.0 * PEFRL_LAMBDA);   // +0.7123418310626054

// Kahan-Babuska-Neumaier compensated accumulation (106-bit effective mantissa in registers)
// Prevents precision loss when adding small moon displacements to large heliocentric coordinates.
__device__ __forceinline__ void kahanAdd(double& sum, double& comp, double delta) {
    const double t = sum + delta;
    if (fabs(sum) >= fabs(delta)) {
        comp += (sum - t) + delta;
    } else {
        comp += (delta - t) + sum;
    }
    sum = t;
}

// ---------------------------------------------------------------------------
// Heavy bodies: fully coupled FP64, one block, 8-way warp-cooperative per body.
// All __shfl_down_sync calls execute unconditionally across full 32-thread warps.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void heavyAccelCoop8(int bodyIdx, int lane, bool validBody,
                                                double x, double y, double z,
                                                const double* sx, const double* sy, const double* sz,
                                                const double* smu, const double* src,
                                                const double* sj2R2,
                                                const double* sSpinX, const double* sSpinY, const double* sSpinZ,
                                                int parentIdx, uint64_t moonMask,
                                                int M, double& ax, double& ay, double& az) {
    double lx = 0.0, ly = 0.0, lz = 0.0;
    if (validBody) {
        const double rc_i = src[bodyIdx];
        #pragma unroll 4
        for (int j = lane; j < M; j += HEAVY_THREADS_PER_BODY) {
            if (j == bodyIdx) continue;
            const double dx = sx[j] - x, dy = sy[j] - y, dz = sz[j] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = rsqrt(fmax(d2, 1.0));
            const double rcPair = rc_i + src[j];
            const double g = smu[j] * inv * inv * (1.0 + rcPair * inv);
            const double s = g * inv;
            lx = fma(s, dx, lx);
            ly = fma(s, dy, ly);
            lz = fma(s, dz, lz);
        }
    }

    // Unconditional 8-way sub-warp reduction (strict Volta+/Blackwell full-warp safety)
    #pragma unroll
    for (int offset = HEAVY_THREADS_PER_BODY / 2; offset > 0; offset >>= 1) {
        lx += __shfl_down_sync(0xFFFFFFFFu, lx, offset, HEAVY_THREADS_PER_BODY);
        ly += __shfl_down_sync(0xFFFFFFFFu, ly, offset, HEAVY_THREADS_PER_BODY);
        lz += __shfl_down_sync(0xFFFFFFFFu, lz, offset, HEAVY_THREADS_PER_BODY);
    }

    if (validBody && lane == 0) {
        // 1. O(1) Parent Quadrupole J2 evaluation outside the O(M) monopole+1PN loop (zero inner-loop divergence):
        if (parentIdx > 0) {
            const double j2r2_p = sj2R2[parentIdx];
            const double mp = smu[parentIdx];
            if (j2r2_p > 0.0 && mp > 0.0) {
                const double dx = sx[parentIdx] - x, dy = sy[parentIdx] - y, dz = sz[parentIdx] - z;
                const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                const double inv = rsqrt(fmax(d2, 1.0));
                const double inv2 = inv * inv;
                const double spX = sSpinX[parentIdx], spY = sSpinY[parentIdx], spZ = sSpinZ[parentIdx];
                const double rDotS = (-dx * spX - dy * spY - dz * spZ) * inv;
                const double baseQ = 1.5 * mp * j2r2_p * inv2 * inv2;
                const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                const double pole  = -2.0 * rDotS * baseQ;
                lx = fma(coeff, dx, fma(pole, spX, lx));
                ly = fma(coeff, dy, fma(pole, spY, ly));
                lz = fma(coeff, dz, fma(pole, spZ, lz));
            }
        }

        // 2. Exact Newton's 3rd Law J2 reaction force on parent planet bodyIdx from its moons (via 64-bit moon bitmask):
        if (moonMask != 0ULL) {
            const double j2r2_i = sj2R2[bodyIdx];
            if (j2r2_i > 0.0 && smu[bodyIdx] > 0.0) {
                const double spX = sSpinX[bodyIdx], spY = sSpinY[bodyIdx], spZ = sSpinZ[bodyIdx];
                for (unsigned long long mask = moonMask; mask != 0ULL; mask &= (mask - 1ULL)) {
                    const int m = __ffsll(static_cast<long long>(mask)) - 1;
                    const double mm = smu[m];
                    if (mm <= 0.0) continue;
                    const double dx = sx[m] - x, dy = sy[m] - y, dz = sz[m] - z;
                    const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                    const double inv = rsqrt(fmax(d2, 1.0));
                    const double inv2 = inv * inv;
                    const double rDotS = (dx * spX + dy * spY + dz * spZ) * inv;
                    const double baseQ = 1.5 * mm * j2r2_i * inv2 * inv2;
                    const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                    const double pole  = 2.0 * rDotS * baseQ;
                    lx = fma(coeff, dx, fma(pole, spX, lx));
                    ly = fma(coeff, dy, fma(pole, spY, ly));
                    lz = fma(coeff, dz, fma(pole, spZ, lz));
                }
            }
        }

        ax = lx;
        ay = ly;
        az = lz;
    }
}

__global__ void heavyInitAccKernel(double* __restrict__ st, const double* __restrict__ mu,
                                   const double* __restrict__ rc, const double* __restrict__ j2R2,
                                   const double* __restrict__ spinAxis, const int* __restrict__ parentIdx,
                                   const uint64_t* __restrict__ moonMask, int M, bool rel) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY], src[MAX_HEAVY];
    __shared__ double sj2R2[MAX_HEAVY], sSpinX[MAX_HEAVY], sSpinY[MAX_HEAVY], sSpinZ[MAX_HEAVY];
    const int bodyIdx = threadIdx.x / HEAVY_THREADS_PER_BODY;
    const int lane    = threadIdx.x % HEAVY_THREADS_PER_BODY;
    const bool validBody = (bodyIdx < M);
    const bool isLeader  = (validBody && lane == 0);

    int pIdx = -1;
    uint64_t mMask = 0ULL;
    if (isLeader) {
        sx[bodyIdx] = st[bodyIdx];
        sy[bodyIdx] = st[M + bodyIdx];
        sz[bodyIdx] = st[2 * M + bodyIdx];
        smu[bodyIdx] = mu[bodyIdx];
        src[bodyIdx] = (rel && mu[bodyIdx] > 0.0) ? rc[bodyIdx] : 0.0;
        sj2R2[bodyIdx] = (mu[bodyIdx] > 0.0) ? j2R2[bodyIdx] : 0.0;
        sSpinX[bodyIdx] = spinAxis[bodyIdx];
        sSpinY[bodyIdx] = spinAxis[M + bodyIdx];
        sSpinZ[bodyIdx] = spinAxis[2 * M + bodyIdx];
        pIdx = parentIdx[bodyIdx];
        mMask = moonMask[bodyIdx];
    }
    __syncthreads();

    const double bx = validBody ? sx[bodyIdx] : 0.0;
    const double by = validBody ? sy[bodyIdx] : 0.0;
    const double bz = validBody ? sz[bodyIdx] : 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    heavyAccelCoop8(bodyIdx, lane, validBody, bx, by, bz,
                    sx, sy, sz, smu, src, sj2R2, sSpinX, sSpinY, sSpinZ,
                    pIdx, mMask, M, ax, ay, az);
    if (isLeader) {
        st[6 * M + bodyIdx] = ax;
        st[7 * M + bodyIdx] = ay;
        st[8 * M + bodyIdx] = az;
    }
}

__global__ void heavySubstepKernel(double* __restrict__ st, const double* __restrict__ mu,
                                   const double* __restrict__ rc, const double* __restrict__ j2R2,
                                   const double* __restrict__ spinAxis, const int* __restrict__ parentIdx,
                                   const uint64_t* __restrict__ moonMask,
                                   int M, bool rel, double dt, int substeps) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY], src[MAX_HEAVY];
    __shared__ double sj2R2[MAX_HEAVY], sSpinX[MAX_HEAVY], sSpinY[MAX_HEAVY], sSpinZ[MAX_HEAVY];
    __shared__ double sMaxOmega2[MAX_HEAVY];
    __shared__ int sEffectiveSubsteps;

    const int bodyIdx = threadIdx.x / HEAVY_THREADS_PER_BODY;
    const int lane    = threadIdx.x % HEAVY_THREADS_PER_BODY;
    const bool validBody = (bodyIdx < M);
    const bool isLeader  = (validBody && lane == 0);
    const double macroDt = dt * static_cast<double>(substeps);

    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    // Kahan-Babuska compensation registers for double-double precision across sub-steps
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double cvx = 0.0, cvy = 0.0, cvz = 0.0;
    int pIdx = -1;
    uint64_t mMask = 0ULL;

    if (isLeader) {
        x = st[bodyIdx];          y = st[M + bodyIdx];      z = st[2 * M + bodyIdx];
        vx = st[3 * M + bodyIdx]; vy = st[4 * M + bodyIdx]; vz = st[5 * M + bodyIdx];
        ax = st[6 * M + bodyIdx]; ay = st[7 * M + bodyIdx]; az = st[8 * M + bodyIdx];
        sx[bodyIdx] = x;
        sy[bodyIdx] = y;
        sz[bodyIdx] = z;
        smu[bodyIdx] = mu[bodyIdx];
        src[bodyIdx] = (rel && mu[bodyIdx] > 0.0) ? rc[bodyIdx] : 0.0;
        sj2R2[bodyIdx] = (mu[bodyIdx] > 0.0) ? j2R2[bodyIdx] : 0.0;
        sSpinX[bodyIdx] = spinAxis[bodyIdx];
        sSpinY[bodyIdx] = spinAxis[M + bodyIdx];
        sSpinZ[bodyIdx] = spinAxis[2 * M + bodyIdx];
        pIdx = parentIdx[bodyIdx];
        mMask = moonMask[bodyIdx];
    }
    __syncthreads();

    // Instantaneous Courant-Friedrichs-Lewy (CFL) dynamical orbital-frequency subdivision (8-way parallel):
    double localMaxOmega2 = 0.0;
    if (validBody) {
        const double bx = sx[bodyIdx];
        const double by = sy[bodyIdx];
        const double bz = sz[bodyIdx];
        const double mui = smu[bodyIdx];
        for (int j = lane; j < M; j += HEAVY_THREADS_PER_BODY) {
            if (j == bodyIdx) continue;
            const double muPair = mui + smu[j];
            if (muPair <= 0.0) continue;
            const double dx = sx[j] - bx;
            const double dy = sy[j] - by;
            const double dz = sz[j] - bz;
            const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
            const double invR = rsqrt(d2);
            const double omega2 = muPair * invR * invR * invR;
            localMaxOmega2 = fmax(localMaxOmega2, omega2);
        }
    }
    #pragma unroll
    for (int offset = HEAVY_THREADS_PER_BODY / 2; offset > 0; offset >>= 1) {
        const double other = __shfl_down_sync(0xFFFFFFFFu, localMaxOmega2, offset, HEAVY_THREADS_PER_BODY);
        localMaxOmega2 = fmax(localMaxOmega2, other);
    }
    if (isLeader) {
        sMaxOmega2[bodyIdx] = localMaxOmega2;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sysMaxOmega2 = 0.0;
        for (int j = 0; j < M; ++j) {
            sysMaxOmega2 = fmax(sysMaxOmega2, sMaxOmega2[j]);
        }
        const double omegaMax = sqrt(sysMaxOmega2);
        // Symplectic Courant number eta_CFL = 0.50 rad per PEFRL sub-step (~12.6 sub-steps per fastest orbit)
        constexpr double ETA_CFL = 0.50;
        const int cflSubsteps = (omegaMax > 0.0)
            ? static_cast<int>(ceil((omegaMax * fabs(macroDt)) / ETA_CFL))
            : 1;
        sEffectiveSubsteps = max(substeps, max(1, cflSubsteps));
    }
    __syncthreads();

    const int actualSubsteps = sEffectiveSubsteps;
    const double subDt = macroDt / static_cast<double>(actualSubsteps);

    const double dtA1 = PEFRL_XI * subDt;
    const double dtA2 = PEFRL_CHI * subDt;
    const double dtA3 = PEFRL_A3 * subDt;
    const double dtB1 = PEFRL_B1 * subDt;
    const double dtB2 = PEFRL_LAMBDA * subDt;

    auto drift = [&](double dtStage) {
        if (isLeader) {
            kahanAdd(x, cx, vx * dtStage);
            kahanAdd(y, cy, vy * dtStage);
            kahanAdd(z, cz, vz * dtStage);
        }
    };

    auto syncAndKick = [&](double dtKick) {
        __syncthreads();
        if (isLeader) {
            sx[bodyIdx] = x + cx;
            sy[bodyIdx] = y + cy;
            sz[bodyIdx] = z + cz;
        }
        __syncthreads();
        const double bx = validBody ? sx[bodyIdx] : 0.0;
        const double by = validBody ? sy[bodyIdx] : 0.0;
        const double bz = validBody ? sz[bodyIdx] : 0.0;
        heavyAccelCoop8(bodyIdx, lane, validBody, bx, by, bz,
                        sx, sy, sz, smu, src, sj2R2, sSpinX, sSpinY, sSpinZ,
                        pIdx, mMask, M, ax, ay, az);
        if (isLeader) {
            kahanAdd(vx, cvx, ax * dtKick);
            kahanAdd(vy, cvy, ay * dtKick);
            kahanAdd(vz, cvz, az * dtKick);
        }
    };

    for (int k = 0; k < actualSubsteps; ++k) {
        drift(dtA1);
        syncAndKick(dtB1);

        drift(dtA2);
        syncAndKick(dtB2);

        drift(dtA3);
        syncAndKick(dtB2);

        drift(dtA2);
        syncAndKick(dtB1);

        drift(dtA1);
    }

    // Finalize compensated state and compute exact end-of-step acceleration
    __syncthreads();
    if (isLeader) {
        x += cx;   y += cy;   z += cz;
        vx += cvx; vy += cvy; vz += cvz;
        sx[bodyIdx] = x; sy[bodyIdx] = y; sz[bodyIdx] = z;
    }
    __syncthreads();
    const double bx = validBody ? sx[bodyIdx] : 0.0;
    const double by = validBody ? sy[bodyIdx] : 0.0;
    const double bz = validBody ? sz[bodyIdx] : 0.0;
    heavyAccelCoop8(bodyIdx, lane, validBody, bx, by, bz,
                    sx, sy, sz, smu, src, sj2R2, sSpinX, sSpinY, sSpinZ,
                    pIdx, mMask, M, ax, ay, az);
    if (isLeader) {
        st[bodyIdx] = x;          st[M + bodyIdx] = y;      st[2 * M + bodyIdx] = z;
        st[3 * M + bodyIdx] = vx; st[4 * M + bodyIdx] = vy; st[5 * M + bodyIdx] = vz;
        st[6 * M + bodyIdx] = ax; st[7 * M + bodyIdx] = ay; st[8 * M + bodyIdx] = az;
    }
}

// ---------------------------------------------------------------------------
// Test particles: O(N*M), heavy bodies broadcast from shared memory.
// ---------------------------------------------------------------------------
struct HeavyShared {
    double x[MAX_HEAVY], y[MAX_HEAVY], z[MAX_HEAVY];
    double mu[MAX_HEAVY], rc[MAX_HEAVY], j2R2[MAX_HEAVY];
    double spinX[MAX_HEAVY], spinY[MAX_HEAVY], spinZ[MAX_HEAVY];
    float muAU[MAX_HEAVY];
    float rcAU[MAX_HEAVY];
    float j2R2AU[MAX_HEAVY];
    float spinXf[MAX_HEAVY], spinYf[MAX_HEAVY], spinZf[MAX_HEAVY];
};

__device__ __forceinline__ void loadHeavyShared(HeavyShared& s, const double* __restrict__ hs,
                                                const double* __restrict__ mu,
                                                const double* __restrict__ rc,
                                                const double* __restrict__ j2R2,
                                                const double* __restrict__ spinAxis,
                                                int M, bool rel) {
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        s.x[j] = hs[j];
        s.y[j] = hs[M + j];
        s.z[j] = hs[2 * M + j];
        const double m = mu[j];
        const double r = (rel && m > 0.0) ? rc[j] : 0.0;
        const double q = (m > 0.0) ? j2R2[j] : 0.0;
        const double sx = spinAxis[j];
        const double sy = spinAxis[M + j];
        const double sz = spinAxis[2 * M + j];
        s.mu[j] = m;
        s.rc[j] = r;
        s.j2R2[j] = q;
        s.spinX[j] = sx;
        s.spinY[j] = sy;
        s.spinZ[j] = sz;
        s.muAU[j] = static_cast<float>(m * (INV_AU * INV_AU * INV_AU));
        s.rcAU[j] = static_cast<float>(r * INV_AU);
        s.j2R2AU[j] = static_cast<float>(q * (INV_AU * INV_AU));
        s.spinXf[j] = static_cast<float>(sx);
        s.spinYf[j] = static_cast<float>(sy);
        s.spinZf[j] = static_cast<float>(sz);
    }
}

template <bool MIXED>
__device__ __forceinline__ void particleAccel(double x, double y, double z, const HeavyShared& s,
                                              int M, double& ax, double& ay, double& az,
                                              double& maxOmega2) {
    if constexpr (!MIXED) {
        ax = ay = az = 0.0;
        double maxW2 = 0.0;
        #pragma unroll 4
        for (int j = 0; j < M; ++j) {
            const double dx = s.x[j] - x, dy = s.y[j] - y, dz = s.z[j] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = rsqrt(fmax(d2, 1.0));
            const double inv2 = inv * inv;
            const double w2 = s.mu[j] * inv2 * inv;
            maxW2 = fmax(maxW2, w2);
            const double sc = w2 * (1.0 + s.rc[j] * inv);
            ax = fma(sc, dx, ax);
            ay = fma(sc, dy, ay);
            az = fma(sc, dz, az);

            // Sub-mantissa J2 distance gate: evaluate quadrupole only within 0.01 AU (~1.5M km)
            if (d2 < J2_GATE_D2_M2 && s.j2R2[j] > 0.0) {
                const double rDotS = (-dx * s.spinX[j] - dy * s.spinY[j] - dz * s.spinZ[j]) * inv;
                const double baseQ = 1.5 * s.mu[j] * s.j2R2[j] * inv2 * inv2;
                const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                const double pole = -2.0 * rDotS * baseQ;
                ax = fma(coeff, dx, fma(pole, s.spinX[j], ax));
                ay = fma(coeff, dy, fma(pole, s.spinY[j], ay));
                az = fma(coeff, dz, fma(pole, s.spinZ[j], az));
            }
        }
        maxOmega2 = maxW2;
    } else {
        float axN = 0.f, ayN = 0.f, azN = 0.f;
        float axR = 0.f, ayR = 0.f, azR = 0.f;
        float maxW2 = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < M; ++j) {
            const float dx = static_cast<float>((s.x[j] - x) * INV_AU);
            const float dy = static_cast<float>((s.y[j] - y) * INV_AU);
            const float dz = static_cast<float>((s.z[j] - z) * INV_AU);
            const float d2 = fmaxf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)), MIN_D2_AU);
            const float inv = rsqrtf(d2);
            const float inv2 = inv * inv;
            const float s1 = s.muAU[j] * inv2 * inv;
            maxW2 = fmaxf(maxW2, s1);
            const float s2 = s1 * s.rcAU[j] * inv;
            axN = fmaf(s1, dx, axN); ayN = fmaf(s1, dy, ayN); azN = fmaf(s1, dz, azN);
            axR = fmaf(s2, dx, axR); ayR = fmaf(s2, dy, ayR); azR = fmaf(s2, dz, azR);

            // Sub-mantissa J2 distance gate: evaluate quadrupole only within 0.01 AU (~1.5M km)
            if (d2 < J2_GATE_D2_AU2 && s.j2R2AU[j] > 0.0f) {
                const float rDotS = (-dx * s.spinXf[j] - dy * s.spinYf[j] - dz * s.spinZf[j]) * inv;
                const float baseQ = 1.5f * s.muAU[j] * s.j2R2AU[j] * inv2 * inv2;
                const float coeff = baseQ * inv * (1.0f - 5.0f * rDotS * rDotS);
                const float pole = -2.0f * rDotS * baseQ;
                axR = fmaf(coeff, dx, fmaf(pole, s.spinXf[j], axR));
                ayR = fmaf(coeff, dy, fmaf(pole, s.spinYf[j], ayR));
                azR = fmaf(coeff, dz, fmaf(pole, s.spinZf[j], azR));
            }
        }
        ax = (static_cast<double>(axN) + static_cast<double>(axR)) * AU;
        ay = (static_cast<double>(ayN) + static_cast<double>(ayR)) * AU;
        az = (static_cast<double>(azN) + static_cast<double>(azR)) * AU;
        maxOmega2 = static_cast<double>(maxW2);
    }
}

template <bool MIXED>
__global__ void particleInitAccKernel(double* __restrict__ p, size_t N,
                                      const double* __restrict__ hs, const double* __restrict__ mu,
                                      const double* __restrict__ rc, const double* __restrict__ j2R2,
                                      const double* __restrict__ spinAxis, int M, bool rel) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, rc, j2R2, spinAxis, M, rel);
    __syncthreads();
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    double ax, ay, az, dummyW2 = 0.0;
    particleAccel<MIXED>(p[i], p[N + i], p[2 * N + i], s, M, ax, ay, az, dummyW2);
    p[6 * N + i] = ax;
    p[7 * N + i] = ay;
    p[8 * N + i] = az;
}

template <bool MIXED>
__global__ void particleStepKernel(double* __restrict__ p, size_t N,
                                   const double* __restrict__ hs, const double* __restrict__ mu,
                                   const double* __restrict__ rc, const double* __restrict__ j2R2,
                                   const double* __restrict__ spinAxis, int M, bool rel, double dt) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, rc, j2R2, spinAxis, M, rel);
    __syncthreads();
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;

    const double x0 = p[i], y0 = p[N + i], z0 = p[2 * N + i];
    const double vx0 = p[3 * N + i], vy0 = p[4 * N + i], vz0 = p[5 * N + i];
    const double ax0 = p[6 * N + i], ay0 = p[7 * N + i], az0 = p[8 * N + i];

    double x = x0, y = y0, z = z0;
    double vx = vx0, vy = vy0, vz = vz0;
    double ax = ax0, ay = ay0, az = az0;

    const double h = 0.5 * dt;
    vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);
    x = fma(vx, dt, x);  y = fma(vy, dt, y);  z = fma(vz, dt, z);
    double maxOmega2 = 0.0;
    particleAccel<MIXED>(x, y, z, s, M, ax, ay, az, maxOmega2);

    // Optimistic per-particle adaptive CFL close-encounter sub-stepping
    constexpr double ETA_PARTICLE = 0.25;
    const double daL1 = fabs(ax - ax0) + fabs(ay - ay0) + fabs(az - az0);
    const double a0L1 = fabs(ax0) + fabs(ay0) + fabs(az0);
    const bool sharpEncounter = (maxOmega2 * dt * dt > (ETA_PARTICLE * ETA_PARTICLE)) ||
                                (a0L1 > 1e-12 && daL1 > 0.35 * a0L1);
    if (sharpEncounter) {
        // Evaluate start-of-step omega2 as well in case the 1-step drift flung the particle out of periapsis
        double startAx, startAy, startAz, startOmega2 = 0.0;
        particleAccel<MIXED>(x0, y0, z0, s, M, startAx, startAy, startAz, startOmega2);
        const double peakOmega = sqrt(fmax(maxOmega2, startOmega2));
        const int nSub = min(64, max(4, static_cast<int>(ceil((peakOmega * fabs(dt)) / ETA_PARTICLE))));
        const double subDt = dt / static_cast<double>(nSub);
        const double subH  = 0.5 * subDt;
        x = x0; y = y0; z = z0;
        vx = vx0; vy = vy0; vz = vz0;
        ax = startAx; ay = startAy; az = startAz;
        for (int k = 0; k < nSub; ++k) {
            vx = fma(subH, ax, vx); vy = fma(subH, ay, vy); vz = fma(subH, az, vz);
            x  = fma(vx, subDt, x); y  = fma(vy, subDt, y); z  = fma(vz, subDt, z);
            double stepW2 = 0.0;
            particleAccel<MIXED>(x, y, z, s, M, ax, ay, az, stepW2);
            vx = fma(subH, ax, vx); vy = fma(subH, ay, vy); vz = fma(subH, az, vz);
        }
    } else {
        vx = fma(h, ax, vx); vy = fma(h, ay, vy); vz = fma(h, az, vz);
    }

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

                // Inner System histogram [1.90, 5.50] AU (Main Belt Kirkwood gaps, 3:2 Hildas & 1:1 Trojans)
                if (a_AU >= BeltStats::MAIN_MIN_AU && a_AU < BeltStats::MAIN_MAX_AU) {
                    constexpr double span = BeltStats::MAIN_MAX_AU - BeltStats::MAIN_MIN_AU;
                    int bin = static_cast<int>((a_AU - BeltStats::MAIN_MIN_AU) / span * BeltStats::MAIN_BINS);
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
    CUDA_CHECK(cudaEventCreateWithFlags(&m_beltEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&m_timerStart));
    CUDA_CHECK(cudaEventCreate(&m_timerStop));
}

SubstrateEngine::~SubstrateEngine() {
    release();
    cudaEventDestroy(m_timerStop);
    cudaEventDestroy(m_timerStart);
    cudaEventDestroy(m_beltEvent);
    cudaEventDestroy(m_snapEvent);
    cudaStreamDestroy(m_stream);
}

void SubstrateEngine::release() {
    if (m_stream) cudaStreamSynchronize(m_stream);
    cudaFree(d_heavy);      d_heavy = nullptr;
    cudaFree(d_mu);         d_mu = nullptr;
    cudaFree(d_rc);         d_rc = nullptr;
    cudaFree(d_j2R2);       d_j2R2 = nullptr;
    cudaFree(d_spinAxis);   d_spinAxis = nullptr;
    cudaFree(d_parentIdx);  d_parentIdx = nullptr;
    cudaFree(d_moonMask);   d_moonMask = nullptr;
    cudaFree(d_particles);  d_particles = nullptr;
    cudaFree(d_beltBins);   d_beltBins = nullptr;
    cudaFree(d_beltSums);   d_beltSums = nullptr;
    cudaFreeHost(h_snapPinned);     h_snapPinned = nullptr;
    cudaFreeHost(h_beltBinsPinned); h_beltBinsPinned = nullptr;
    cudaFreeHost(h_beltSumsPinned); h_beltSumsPinned = nullptr;
    m_deviceBytes = 0;
    m_snapPending = false;
    m_beltPending = false;
    m_timerPending = false;
}

void SubstrateEngine::refreshCachedAccelerations(bool relativistic) {
    if (m_numHeavy <= 0) return;
    // Exact multiple of 32 covering M * 8 threads (e.g., 55 * 8 = 440 -> 448 threads = 14 warps, <= 1024).
    const int heavyBlockThreads = std::min(1024, ((m_numHeavy * HEAVY_THREADS_PER_BODY + 31) / 32) * 32);
    heavyInitAccKernel<<<1, heavyBlockThreads, 0, m_stream>>>(d_heavy, d_mu, d_rc, d_j2R2, d_spinAxis,
                                                              d_parentIdx, d_moonMask, m_numHeavy, relativistic);
    if (m_numParticles > 0) {
        particleInitAccKernel<false><<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_particles, m_numParticles, d_heavy, d_mu, d_rc, d_j2R2, d_spinAxis, m_numHeavy, relativistic);
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
    m_spinOmega.assign(M, 0.0);
    m_spinAxisX.assign(M, 0.0);
    m_spinAxisY.assign(M, 0.0);
    m_spinAxisZ.assign(M, 1.0);
    m_momentC.assign(M, 0.34);
    m_j2.assign(M, 0.0);
    m_j2R2.assign(M, 0.0);
    m_obliquityDeg.assign(M, 0.0);
    m_tidalLockMode.assign(M, 0);
    m_parentIdx.assign(M, -1);
    m_moonMask.assign(M, 0ULL);
    m_overrideActive.assign(M, false);

    std::vector<double> hs(9 * static_cast<size_t>(M), 0.0);
    std::vector<double> spinSoA(3 * static_cast<size_t>(M), 0.0);
    for (int i = 0; i < M; ++i) {
        const auto& b = heavy[i];
        for (int p = 0; p < M; ++p) {
            if (p != i && heavy[p].name == b.parent) {
                m_parentIdx[i] = p;
                if (p > 0 && i < 64) {
                    m_moonMask[p] |= (1ULL << static_cast<unsigned int>(i));
                }
                break;
            }
        }
        m_mass[i] = b.mass;
        m_radius[i] = b.radiusM;
        m_nodes[i] = b.mass / M_P;
        m_baseDensity[i] = m_nodes[i] / b.vol;
        m_density[i] = m_baseDensity[i];
        const double theta = (m_density[i] >= RHO_GC) ? 1.0 : 0.0;
        m_mu[i] = m_nodes[i] * K_OMEGA * theta;
        m_rc[i] = 3.0 * m_nodes[i] * K_OMEGA / (C_LIGHT * C_LIGHT);

        m_tidalLockMode[i] = b.tidalLockMode;
        m_obliquityDeg[i] = b.obliquityDeg;
        m_momentC[i] = b.momentOfInertiaC;
        const double signRot = (b.spinPeriodHours < 0.0) ? -1.0 : 1.0;
        const double absPeriodSec = std::max(std::abs(b.spinPeriodHours) * 3600.0, 1.0);
        m_spinOmega[i] = signRot * (2.0 * PI_CONST / absPeriodSec);
        m_spinAxisX[i] = b.spinAxis[0];
        m_spinAxisY[i] = b.spinAxis[1];
        m_spinAxisZ[i] = b.spinAxis[2];
        spinSoA[0 * M + i] = b.spinAxis[0];
        spinSoA[1 * M + i] = b.spinAxis[1];
        spinSoA[2 * M + i] = b.spinAxis[2];

        m_j2[i] = deriveHydrostaticJ2(std::abs(m_spinOmega[i]), m_radius[i], m_mu[i], m_momentC[i]);
        m_j2R2[i] = m_j2[i] * m_radius[i] * m_radius[i] * theta;

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
    CUDA_CHECK(cudaMalloc(&d_j2R2, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spinAxis, spinSoA.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_parentIdx, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moonMask, M * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_beltBins, TOTAL_INT_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_beltSums, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_heavy, hs.data(), hs.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, m_mu.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc, m_rc.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_j2R2, m_j2R2.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spinAxis, spinSoA.data(), spinSoA.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parentIdx, m_parentIdx.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moonMask, m_moonMask.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&h_snapPinned, 6 * M * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_beltBinsPinned, TOTAL_INT_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_beltSumsPinned, 2 * sizeof(float)));
    m_deviceBytes = (hs.size() + 7 * M) * sizeof(double);

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
    m_j2[bodyIdx] = deriveHydrostaticJ2(std::abs(m_spinOmega[bodyIdx]), m_radius[bodyIdx], m_mu[bodyIdx], m_momentC[bodyIdx]);
    m_j2R2[bodyIdx] = m_j2[bodyIdx] * m_radius[bodyIdx] * m_radius[bodyIdx] * theta;

    if (m_mu[bodyIdx] != prevMu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu + bodyIdx, &m_mu[bodyIdx], sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_j2R2 + bodyIdx, &m_j2R2[bodyIdx], sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        refreshCachedAccelerations(relativistic);
    }
}

void SubstrateEngine::resetDensityOverrides(bool relativistic) {
    bool anyChanged = false;
    for (int i = 0; i < m_numHeavy; ++i) {
        m_overrideActive[i] = false;
        m_density[i] = m_baseDensity[i];
        const double theta = (m_density[i] >= RHO_GC) ? 1.0 : 0.0;
        const double newMu = m_nodes[i] * K_OMEGA * theta;
        if (newMu != m_mu[i]) {
            m_mu[i] = newMu;
            m_j2[i] = deriveHydrostaticJ2(std::abs(m_spinOmega[i]), m_radius[i], m_mu[i], m_momentC[i]);
            m_j2R2[i] = m_j2[i] * m_radius[i] * m_radius[i] * theta;
            anyChanged = true;
        }
    }
    if (anyChanged && d_mu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu, m_mu.data(), m_numHeavy * sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_j2R2, m_j2R2.data(), m_numHeavy * sizeof(double),
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

void SubstrateEngine::requestBeltStats(double jupiterSemiMajorAU) {
    if (m_numParticles == 0 || !d_particles || !d_beltBins || !h_beltBinsPinned || !h_beltSumsPinned || m_beltPending) {
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(d_beltBins, 0, TOTAL_INT_BINS * sizeof(uint32_t), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_beltSums, 0, 2 * sizeof(float), m_stream));

    beltStatsKernel<<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
        d_particles, m_numParticles, d_heavy, d_mu, m_numHeavy, jupiterSemiMajorAU,
        d_beltBins, d_beltSums);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_beltBinsPinned, d_beltBins, TOTAL_INT_BINS * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_beltSumsPinned, d_beltSums, 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaEventRecord(m_beltEvent, m_stream));
    m_beltPending = true;
}

bool SubstrateEngine::pollBeltStats(BeltStats& out) {
    if (!m_beltPending) return false;
    const cudaError_t q = cudaEventQuery(m_beltEvent);
    if (q == cudaErrorNotReady) return false;
    CUDA_CHECK(q);

    out = BeltStats{};
    for (int i = 0; i < BeltStats::MAIN_BINS; ++i)
        out.mainBeltHist[i] = static_cast<float>(h_beltBinsPinned[i]);
    for (int i = 0; i < BeltStats::KUIPER_BINS; ++i)
        out.kuiperBeltHist[i] = static_cast<float>(h_beltBinsPinned[BeltStats::MAIN_BINS + i]);
    for (int i = 0; i < BeltStats::ECC_BINS; ++i)
        out.eccHist[i] = static_cast<float>(h_beltBinsPinned[BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + i]);

    const int base = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS;
    out.asteroidalCount = h_beltBinsPinned[base + 0];
    out.jfcCount        = h_beltBinsPinned[base + 1];
    out.halleyCount     = h_beltBinsPinned[base + 2];
    out.unboundCount    = h_beltBinsPinned[base + 3];
    out.boundCount      = h_beltBinsPinned[base + 4];
    if (out.boundCount > 0) {
        out.meanEcc    = h_beltSumsPinned[0] / static_cast<float>(out.boundCount);
        out.meanIncDeg = h_beltSumsPinned[1] / static_cast<float>(out.boundCount);
    }

    m_beltPending = false;
    return true;
}

void SubstrateEngine::computeBeltStats(BeltStats& out, double jupiterSemiMajorAU) {
    if (m_beltPending) {
        CUDA_CHECK(cudaEventSynchronize(m_beltEvent));
        pollBeltStats(out);
    }
    out = BeltStats{};
    requestBeltStats(jupiterSemiMajorAU);
    if (m_beltPending) {
        CUDA_CHECK(cudaEventSynchronize(m_beltEvent));
        pollBeltStats(out);
    }
}

void SubstrateEngine::step(const StepParams& p, bool captureSnapshot) {
    const int M = m_numHeavy;
    const int sub = std::max(1, p.substeps);
    // Exact multiple of 32 covering M * 8 threads (e.g., 55 * 8 = 440 -> 448 threads = 14 warps, <= 1024).
    const int heavyBlockThreads = std::min(1024, ((M * HEAVY_THREADS_PER_BODY + 31) / 32) * 32);

    heavySubstepKernel<<<1, heavyBlockThreads, 0, m_stream>>>(d_heavy, d_mu, d_rc, d_j2R2, d_spinAxis,
                                                              d_parentIdx, d_moonMask,
                                                              M, p.relativistic, p.dtSeconds / sub, sub);
    m_simTime += p.dtSeconds;
    if (captureSnapshot) requestSnapshot();

    if (m_numParticles > 0) {
        const unsigned int blocks = blocksFor(m_numParticles, PARTICLE_BLOCK);
        if (p.precision == Precision::Mixed)
            particleStepKernel<true><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_rc, d_j2R2, d_spinAxis,
                M, p.relativistic, p.dtSeconds);
        else
            particleStepKernel<false><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_rc, d_j2R2, d_spinAxis,
                M, p.relativistic, p.dtSeconds);
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

double SubstrateEngine::outerOortStagnationAU() const {
    return gravitationalFloorAU() * C_DELTA_INV_FOURTH_ROOT;
}

double SubstrateEngine::terminationShockRadiusAU() const {
    if (m_numHeavy <= 0 || m_density[0] < RHO_GC) return 0.0;
    return std::sqrt(P_SW_1AU_PA / P_DRAG_FLOOR);
}

double SubstrateEngine::heliopauseRadiusAU() const {
    // Solar wind ram-pressure balance against P_delta = C_delta * Sigma_{0,Ledger} = 0.2834 pPa:
    //   R_TS = 1 AU * sqrt(P_SW(1 AU) / P_delta) = 89.44 AU
    //   R_HP = 1.35 * R_TS = 120.74 AU (collapses to 0 if Sun is decompiled below 4.0 nodes/m^3)
    return HELIOSHEATH_STANDOFF * terminationShockRadiusAU();
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

double SubstrateEngine::newtonianEnergy(const HeavySnapshot& s, bool relativistic) const {
    const int M = m_numHeavy;
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* v = &s.vel[3 * i];
        ke += 0.5 * m_mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double rc_i = (m_mu[i] > 0.0) ? m_rc[i] : 0.0;
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const double dx = s.pos[3 * j] - s.pos[3 * i];
            const double dy = s.pos[3 * j + 1] - s.pos[3 * i + 1];
            const double dz = s.pos[3 * j + 2] - s.pos[3 * i + 2];
            const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
            pe -= 0.5 * m_mass[i] * m_mu[j] / r;
            if (relativistic && m_mu[j] > 0.0) {
                const double rc_j = m_rc[j];
                // Pairwise 1PN potential U_1PN = -m_i * mu_j * (rc_i + rc_j) / (2 * r^2);
                // factor 0.25 accounts for double-counting (i,j) and (j,i) in the full double sum.
                pe -= 0.25 * m_mass[i] * m_mu[j] * (rc_i + rc_j) / (r * r);
            }
        }
        const int p = m_parentIdx[i];
        if (p > 0 && m_j2R2[p] > 0.0 && m_mu[p] > 0.0 && m_mu[i] > 0.0) {
            const double dx = s.pos[3 * p] - s.pos[3 * i];
            const double dy = s.pos[3 * p + 1] - s.pos[3 * i + 1];
            const double dz = s.pos[3 * p + 2] - s.pos[3 * i + 2];
            const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
            const double inv = 1.0 / r;
            const double rDotS = (-dx * m_spinAxisX[p] - dy * m_spinAxisY[p] - dz * m_spinAxisZ[p]) * inv;
            pe += 0.5 * m_mass[i] * m_mu[p] * m_j2R2[p] * inv * inv * inv * (3.0 * rDotS * rDotS - 1.0);
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

        // Add axial spin angular momentum S_i = C_i * M_i * R_i^2 * |omega_i| * s_hat_i
        const double spinMag = m_momentC[i] * m * m_radius[i] * m_radius[i] * std::abs(m_spinOmega[i]);
        Lx += spinMag * m_spinAxisX[i];
        Ly += spinMag * m_spinAxisY[i];
        Lz += spinMag * m_spinAxisZ[i];
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
