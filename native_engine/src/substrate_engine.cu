// Substrate Logistics CUDA kernels.
//
// Physics:
//   g_Omega      = N_j * K_Omega * Theta(rho_N_j - 4.0) / R^2
//   1PN EIH      = Full velocity-dependent Einstein-Infeld-Hoffmann 1PN in mu_j = N_j*K_Omega*Theta and c
//   Test 1PN     = Exact harmonic-gauge Schwarzschild 1PN geodesic acceleration
// Integration:
//   - Heavy bodies: 4th-order Omelyan PEFRL symplectic composition with Kahan-Babuska-Neumaier
//     compensated summation and 1-pass time-reversible symmetric midpoint EIH kicks.
//   - Test particles: 2nd-order symplectic Velocity Verlet (FP64 or Dual-Accumulator Mixed).

#include "substrate_engine.h"
#include "substrate_constants.h"

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                                   \
    do {                                                                                   \
        cudaError_t _e = (expr);                                                           \
        if (_e != cudaSuccess)                                                             \
            throw std::runtime_error(std::string(#expr " failed: ") + cudaGetErrorString(_e)); \
    } while (0)

namespace substrate {
namespace {

constexpr int PARTICLE_BLOCK         = 256;
constexpr int MAX_RING_HIST          = 512;
constexpr int MAX_RING_PERT          = 12;
// Frame layout per substep k: slot 0 = Saturn spin pole, slot 1 = Saturn world pos, slots 2.. = perturber world pos
constexpr int RING_FRAME_STRIDE      = (2 + MAX_RING_PERT) * 3;
constexpr double RING_MAX_SUBDT      = 1150.0; // ETA_RING ~ 0.35 rad (~19 steps/orbit for inner C-ring, ~48 for A-ring)
constexpr double INV_AU = 1.0 / AU;
constexpr double INV_C  = 1.0 / C_LIGHT;
constexpr double INV_C2 = 1.0 / (C_LIGHT * C_LIGHT);
constexpr float MIN_D2_AU = static_cast<float>((1.0 / AU) * (1.0 / AU));  // 1 m clamp, in AU^2
constexpr double J2_GATE_D2_M2 = (0.01 * AU) * (0.01 * AU);               // 0.01 AU (~1.5M km) sub-mantissa particle J2 gate
constexpr float J2_GATE_D2_AU2 = 1.0e-4f;                                 // (0.01 AU)^2 in AU^2
constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

// Total integer bins: MAIN_BINS + KUIPER_BINS + ECC_BINS + 5 regime counters
constexpr int TOTAL_INT_BINS = BeltStats::MAIN_BINS + BeltStats::KUIPER_BINS + BeltStats::ECC_BINS + 5;
// Total Saturn ring integer bins: RING_BINS + boundCount + escapedCount
constexpr int TOTAL_RING_INT_BINS = RingStats::RING_BINS + 2;

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

// Full 53-bit IEEE-754 FP64 reciprocal square root (< 2 ULP across [1e-35, 1e35])
// using the full-rate FP32 SFU rsqrtf + 2 FP64 Newton-Raphson iterations.
__device__ __forceinline__ double fastRsqrt64(double x) {
    if (x > 1.0e-35 && x < 1.0e35) {
        double y = static_cast<double>(rsqrtf(static_cast<float>(x)));
        const double halfX = 0.5 * x;
        y = y * fma(-halfX, y * y, 1.5);
        y = y * fma(-halfX, y * y, 1.5);
        return y;
    }
    return rsqrt(x);
}

// Exact Lie-group Rodrigues rotation on S^2: preserves |s| == 1 and U_J2 identically.
__device__ __noinline__ void rodriguesSpinUpdate(double& sx, double& sy, double& sz,
                                                 double wx, double wy, double wz, double dt) {
    const double w2 = fma(wx, wx, fma(wy, wy, wz * wz));
    if (w2 <= 1e-36) return;
    const double w = sqrt(w2);
    const double ang = w * dt;
    if (fabs(ang) <= 1e-18) return;
    const double invW = 1.0 / w;
    const double ux = wx * invW, uy = wy * invW, uz = wz * invW;
    const double ca = cos(ang), sa = sin(ang);
    const double oneMinusCa = 1.0 - ca;
    const double uDotS = fma(ux, sx, fma(uy, sy, uz * sz));
    // u x s
    const double cx = uy * sz - uz * sy;
    const double cy = uz * sx - ux * sz;
    const double cz = ux * sy - uy * sx;
    const double nx = sx * ca + cx * sa + ux * uDotS * oneMinusCa;
    const double ny = sy * ca + cy * sa + uy * uDotS * oneMinusCa;
    const double nz = sz * ca + cz * sa + uz * uDotS * oneMinusCa;
    const double invN = fastRsqrt64(fma(nx, nx, fma(ny, ny, nz * nz)));
    sx = nx * invN;
    sy = ny * invN;
    sz = nz * invN;
}

// ---------------------------------------------------------------------------
// Heavy bodies: single-thread-per-body FP64 force helpers (init & 1-block fallback).
// Sub-pass 1: Pure Newtonian monopole aN + scalar potential phi + exact 4-case J2 & spin torque.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void heavyNewtonianAndJ2Single(int bodyIdx,
                                                          double x, double y, double z,
                                                          const double* sx, const double* sy, const double* sz,
                                                          const double* smu, const double* sj2R2,
                                                          const double* sSpinX, const double* sSpinY, const double* sSpinZ,
                                                          const double* sPrecCoeff,
                                                          int parentIdx, uint64_t moonMask, uint64_t oblateSunMask,
                                                          int M,
                                                          double& aNx, double& aNy, double& aNz, double& phiOut,
                                                          double& aJ2x, double& aJ2y, double& aJ2z,
                                                          double& precWx, double& precWy, double& precWz) {
    double lx = 0.0, ly = 0.0, lz = 0.0, phi = 0.0;
    #pragma unroll 4
    for (int j = 0; j < M; ++j) {
        if (j == bodyIdx) continue;
        const double mj = smu[j];
        if (mj <= 0.0) continue;
        const double dx = sx[j] - x, dy = sy[j] - y, dz = sz[j] - z;
        const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
        const double inv = fastRsqrt64(d2);
        const double s = mj * inv * inv * inv;
        lx = fma(s, dx, lx);
        ly = fma(s, dy, ly);
        lz = fma(s, dz, lz);
        phi = fma(mj, inv, phi);
    }

    double jx = 0.0, jy = 0.0, jz = 0.0;
    double wx = 0.0, wy = 0.0, wz = 0.0;

    // Case 1: Moon feeling parent's J2 quadrupole acceleration
    if (parentIdx > 0) {
        const double j2r2_p = sj2R2[parentIdx];
        const double mp = smu[parentIdx];
        if (j2r2_p > 0.0 && mp > 0.0) {
            const double dx = sx[parentIdx] - x, dy = sy[parentIdx] - y, dz = sz[parentIdx] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = fastRsqrt64(fmax(d2, 1.0));
            const double inv2 = inv * inv;
            const double spX = sSpinX[parentIdx], spY = sSpinY[parentIdx], spZ = sSpinZ[parentIdx];
            const double rDotS = (-dx * spX - dy * spY - dz * spZ) * inv;
            const double baseQ = 1.5 * mp * j2r2_p * inv2 * inv2;
            const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
            const double pole  = -2.0 * rDotS * baseQ;
            jx = fma(coeff, dx, fma(pole, spX, jx));
            jy = fma(coeff, dy, fma(pole, spY, jy));
            jz = fma(coeff, dz, fma(pole, spZ, jz));
        }
    }

    // Case 2: Oblate planet feeling exact Newton's 3rd Law J2 reaction + lunar precession torque from its moons
    if (moonMask != 0ULL) {
        const double j2r2_i = sj2R2[bodyIdx];
        if (j2r2_i > 0.0 && smu[bodyIdx] > 0.0) {
            const double spX = sSpinX[bodyIdx], spY = sSpinY[bodyIdx], spZ = sSpinZ[bodyIdx];
            const double pCoeff = sPrecCoeff[bodyIdx];
            for (unsigned long long mask = moonMask; mask != 0ULL; mask &= (mask - 1ULL)) {
                const int m = __ffsll(static_cast<long long>(mask)) - 1;
                const double mm = smu[m];
                if (mm <= 0.0) continue;
                const double dx = sx[m] - x, dy = sy[m] - y, dz = sz[m] - z;
                const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                const double inv = fastRsqrt64(fmax(d2, 1.0));
                const double inv2 = inv * inv;
                const double rDotS = (dx * spX + dy * spY + dz * spZ) * inv;
                const double baseQ = 1.5 * mm * j2r2_i * inv2 * inv2;
                const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                const double pole  = 2.0 * rDotS * baseQ;
                jx = fma(coeff, dx, fma(pole, spX, jx));
                jy = fma(coeff, dy, fma(pole, spY, jy));
                jz = fma(coeff, dz, fma(pole, spZ, jz));

                if (pCoeff > 0.0) {
                    const double pScale = pCoeff * mm * inv2 * inv * rDotS * inv;
                    wx = fma(pScale, dx, wx);
                    wy = fma(pScale, dy, wy);
                    wz = fma(pScale, dz, wz);
                }
            }
        }
    }

    // Case 3: Oblate planet feeling solar J2 reaction force + solar gyroscopic precession torque from Sun (j = 0)
    if (parentIdx == 0 && sj2R2[bodyIdx] > 0.0 && smu[bodyIdx] > 0.0 && smu[0] > 0.0) {
        const double mSun = smu[0];
        const double j2r2_i = sj2R2[bodyIdx];
        const double spX = sSpinX[bodyIdx], spY = sSpinY[bodyIdx], spZ = sSpinZ[bodyIdx];
        const double dx = sx[0] - x, dy = sy[0] - y, dz = sz[0] - z;
        const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
        const double inv = fastRsqrt64(fmax(d2, 1.0));
        const double inv2 = inv * inv;
        const double rDotS = (dx * spX + dy * spY + dz * spZ) * inv;
        const double baseQ = 1.5 * mSun * j2r2_i * inv2 * inv2;
        const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
        const double pole  = 2.0 * rDotS * baseQ;
        jx = fma(coeff, dx, fma(pole, spX, jx));
        jy = fma(coeff, dy, fma(pole, spY, jy));
        jz = fma(coeff, dz, fma(pole, spZ, jz));

        const double pCoeff = sPrecCoeff[bodyIdx];
        if (pCoeff > 0.0) {
            const double pScale = pCoeff * mSun * inv2 * inv * rDotS * inv;
            wx = fma(pScale, dx, wx);
            wy = fma(pScale, dy, wy);
            wz = fma(pScale, dz, wz);
        }
    }

    // Case 4: Exact Newton's 3rd Law J2 reaction on Sun (bodyIdx == 0) from all oblate heliocentric bodies
    if (bodyIdx == 0 && oblateSunMask != 0ULL && smu[0] > 0.0) {
        for (unsigned long long mask = oblateSunMask; mask != 0ULL; mask &= (mask - 1ULL)) {
            const int k = __ffsll(static_cast<long long>(mask)) - 1;
            const double mk = smu[k];
            const double j2r2_k = sj2R2[k];
            if (mk <= 0.0 || j2r2_k <= 0.0) continue;
            const double dx = sx[k] - x, dy = sy[k] - y, dz = sz[k] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = fastRsqrt64(fmax(d2, 1.0));
            const double inv2 = inv * inv;
            const double spX = sSpinX[k], spY = sSpinY[k], spZ = sSpinZ[k];
            const double rDotS = (-dx * spX - dy * spY - dz * spZ) * inv;
            const double baseQ = 1.5 * mk * j2r2_k * inv2 * inv2;
            const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
            const double pole  = -2.0 * rDotS * baseQ;
            jx = fma(coeff, dx, fma(pole, spX, jx));
            jy = fma(coeff, dy, fma(pole, spY, jy));
            jz = fma(coeff, dz, fma(pole, spZ, jz));
        }
    }

    aNx = lx; aNy = ly; aNz = lz; phiOut = phi;
    aJ2x = jx; aJ2y = jy; aJ2z = jz;
    precWx = wx; precWy = wy; precWz = wz;
}

// Sub-pass 2 (single-thread-per-body): Exact velocity-dependent Einstein-Infeld-Hoffmann 1PN acceleration.
__device__ __forceinline__ void heavyEih1PNSingle(int bodyIdx,
                                                  double myX, double myY, double myZ,
                                                  const double* sx, const double* sy, const double* sz,
                                                  const double* smu,
                                                  const double* sEihAx, const double* sEihAy, const double* sEihAz,
                                                  const double* sEihPhi,
                                                  const double* sEihVx, const double* sEihVy, const double* sEihVz,
                                                  int M, double& a1x, double& a1y, double& a1z) {
    const double myPhi = sEihPhi[bodyIdx];
    const double myVx = sEihVx[bodyIdx], myVy = sEihVy[bodyIdx], myVz = sEihVz[bodyIdx];
    const double myV2 = fma(myVx, myVx, fma(myVy, myVy, myVz * myVz));
    double lx = 0.0, ly = 0.0, lz = 0.0;
    #pragma unroll 4
    for (int j = 0; j < M; ++j) {
        if (j == bodyIdx) continue;
        const double mj = smu[j];
        if (mj <= 0.0) continue;
        const double dx = sx[j] - myX, dy = sy[j] - myY, dz = sz[j] - myZ; // r_j - r_i
        const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
        const double inv = fastRsqrt64(d2);
        const double inv3 = inv * inv * inv;

        const double vjx = sEihVx[j], vjy = sEihVy[j], vjz = sEihVz[j];
        const double aNjx = sEihAx[j], aNjy = sEihAy[j], aNjz = sEihAz[j];
        const double vj2 = fma(vjx, vjx, fma(vjy, vjy, vjz * vjz));
        const double viDotVj = fma(myVx, vjx, fma(myVy, vjy, myVz * vjz));
        const double dDotVj = fma(dx, vjx, fma(dy, vjy, dz * vjz)) * inv;
        const double dDotAnj = fma(dx, aNjx, fma(dy, aNjy, dz * aNjz));

        const double bracket = -4.0 * myPhi - sEihPhi[j] + myV2 + 2.0 * vj2
                               - 4.0 * viDotVj - 1.5 * dDotVj * dDotVj + 0.5 * dDotAnj;
        const double wvx = 4.0 * myVx - 3.0 * vjx;
        const double wvy = 4.0 * myVy - 3.0 * vjy;
        const double wvz = 4.0 * myVz - 3.0 * vjz;
        const double velDot = -fma(dx, wvx, fma(dy, wvy, dz * wvz));

        const double base1PN = mj * inv3 * INV_C2;
        const double c1 = base1PN * bracket;
        const double c2 = base1PN * velDot;
        const double c3 = (3.5 * INV_C2) * mj * inv;

        lx = fma(c1, dx, fma(c2, myVx - vjx, fma(c3, aNjx, lx)));
        ly = fma(c1, dy, fma(c2, myVy - vjy, fma(c3, aNjy, ly)));
        lz = fma(c1, dz, fma(c2, myVz - vjz, fma(c3, aNjz, lz)));
    }
    a1x = lx; a1y = ly; a1z = lz;
}

__global__ void heavyInitAccKernel(double* __restrict__ st, const double* __restrict__ mu,
                                   const double* __restrict__ j2R2,
                                   const double* __restrict__ spinAxis, const double* __restrict__ spinPrecCoeff,
                                   const int* __restrict__ parentIdx, const uint64_t* __restrict__ moonMask,
                                   uint64_t oblateSunMask, int M, bool rel) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY];
    __shared__ double sj2R2[MAX_HEAVY], sSpinX[MAX_HEAVY], sSpinY[MAX_HEAVY], sSpinZ[MAX_HEAVY];
    __shared__ double sPrecCoeff[MAX_HEAVY];
    __shared__ double sEihAx[MAX_HEAVY], sEihAy[MAX_HEAVY], sEihAz[MAX_HEAVY], sEihPhi[MAX_HEAVY];
    __shared__ double sEihVx[MAX_HEAVY], sEihVy[MAX_HEAVY], sEihVz[MAX_HEAVY];

    const int i = threadIdx.x;
    const bool active = (i < M);
    int pIdx = -1;
    uint64_t mMask = 0ULL;
    if (active) {
        sx[i] = st[i];
        sy[i] = st[M + i];
        sz[i] = st[2 * M + i];
        sEihVx[i] = st[3 * M + i];
        sEihVy[i] = st[4 * M + i];
        sEihVz[i] = st[5 * M + i];
        smu[i] = mu[i];
        sj2R2[i] = (mu[i] > 0.0) ? j2R2[i] : 0.0;
        sSpinX[i] = spinAxis[i];
        sSpinY[i] = spinAxis[M + i];
        sSpinZ[i] = spinAxis[2 * M + i];
        sPrecCoeff[i] = (mu[i] > 0.0) ? spinPrecCoeff[i] : 0.0;
        pIdx = parentIdx[i];
        mMask = moonMask[i];
    }
    __syncthreads();

    double aTotNx = 0.0, aTotNy = 0.0, aTotNz = 0.0;
    if (active) {
        double aNx = 0.0, aNy = 0.0, aNz = 0.0, phi = 0.0;
        double aJ2x = 0.0, aJ2y = 0.0, aJ2z = 0.0;
        double wx = 0.0, wy = 0.0, wz = 0.0;
        heavyNewtonianAndJ2Single(i, sx[i], sy[i], sz[i],
                                  sx, sy, sz, smu, sj2R2, sSpinX, sSpinY, sSpinZ, sPrecCoeff,
                                  pIdx, mMask, oblateSunMask, M,
                                  aNx, aNy, aNz, phi, aJ2x, aJ2y, aJ2z, wx, wy, wz);
        aTotNx = aNx + aJ2x;
        aTotNy = aNy + aJ2y;
        aTotNz = aNz + aJ2z;
        if (rel) {
            sEihAx[i] = aNx; sEihAy[i] = aNy; sEihAz[i] = aNz;
            sEihPhi[i] = phi;
        }
    }

    if (rel) {
        __syncthreads();
        if (active) {
            double a1x = 0.0, a1y = 0.0, a1z = 0.0;
            heavyEih1PNSingle(i, sx[i], sy[i], sz[i], sx, sy, sz, smu,
                              sEihAx, sEihAy, sEihAz, sEihPhi, sEihVx, sEihVy, sEihVz,
                              M, a1x, a1y, a1z);
            aTotNx += a1x;
            aTotNy += a1y;
            aTotNz += a1z;
        }
    }

    if (active) {
        st[6 * M + i] = aTotNx;
        st[7 * M + i] = aTotNy;
        st[8 * M + i] = aTotNz;
    }
}

// 1-block single-thread-per-body fallback kernel (used automatically if cooperative launch is unavailable)
__global__ void heavySubstepFallbackKernel(double* __restrict__ st, const double* __restrict__ mu,
                                           const double* __restrict__ j2R2,
                                           double* __restrict__ spinAxis, const double* __restrict__ spinPrecCoeff,
                                           const int* __restrict__ parentIdx, const uint64_t* __restrict__ moonMask,
                                           uint64_t oblateSunMask,
                                           int saturnIdx, const int* __restrict__ bodyRingSlot,
                                           double* __restrict__ ringHist, int* __restrict__ ringActualSubsteps,
                                           int M, bool rel, double dt, int substeps) {
    __shared__ double sx[MAX_HEAVY], sy[MAX_HEAVY], sz[MAX_HEAVY], smu[MAX_HEAVY];
    __shared__ double sj2R2[MAX_HEAVY], sSpinX[MAX_HEAVY], sSpinY[MAX_HEAVY], sSpinZ[MAX_HEAVY];
    __shared__ double sPrecCoeff[MAX_HEAVY];
    __shared__ double sEihAx[MAX_HEAVY], sEihAy[MAX_HEAVY], sEihAz[MAX_HEAVY], sEihPhi[MAX_HEAVY];
    __shared__ double sEihVx[MAX_HEAVY], sEihVy[MAX_HEAVY], sEihVz[MAX_HEAVY];
    __shared__ double sMaxOmega2[MAX_HEAVY];
    __shared__ int sEffectiveSubsteps;

    const int i = threadIdx.x;
    const bool active = (i < M);
    const double macroDt = dt * static_cast<double>(substeps);

    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double spX = 0.0, spY = 0.0, spZ = 1.0;
    double precCoeffLocal = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double cvx = 0.0, cvy = 0.0, cvz = 0.0;
    int pIdx = -1;
    int ringSlot = -1;
    uint64_t mMask = 0ULL;

    if (active) {
        x = st[i];          y = st[M + i];      z = st[2 * M + i];
        vx = st[3 * M + i]; vy = st[4 * M + i]; vz = st[5 * M + i];
        ax = st[6 * M + i]; ay = st[7 * M + i]; az = st[8 * M + i];
        spX = spinAxis[i];
        spY = spinAxis[M + i];
        spZ = spinAxis[2 * M + i];
        precCoeffLocal = (mu[i] > 0.0) ? spinPrecCoeff[i] : 0.0;
        sx[i] = x; sy[i] = y; sz[i] = z;
        smu[i] = mu[i];
        sj2R2[i] = (mu[i] > 0.0) ? j2R2[i] : 0.0;
        sSpinX[i] = spX; sSpinY[i] = spY; sSpinZ[i] = spZ;
        sPrecCoeff[i] = precCoeffLocal;
        pIdx = parentIdx[i];
        mMask = moonMask[i];
        ringSlot = (bodyRingSlot != nullptr) ? bodyRingSlot[i] : -1;
    }
    __syncthreads();

    double localMaxOmega2 = 0.0;
    if (active) {
        const double mui = smu[i];
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const double muPair = mui + smu[j];
            if (muPair <= 0.0) continue;
            const double dx = sx[j] - x, dy = sy[j] - y, dz = sz[j] - z;
            const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
            const double invR = fastRsqrt64(d2);
            localMaxOmega2 = fmax(localMaxOmega2, muPair * invR * invR * invR);
        }
        sMaxOmega2[i] = localMaxOmega2;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sysMaxOmega2 = 0.0;
        for (int j = 0; j < M; ++j) sysMaxOmega2 = fmax(sysMaxOmega2, sMaxOmega2[j]);
        const double omegaMax = sqrt(sysMaxOmega2);
        constexpr double ETA_CFL = 0.50;
        const int cflSubsteps = (omegaMax > 0.0)
            ? static_cast<int>(ceil((omegaMax * fabs(macroDt)) / ETA_CFL))
            : 1;
        sEffectiveSubsteps = max(substeps, max(1, cflSubsteps));
        if (ringActualSubsteps != nullptr) *ringActualSubsteps = sEffectiveSubsteps;
    }
    __syncthreads();

    const int actualSubsteps = sEffectiveSubsteps;
    const double subDt = macroDt / static_cast<double>(actualSubsteps);
    const double dtA1 = PEFRL_XI * subDt;
    const double dtA2 = PEFRL_CHI * subDt;
    const double dtA3 = PEFRL_A3 * subDt;
    const double dtB1 = PEFRL_B1 * subDt;
    const double dtB2 = PEFRL_LAMBDA * subDt;

    auto recordRingFrame = [&](int frameIdx, double wx, double wy, double wz) {
        if (ringHist != nullptr && active) {
            const int kHist = min(frameIdx, MAX_RING_HIST - 1);
            double* dst = ringHist + kHist * RING_FRAME_STRIDE;
            if (i == saturnIdx) {
                dst[0] = spX; dst[1] = spY; dst[2] = spZ;
                dst[3] = wx;  dst[4] = wy;  dst[5] = wz;
            }
            if (ringSlot >= 0) {
                dst[(2 + ringSlot) * 3 + 0] = wx;
                dst[(2 + ringSlot) * 3 + 1] = wy;
                dst[(2 + ringSlot) * 3 + 2] = wz;
            }
        }
    };

    recordRingFrame(0, x, y, z);

    auto drift = [&](double dtStage) {
        if (active) {
            kahanAdd(x, cx, vx * dtStage);
            kahanAdd(y, cy, vy * dtStage);
            kahanAdd(z, cz, vz * dtStage);
        }
    };

    auto syncAndKick = [&](double dtKick, bool applyKick) {
        __syncthreads();
        if (active) {
            sx[i] = x + cx; sy[i] = y + cy; sz[i] = z + cz;
            sSpinX[i] = spX; sSpinY[i] = spY; sSpinZ[i] = spZ;
        }
        __syncthreads();

        double aTotNx = 0.0, aTotNy = 0.0, aTotNz = 0.0;
        double precWx = 0.0, precWy = 0.0, precWz = 0.0;
        if (active) {
            double aNx = 0.0, aNy = 0.0, aNz = 0.0, phi = 0.0;
            double aJ2x = 0.0, aJ2y = 0.0, aJ2z = 0.0;
            heavyNewtonianAndJ2Single(i, sx[i], sy[i], sz[i],
                                      sx, sy, sz, smu, sj2R2, sSpinX, sSpinY, sSpinZ, sPrecCoeff,
                                      pIdx, mMask, oblateSunMask, M,
                                      aNx, aNy, aNz, phi, aJ2x, aJ2y, aJ2z, precWx, precWy, precWz);
            aTotNx = aNx + aJ2x;
            aTotNy = aNy + aJ2y;
            aTotNz = aNz + aJ2z;
            if (rel) {
                const double halfDt = applyKick ? (0.5 * dtKick) : 0.0;
                sEihAx[i] = aNx; sEihAy[i] = aNy; sEihAz[i] = aNz;
                sEihPhi[i] = phi;
                sEihVx[i] = (vx + cvx) + halfDt * aTotNx;
                sEihVy[i] = (vy + cvy) + halfDt * aTotNy;
                sEihVz[i] = (vz + cvz) + halfDt * aTotNz;
            }
        }

        if (rel) {
            __syncthreads();
            if (active) {
                double a1x = 0.0, a1y = 0.0, a1z = 0.0;
                heavyEih1PNSingle(i, sx[i], sy[i], sz[i], sx, sy, sz, smu,
                                  sEihAx, sEihAy, sEihAz, sEihPhi, sEihVx, sEihVy, sEihVz,
                                  M, a1x, a1y, a1z);
                aTotNx += a1x;
                aTotNy += a1y;
                aTotNz += a1z;
            }
        }

        if (active) {
            ax = aTotNx; ay = aTotNy; az = aTotNz;
            if (applyKick) {
                kahanAdd(vx, cvx, ax * dtKick);
                kahanAdd(vy, cvy, ay * dtKick);
                kahanAdd(vz, cvz, az * dtKick);
                if (precCoeffLocal > 0.0) {
                    rodriguesSpinUpdate(spX, spY, spZ, precWx, precWy, precWz, dtKick);
                }
            }
        }
    };

    const double driftDts[4] = {dtA1, dtA2, dtA3, dtA2};
    const double kickDts[4]  = {dtB1, dtB2, dtB2, dtB1};
    for (int k = 0; k < actualSubsteps; ++k) {
        #pragma unroll 1
        for (int stg = 0; stg < 4; ++stg) {
            drift(driftDts[stg]);
            syncAndKick(kickDts[stg], true);
        }
        drift(dtA1);
        recordRingFrame(k + 1, x + cx, y + cy, z + cz);
    }

    if (active) {
        x += cx;   y += cy;   z += cz;
        vx += cvx; vy += cvy; vz += cvz;
        cx = cy = cz = cvx = cvy = cvz = 0.0;
    }
    syncAndKick(0.0, false);
    if (active) {
        st[i] = x;          st[M + i] = y;      st[2 * M + i] = z;
        st[3 * M + i] = vx; st[4 * M + i] = vy; st[5 * M + i] = vz;
        st[6 * M + i] = ax; st[7 * M + i] = ay; st[8 * M + i] = az;
        spinAxis[i] = spX; spinAxis[M + i] = spY; spinAxis[2 * M + i] = spZ;
    }
}

// ---------------------------------------------------------------------------
// Cooperative Multi-SM Heavy Kernel: gridDim = M blocks, blockDim = 32 (1 warp/body across M SMs).
// L2-resident volatile double-buffered ping-pong + 2-stage symmetric midpoint EIH 1PN kick.
// ---------------------------------------------------------------------------
__global__ void heavySubstepCoopKernel(double* __restrict__ st,
                                       volatile double* pingPong,
                                       volatile double* eihScratch,
                                       volatile double* cflOmega2,
                                       const double* __restrict__ mu,
                                       const double* __restrict__ j2R2,
                                       double* __restrict__ spinAxis,
                                       const double* __restrict__ spinPrecCoeff,
                                       const int* __restrict__ parentIdx,
                                       const uint64_t* __restrict__ moonMask,
                                       uint64_t oblateSunMask,
                                       int saturnIdx,
                                       const int* __restrict__ bodyRingSlot,
                                       double* __restrict__ ringHist,
                                       int* __restrict__ ringActualSubsteps,
                                       int M, bool rel, double dt, int substeps) {
    cg::grid_group grid = cg::this_grid();
    const int bodyIdx  = blockIdx.x;
    const int lane     = threadIdx.x;
    const bool isLeader = (lane == 0);
    const double macroDt = dt * static_cast<double>(substeps);

    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double spX = 0.0, spY = 0.0, spZ = 1.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double cvx = 0.0, cvy = 0.0, cvz = 0.0;

    const double muSelf = mu[bodyIdx];
    const double j2r2Self = (muSelf > 0.0) ? j2R2[bodyIdx] : 0.0;
    const double precCoeffLocal = (muSelf > 0.0) ? spinPrecCoeff[bodyIdx] : 0.0;
    const int pIdx = parentIdx[bodyIdx];
    const uint64_t mMask = moonMask[bodyIdx];
    const int ringSlot = (bodyRingSlot != nullptr) ? bodyRingSlot[bodyIdx] : -1;

    // L2-resident volatile SoA views into d_eihScratch (7 * MAX_HEAVY)
    volatile double* eihAx  = eihScratch + 0 * MAX_HEAVY;
    volatile double* eihAy  = eihScratch + 1 * MAX_HEAVY;
    volatile double* eihAz  = eihScratch + 2 * MAX_HEAVY;
    volatile double* eihPhi = eihScratch + 3 * MAX_HEAVY;
    volatile double* eihVx  = eihScratch + 4 * MAX_HEAVY;
    volatile double* eihVy  = eihScratch + 5 * MAX_HEAVY;
    volatile double* eihVz  = eihScratch + 6 * MAX_HEAVY;

    auto writeBuf = [&](int buf, double wx, double wy, double wz, double sx, double sy, double sz) {
        if (isLeader) {
            volatile double* base = pingPong + buf * (6 * MAX_HEAVY);
            base[0 * MAX_HEAVY + bodyIdx] = wx;
            base[1 * MAX_HEAVY + bodyIdx] = wy;
            base[2 * MAX_HEAVY + bodyIdx] = wz;
            base[3 * MAX_HEAVY + bodyIdx] = sx;
            base[4 * MAX_HEAVY + bodyIdx] = sy;
            base[5 * MAX_HEAVY + bodyIdx] = sz;
        }
    };

    auto recordRingFrame = [&](int frameIdx, double wx, double wy, double wz) {
        if (ringHist != nullptr && isLeader) {
            const int kHist = min(frameIdx, MAX_RING_HIST - 1);
            double* dst = ringHist + kHist * RING_FRAME_STRIDE;
            if (bodyIdx == saturnIdx) {
                dst[0] = spX; dst[1] = spY; dst[2] = spZ;
                dst[3] = wx;  dst[4] = wy;  dst[5] = wz;
            }
            if (ringSlot >= 0) {
                dst[(2 + ringSlot) * 3 + 0] = wx;
                dst[(2 + ringSlot) * 3 + 1] = wy;
                dst[(2 + ringSlot) * 3 + 2] = wz;
            }
        }
    };

    // Step A: Load state, write initial buf 0 & ringHist[0], compute cooperative CFL subdivision
    if (isLeader) {
        x = st[bodyIdx];          y = st[M + bodyIdx];      z = st[2 * M + bodyIdx];
        vx = st[3 * M + bodyIdx]; vy = st[4 * M + bodyIdx]; vz = st[5 * M + bodyIdx];
        spX = spinAxis[bodyIdx];  spY = spinAxis[M + bodyIdx]; spZ = spinAxis[2 * M + bodyIdx];
    }
    writeBuf(0, x, y, z, spX, spY, spZ);
    recordRingFrame(0, x, y, z);
    grid.sync();

    // 32-way parallel CFL frequency scan per body
    {
        volatile double* px = pingPong;
        volatile double* py = pingPong + MAX_HEAVY;
        volatile double* pz = pingPong + 2 * MAX_HEAVY;
        const double rx = px[bodyIdx], ry = py[bodyIdx], rz = pz[bodyIdx];
        double localMaxOmega2 = 0.0;
        for (int j = lane; j < M; j += 32) {
            if (j == bodyIdx) continue;
            const double muPair = muSelf + mu[j];
            if (muPair <= 0.0) continue;
            const double dx = px[j] - rx, dy = py[j] - ry, dz = pz[j] - rz;
            const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
            const double invR = fastRsqrt64(d2);
            localMaxOmega2 = fmax(localMaxOmega2, muPair * invR * invR * invR);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            localMaxOmega2 = fmax(localMaxOmega2, __shfl_down_sync(0xFFFFFFFFu, localMaxOmega2, offset, 32));
        }
        if (isLeader) cflOmega2[bodyIdx] = localMaxOmega2;
    }
    grid.sync();

    // Every warp reduces cflOmega2 and broadcasts exact sysMaxOmega2 to all 32 lanes
    double sysMaxOmega2 = 0.0;
    for (int j = lane; j < M; j += 32) {
        sysMaxOmega2 = fmax(sysMaxOmega2, cflOmega2[j]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sysMaxOmega2 = fmax(sysMaxOmega2, __shfl_down_sync(0xFFFFFFFFu, sysMaxOmega2, offset, 32));
    }
    sysMaxOmega2 = __shfl_sync(0xFFFFFFFFu, sysMaxOmega2, 0, 32);

    const double omegaMax = sqrt(sysMaxOmega2);
    constexpr double ETA_CFL = 0.50;
    const int cflSubsteps = (omegaMax > 0.0)
        ? static_cast<int>(ceil((omegaMax * fabs(macroDt)) / ETA_CFL))
        : 1;
    const int actualSubsteps = max(substeps, max(1, cflSubsteps));
    if (bodyIdx == 0 && isLeader && ringActualSubsteps != nullptr) {
        *ringActualSubsteps = actualSubsteps;
    }

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

    // 2-Stage Symmetric Midpoint EIH Kick (or pure Newtonian+J2 fast path when !rel)
    auto coopEvalAndKick = [&](int readBuf, double dtKick, bool applyKick) {
        volatile double* pBase = pingPong + readBuf * (6 * MAX_HEAVY);
        volatile double* bx  = pBase;
        volatile double* by  = pBase + MAX_HEAVY;
        volatile double* bz  = pBase + 2 * MAX_HEAVY;
        volatile double* psx = pBase + 3 * MAX_HEAVY;
        volatile double* psy = pBase + 4 * MAX_HEAVY;
        volatile double* psz = pBase + 5 * MAX_HEAVY;

        const double myX = bx[bodyIdx], myY = by[bodyIdx], myZ = bz[bodyIdx];

        // Sub-pass 1: Pure Newtonian monopole aN + scalar potential phi + 32-way Case 4 Sun J2
        double aNx = 0.0, aNy = 0.0, aNz = 0.0, phiSum = 0.0;
        double aJ2x = 0.0, aJ2y = 0.0, aJ2z = 0.0;

        for (int j = lane; j < M; j += 32) {
            if (j == bodyIdx) continue;
            const double mj = mu[j];
            if (mj <= 0.0) continue;
            const double dx = bx[j] - myX, dy = by[j] - myY, dz = bz[j] - myZ;
            const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
            const double inv = fastRsqrt64(d2);
            const double s = mj * inv * inv * inv;
            aNx = fma(s, dx, aNx);
            aNy = fma(s, dy, aNy);
            aNz = fma(s, dz, aNz);
            phiSum = fma(mj, inv, phiSum);

            // Case 4 (32-way warp-parallel): Sun J2 reaction from oblate heliocentric bodies
            if (bodyIdx == 0 && j < 64 && ((oblateSunMask >> static_cast<unsigned int>(j)) & 1ULL) != 0ULL) {
                const double j2r2_k = j2R2[j];
                if (j2r2_k > 0.0 && muSelf > 0.0) {
                    const double inv2 = inv * inv;
                    const double kspX = psx[j], kspY = psy[j], kspZ = psz[j];
                    const double rDotS = (-dx * kspX - dy * kspY - dz * kspZ) * inv;
                    const double baseQ = 1.5 * mj * j2r2_k * inv2 * inv2;
                    const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                    const double pole  = -2.0 * rDotS * baseQ;
                    aJ2x = fma(coeff, dx, fma(pole, kspX, aJ2x));
                    aJ2y = fma(coeff, dy, fma(pole, kspY, aJ2y));
                    aJ2z = fma(coeff, dz, fma(pole, kspZ, aJ2z));
                }
            }
        }

        // Unconditional 5-step warp sum-reduction across all 32 lanes
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            aNx    += __shfl_down_sync(0xFFFFFFFFu, aNx,    offset, 32);
            aNy    += __shfl_down_sync(0xFFFFFFFFu, aNy,    offset, 32);
            aNz    += __shfl_down_sync(0xFFFFFFFFu, aNz,    offset, 32);
            phiSum += __shfl_down_sync(0xFFFFFFFFu, phiSum, offset, 32);
            aJ2x   += __shfl_down_sync(0xFFFFFFFFu, aJ2x,   offset, 32);
            aJ2y   += __shfl_down_sync(0xFFFFFFFFu, aJ2y,   offset, 32);
            aJ2z   += __shfl_down_sync(0xFFFFFFFFu, aJ2z,   offset, 32);
        }

        double aTotNx = 0.0, aTotNy = 0.0, aTotNz = 0.0;
        double precWx = 0.0, precWy = 0.0, precWz = 0.0;

        if (isLeader) {
            // Case 1: Moon feeling parent's J2 quadrupole
            if (pIdx > 0) {
                const double mp = mu[pIdx];
                const double j2r2_p = (mp > 0.0) ? j2R2[pIdx] : 0.0;
                if (j2r2_p > 0.0 && mp > 0.0) {
                    const double dx = bx[pIdx] - myX, dy = by[pIdx] - myY, dz = bz[pIdx] - myZ;
                    const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                    const double inv = fastRsqrt64(fmax(d2, 1.0));
                    const double inv2 = inv * inv;
                    const double pspX = psx[pIdx], pspY = psy[pIdx], pspZ = psz[pIdx];
                    const double rDotS = (-dx * pspX - dy * pspY - dz * pspZ) * inv;
                    const double baseQ = 1.5 * mp * j2r2_p * inv2 * inv2;
                    const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                    const double pole  = -2.0 * rDotS * baseQ;
                    aJ2x = fma(coeff, dx, fma(pole, pspX, aJ2x));
                    aJ2y = fma(coeff, dy, fma(pole, pspY, aJ2y));
                    aJ2z = fma(coeff, dz, fma(pole, pspZ, aJ2z));
                }
            }

            // Case 2: Oblate planet feeling reaction + lunar precession torque from its moons
            if (mMask != 0ULL && j2r2Self > 0.0 && muSelf > 0.0) {
                const double bspX = psx[bodyIdx], bspY = psy[bodyIdx], bspZ = psz[bodyIdx];
                for (unsigned long long mask = mMask; mask != 0ULL; mask &= (mask - 1ULL)) {
                    const int m = __ffsll(static_cast<long long>(mask)) - 1;
                    const double mm = mu[m];
                    if (mm <= 0.0) continue;
                    const double dx = bx[m] - myX, dy = by[m] - myY, dz = bz[m] - myZ;
                    const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                    const double inv = fastRsqrt64(fmax(d2, 1.0));
                    const double inv2 = inv * inv;
                    const double rDotS = (dx * bspX + dy * bspY + dz * bspZ) * inv;
                    const double baseQ = 1.5 * mm * j2r2Self * inv2 * inv2;
                    const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                    const double pole  = 2.0 * rDotS * baseQ;
                    aJ2x = fma(coeff, dx, fma(pole, bspX, aJ2x));
                    aJ2y = fma(coeff, dy, fma(pole, bspY, aJ2y));
                    aJ2z = fma(coeff, dz, fma(pole, bspZ, aJ2z));
                    if (precCoeffLocal > 0.0) {
                        const double pScale = precCoeffLocal * mm * inv2 * inv * rDotS * inv;
                        precWx = fma(pScale, dx, precWx);
                        precWy = fma(pScale, dy, precWy);
                        precWz = fma(pScale, dz, precWz);
                    }
                }
            }

            // Case 3: Oblate planet feeling solar J2 reaction + solar precession torque from Sun (j = 0)
            if (pIdx == 0 && j2r2Self > 0.0 && muSelf > 0.0) {
                const double mSun = mu[0];
                if (mSun > 0.0) {
                    const double bspX = psx[bodyIdx], bspY = psy[bodyIdx], bspZ = psz[bodyIdx];
                    const double dx = bx[0] - myX, dy = by[0] - myY, dz = bz[0] - myZ;
                    const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
                    const double inv = fastRsqrt64(fmax(d2, 1.0));
                    const double inv2 = inv * inv;
                    const double rDotS = (dx * bspX + dy * bspY + dz * bspZ) * inv;
                    const double baseQ = 1.5 * mSun * j2r2Self * inv2 * inv2;
                    const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                    const double pole  = 2.0 * rDotS * baseQ;
                    aJ2x = fma(coeff, dx, fma(pole, bspX, aJ2x));
                    aJ2y = fma(coeff, dy, fma(pole, bspY, aJ2y));
                    aJ2z = fma(coeff, dz, fma(pole, bspZ, aJ2z));
                    if (precCoeffLocal > 0.0) {
                        const double pScale = precCoeffLocal * mSun * inv2 * inv * rDotS * inv;
                        precWx = fma(pScale, dx, precWx);
                        precWy = fma(pScale, dy, precWy);
                        precWz = fma(pScale, dz, precWz);
                    }
                }
            }

            aTotNx = aNx + aJ2x;
            aTotNy = aNy + aJ2y;
            aTotNz = aNz + aJ2z;

            if (rel) {
                // Symmetric midpoint velocity v_mid = v + 0.5 * dtKick * (a_N + a_J2)
                const double halfDt = applyKick ? (0.5 * dtKick) : 0.0;
                eihAx[bodyIdx]  = aNx;   eihAy[bodyIdx]  = aNy;   eihAz[bodyIdx]  = aNz;
                eihPhi[bodyIdx] = phiSum;
                eihVx[bodyIdx]  = (vx + cvx) + halfDt * aTotNx;
                eihVy[bodyIdx]  = (vy + cvy) + halfDt * aTotNy;
                eihVz[bodyIdx]  = (vz + cvz) + halfDt * aTotNz;
            }
        }

        // Sub-pass 2: Full velocity-dependent EIH 1PN acceleration (skipped completely if !rel)
        if (rel) {
            grid.sync();

            const double myPhi = eihPhi[bodyIdx];
            const double myVx = eihVx[bodyIdx], myVy = eihVy[bodyIdx], myVz = eihVz[bodyIdx];
            const double myV2 = fma(myVx, myVx, fma(myVy, myVy, myVz * myVz));
            double a1x = 0.0, a1y = 0.0, a1z = 0.0;

            for (int j = lane; j < M; j += 32) {
                if (j == bodyIdx) continue;
                const double mj = mu[j];
                if (mj <= 0.0) continue;
                const double dx = bx[j] - myX, dy = by[j] - myY, dz = bz[j] - myZ; // r_j - r_i
                const double d2 = fmax(fma(dx, dx, fma(dy, dy, dz * dz)), 1.0);
                const double inv = fastRsqrt64(d2);
                const double inv3 = inv * inv * inv;

                const double vjx = eihVx[j], vjy = eihVy[j], vjz = eihVz[j];
                const double aNjx = eihAx[j], aNjy = eihAy[j], aNjz = eihAz[j];
                const double vj2 = fma(vjx, vjx, fma(vjy, vjy, vjz * vjz));
                const double viDotVj = fma(myVx, vjx, fma(myVy, vjy, myVz * vjz));
                const double dDotVj = fma(dx, vjx, fma(dy, vjy, dz * vjz)) * inv;
                const double dDotAnj = fma(dx, aNjx, fma(dy, aNjy, dz * aNjz));

                const double bracket = -4.0 * myPhi - eihPhi[j] + myV2 + 2.0 * vj2
                                       - 4.0 * viDotVj - 1.5 * dDotVj * dDotVj + 0.5 * dDotAnj;
                // ((r_i - r_j) . (4 v_i - 3 v_j)) = (-d) . (4 v_i - 3 v_j)
                const double wvx = 4.0 * myVx - 3.0 * vjx;
                const double wvy = 4.0 * myVy - 3.0 * vjy;
                const double wvz = 4.0 * myVz - 3.0 * vjz;
                const double velDot = -fma(dx, wvx, fma(dy, wvy, dz * wvz));

                const double base1PN = mj * inv3 * INV_C2;
                const double c1 = base1PN * bracket;
                const double c2 = base1PN * velDot;
                const double c3 = (3.5 * INV_C2) * mj * inv;

                a1x = fma(c1, dx, fma(c2, myVx - vjx, fma(c3, aNjx, a1x)));
                a1y = fma(c1, dy, fma(c2, myVy - vjy, fma(c3, aNjy, a1y)));
                a1z = fma(c1, dz, fma(c2, myVz - vjz, fma(c3, aNjz, a1z)));
            }

            // Unconditional 5-step warp sum-reduction across all 32 lanes
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                a1x += __shfl_down_sync(0xFFFFFFFFu, a1x, offset, 32);
                a1y += __shfl_down_sync(0xFFFFFFFFu, a1y, offset, 32);
                a1z += __shfl_down_sync(0xFFFFFFFFu, a1z, offset, 32);
            }

            if (isLeader) {
                aTotNx += a1x;
                aTotNy += a1y;
                aTotNz += a1z;
            }
        }

        if (isLeader) {
            ax = aTotNx; ay = aTotNy; az = aTotNz;
            if (applyKick) {
                kahanAdd(vx, cvx, ax * dtKick);
                kahanAdd(vy, cvy, ay * dtKick);
                kahanAdd(vz, cvz, az * dtKick);
                if (precCoeffLocal > 0.0) {
                    rodriguesSpinUpdate(spX, spY, spZ, precWx, precWy, precWz, dtKick);
                }
            }
        }
    };

    // Initial Drift(dtA1) -> write buf 0 -> grid.sync()
    drift(dtA1);
    writeBuf(0, x + cx, y + cy, z + cz, spX, spY, spZ);
    grid.sync();

    // Step B: Double-buffered 4-stage PEFRL loop
    for (int k = 0; k < actualSubsteps; ++k) {
        // Stage 1: read buf 0 -> kick(dtB1) -> drift(dtA2) -> write buf 1
        coopEvalAndKick(0, dtB1, true);
        drift(dtA2);
        writeBuf(1, x + cx, y + cy, z + cz, spX, spY, spZ);
        grid.sync();

        // Stage 2: read buf 1 -> kick(dtB2) -> drift(dtA3) -> write buf 0
        coopEvalAndKick(1, dtB2, true);
        drift(dtA3);
        writeBuf(0, x + cx, y + cy, z + cz, spX, spY, spZ);
        grid.sync();

        // Stage 3: read buf 0 -> kick(dtB2) -> drift(dtA2) -> write buf 1
        coopEvalAndKick(0, dtB2, true);
        drift(dtA2);
        writeBuf(1, x + cx, y + cy, z + cz, spX, spY, spZ);
        grid.sync();

        // Stage 4: read buf 1 -> kick(dtB1) -> drift(dtA1) -> exact substep boundary t_{k+1}
        coopEvalAndKick(1, dtB1, true);
        drift(dtA1);
        recordRingFrame(k + 1, x + cx, y + cy, z + cz);

        if (k + 1 < actualSubsteps) {
            // Fused start-of-next-substep drift(dtA1) -> write buf 0 -> grid.sync()
            drift(dtA1);
            writeBuf(0, x + cx, y + cy, z + cz, spX, spY, spZ);
            grid.sync();
        } else {
            if (isLeader) {
                x += cx;   y += cy;   z += cz;
                vx += cvx; vy += cvy; vz += cvz;
                cx = cy = cz = cvx = cvy = cvz = 0.0;
            }
            writeBuf(0, x, y, z, spX, spY, spZ);
            grid.sync();
        }
    }

    // Step C: Final end-of-step acceleration & writeback (vMid = v)
    coopEvalAndKick(0, 0.0, false);
    if (isLeader) {
        st[bodyIdx] = x;          st[M + bodyIdx] = y;      st[2 * M + bodyIdx] = z;
        st[3 * M + bodyIdx] = vx; st[4 * M + bodyIdx] = vy; st[5 * M + bodyIdx] = vz;
        st[6 * M + bodyIdx] = ax; st[7 * M + bodyIdx] = ay; st[8 * M + bodyIdx] = az;
        spinAxis[bodyIdx] = spX;
        spinAxis[M + bodyIdx] = spY;
        spinAxis[2 * M + bodyIdx] = spZ;
    }
}

// ---------------------------------------------------------------------------
// Test particles: O(N*M), heavy bodies broadcast from shared memory.
// Uses exact velocity-dependent harmonic-gauge Schwarzschild 1PN geodesic acceleration.
// ---------------------------------------------------------------------------
struct HeavyShared {
    double x[MAX_HEAVY], y[MAX_HEAVY], z[MAX_HEAVY];
    double vx[MAX_HEAVY], vy[MAX_HEAVY], vz[MAX_HEAVY];
    double mu[MAX_HEAVY], rg[MAX_HEAVY], j2R2[MAX_HEAVY];
    double spinX[MAX_HEAVY], spinY[MAX_HEAVY], spinZ[MAX_HEAVY];
    float xAUf[MAX_HEAVY], yAUf[MAX_HEAVY], zAUf[MAX_HEAVY];
    float betaXf[MAX_HEAVY], betaYf[MAX_HEAVY], betaZf[MAX_HEAVY];
    float muAU[MAX_HEAVY];
    float rgAUf[MAX_HEAVY];
    float j2R2AU[MAX_HEAVY];
    float spinXf[MAX_HEAVY], spinYf[MAX_HEAVY], spinZf[MAX_HEAVY];
};

__device__ __forceinline__ void loadHeavyShared(HeavyShared& s, const double* __restrict__ hs,
                                                const double* __restrict__ mu,
                                                const double* __restrict__ j2R2,
                                                const double* __restrict__ spinAxis,
                                                int M, bool rel) {
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        const double hx = hs[j];
        const double hy = hs[M + j];
        const double hz = hs[2 * M + j];
        const double hvx = hs[3 * M + j];
        const double hvy = hs[4 * M + j];
        const double hvz = hs[5 * M + j];
        s.x[j] = hx;
        s.y[j] = hy;
        s.z[j] = hz;
        s.vx[j] = hvx;
        s.vy[j] = hvy;
        s.vz[j] = hvz;
        s.xAUf[j] = static_cast<float>(hx * INV_AU);
        s.yAUf[j] = static_cast<float>(hy * INV_AU);
        s.zAUf[j] = static_cast<float>(hz * INV_AU);
        s.betaXf[j] = static_cast<float>(hvx * INV_C);
        s.betaYf[j] = static_cast<float>(hvy * INV_C);
        s.betaZf[j] = static_cast<float>(hvz * INV_C);
        const double m = mu[j];
        const double r = (rel && m > 0.0) ? (m * INV_C2) : 0.0;
        const double q = (m > 0.0) ? j2R2[j] : 0.0;
        const double sx = spinAxis[j];
        const double sy = spinAxis[M + j];
        const double sz = spinAxis[2 * M + j];
        s.mu[j] = m;
        s.rg[j] = r;
        s.j2R2[j] = q;
        s.spinX[j] = sx;
        s.spinY[j] = sy;
        s.spinZ[j] = sz;
        s.muAU[j] = static_cast<float>(m * (INV_AU * INV_AU * INV_AU));
        s.rgAUf[j] = static_cast<float>(r * INV_AU);
        s.j2R2AU[j] = static_cast<float>(q * (INV_AU * INV_AU));
        s.spinXf[j] = static_cast<float>(sx);
        s.spinYf[j] = static_cast<float>(sy);
        s.spinZf[j] = static_cast<float>(sz);
    }
}

template <bool MIXED>
__device__ __forceinline__ void particleAccel(double x, double y, double z,
                                              double vx, double vy, double vz,
                                              const HeavyShared& s, int M, bool rel,
                                              double& ax, double& ay, double& az,
                                              double& maxOmega2) {
    if constexpr (!MIXED) {
        double axN = 0.0, ayN = 0.0, azN = 0.0;
        double maxW2 = 0.0;
        #pragma unroll 4
        for (int j = 0; j < M; ++j) {
            const double mj = s.mu[j];
            if (mj <= 0.0) continue;
            const double dx = s.x[j] - x, dy = s.y[j] - y, dz = s.z[j] - z;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = fastRsqrt64(fmax(d2, 1.0));
            const double inv2 = inv * inv;
            const double w2 = mj * inv2 * inv;
            maxW2 = fmax(maxW2, w2);

            if (rel) {
                // Exact velocity-dependent harmonic-gauge Schwarzschild 1PN geodesic acceleration
                const double vrx = vx - s.vx[j], vry = vy - s.vy[j], vrz = vz - s.vz[j];
                const double vr2 = fma(vrx, vrx, fma(vry, vry, vrz * vrz));
                const double cRad = 1.0 + vr2 * INV_C2 - 4.0 * s.rg[j] * inv;
                const double cVel = (-4.0 * INV_C2) * fma(dx, vrx, fma(dy, vry, dz * vrz));
                axN = fma(w2 * cRad, dx, fma(w2 * cVel, vrx, axN));
                ayN = fma(w2 * cRad, dy, fma(w2 * cVel, vry, ayN));
                azN = fma(w2 * cRad, dz, fma(w2 * cVel, vrz, azN));
            } else {
                axN = fma(w2, dx, axN);
                ayN = fma(w2, dy, ayN);
                azN = fma(w2, dz, azN);
            }

            // Sub-mantissa J2 distance gate: evaluate quadrupole only within 0.01 AU (~1.5M km)
            if (d2 < J2_GATE_D2_M2 && s.j2R2[j] > 0.0) {
                const double rDotS = (-dx * s.spinX[j] - dy * s.spinY[j] - dz * s.spinZ[j]) * inv;
                const double baseQ = 1.5 * mj * s.j2R2[j] * inv2 * inv2;
                const double coeff = baseQ * inv * (1.0 - 5.0 * rDotS * rDotS);
                const double pole = -2.0 * rDotS * baseQ;
                axN = fma(coeff, dx, fma(pole, s.spinX[j], axN));
                ayN = fma(coeff, dy, fma(pole, s.spinY[j], ayN));
                azN = fma(coeff, dz, fma(pole, s.spinZ[j], azN));
            }
        }
        ax = axN; ay = ayN; az = azN;
        maxOmega2 = maxW2;
    } else {
        // Pure-FP32 inner loop: convert particle position (AU) & velocity (v/c) once outside the j loop
        const float pxAU = static_cast<float>(x * INV_AU);
        const float pyAU = static_cast<float>(y * INV_AU);
        const float pzAU = static_cast<float>(z * INV_AU);
        const float bix  = static_cast<float>(vx * INV_C);
        const float biy  = static_cast<float>(vy * INV_C);
        const float biz  = static_cast<float>(vz * INV_C);
        float axN = 0.f, ayN = 0.f, azN = 0.f;
        float axR = 0.f, ayR = 0.f, azR = 0.f;
        float maxW2 = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < M; ++j) {
            const float dx = s.xAUf[j] - pxAU;
            const float dy = s.yAUf[j] - pyAU;
            const float dz = s.zAUf[j] - pzAU;
            const float d2 = fmaxf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)), MIN_D2_AU);
            const float inv = rsqrtf(d2);
            const float inv2 = inv * inv;
            const float s1 = s.muAU[j] * inv2 * inv;
            maxW2 = fmaxf(maxW2, s1);
            axN = fmaf(s1, dx, axN);
            ayN = fmaf(s1, dy, ayN);
            azN = fmaf(s1, dz, azN);

            if (rel && s.rgAUf[j] > 0.0f) {
                const float brx = bix - s.betaXf[j];
                const float bry = biy - s.betaYf[j];
                const float brz = biz - s.betaZf[j];
                const float br2 = fmaf(brx, brx, fmaf(bry, bry, brz * brz));
                const float cRad = br2 - 4.0f * s.rgAUf[j] * inv;
                const float cVel = -4.0f * fmaf(dx, brx, fmaf(dy, bry, dz * brz));
                axR = fmaf(s1 * cRad, dx, fmaf(s1 * cVel, brx, axR));
                ayR = fmaf(s1 * cRad, dy, fmaf(s1 * cVel, bry, ayR));
                azR = fmaf(s1 * cRad, dz, fmaf(s1 * cVel, brz, azR));
            }

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
                                      const double* __restrict__ j2R2,
                                      const double* __restrict__ spinAxis, int M, bool rel) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, j2R2, spinAxis, M, rel);
    __syncthreads();
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    double ax, ay, az, dummyW2 = 0.0;
    particleAccel<MIXED>(p[i], p[N + i], p[2 * N + i],
                         p[3 * N + i], p[4 * N + i], p[5 * N + i],
                         s, M, rel, ax, ay, az, dummyW2);
    p[6 * N + i] = ax;
    p[7 * N + i] = ay;
    p[8 * N + i] = az;
}

template <bool MIXED>
__global__ void particleStepKernel(double* __restrict__ p, size_t N,
                                   const double* __restrict__ hs, const double* __restrict__ mu,
                                   const double* __restrict__ j2R2,
                                   const double* __restrict__ spinAxis, int M, bool rel, double dt) {
    __shared__ HeavyShared s;
    loadHeavyShared(s, hs, mu, j2R2, spinAxis, M, rel);
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
    // vx, vy, vz is already the exact symmetric midpoint velocity v_{n+1/2}
    particleAccel<MIXED>(x, y, z, vx, vy, vz, s, M, rel, ax, ay, az, maxOmega2);

    // Optimistic per-particle adaptive CFL close-encounter sub-stepping (exact IEEE FP64 during flybys!)
    constexpr double ETA_PARTICLE = 0.25;
    const double daL1 = fabs(ax - ax0) + fabs(ay - ay0) + fabs(az - az0);
    const double a0L1 = fabs(ax0) + fabs(ay0) + fabs(az0);
    const bool sharpEncounter = (maxOmega2 * dt * dt > (ETA_PARTICLE * ETA_PARTICLE)) ||
                                (a0L1 > 1e-12 && daL1 > 0.35 * a0L1);
    if (sharpEncounter) {
        double startAx, startAy, startAz, startOmega2 = 0.0;
        particleAccel<false>(x0, y0, z0, vx0, vy0, vz0, s, M, rel, startAx, startAy, startAz, startOmega2);
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
            particleAccel<false>(x, y, z, vx, vy, vz, s, M, rel, ax, ay, az, stepW2);
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
// Dedicated Passive Saturnian Ring System (sub-stepped in lockstep with Saturn + moons)
// ---------------------------------------------------------------------------
template <bool MIXED>
__device__ __forceinline__ void ringAccel(double rx, double ry, double rz,
                                          double muSat, double j2R2Sat,
                                          double rSatM, double invRSat,
                                          float muSatN, float j2SatF,
                                          const double* sCurSpin, const float* sCurSpinF,
                                          const double* sPertPos, const float* sPertPosN,
                                          const double* sPertMu, const float* sPertMuN,
                                          const double* sSumInd, const float* sSumIndN,
                                          int numPert,
                                          double& ax, double& ay, double& az) {
    if constexpr (!MIXED) {
        // Full IEEE FP64 force evaluation with fastRsqrt64 and separate Saturn/tidal accumulators
        double axSat = 0.0, aySat = 0.0, azSat = 0.0;
        if (muSat > 0.0) {
            const double d2 = fma(rx, rx, fma(ry, ry, rz * rz));
            const double inv = fastRsqrt64(fmax(d2, 1.0));
            const double inv2 = inv * inv;
            const double sc = -muSat * inv2 * inv;
            axSat = sc * rx;
            aySat = sc * ry;
            azSat = sc * rz;

            if (j2R2Sat > 0.0) {
                const double spX = sCurSpin[0], spY = sCurSpin[1], spZ = sCurSpin[2];
                const double rDotS = fma(rx, spX, fma(ry, spY, rz * spZ)) * inv;
                const double baseQ = 1.5 * muSat * j2R2Sat * inv2 * inv2;
                const double coeff = baseQ * inv * (5.0 * rDotS * rDotS - 1.0);
                const double pole  = -2.0 * rDotS * baseQ;
                axSat = fma(coeff, rx, fma(pole, spX, axSat));
                aySat = fma(coeff, ry, fma(pole, spY, aySat));
                azSat = fma(coeff, rz, fma(pole, spZ, azSat));
            }
        }

        // Direct accelerations from Saturn's moons & Sun added to precomputed net indirect tidal term
        double axT = sSumInd[0], ayT = sSumInd[1], azT = sSumInd[2];
        #pragma unroll 4
        for (int p = 0; p < numPert; ++p) {
            const double mp = sPertMu[p];
            if (mp <= 0.0) continue;
            const double dx = sPertPos[3 * p + 0] - rx;
            const double dy = sPertPos[3 * p + 1] - ry;
            const double dz = sPertPos[3 * p + 2] - rz;
            const double d2 = fma(dx, dx, fma(dy, dy, dz * dz));
            const double inv = fastRsqrt64(fmax(d2, 1.0));
            const double sc = mp * inv * inv * inv;
            axT = fma(sc, dx, axT);
            ayT = fma(sc, dy, ayT);
            azT = fma(sc, dz, azT);
        }
        ax = axSat + axT;
        ay = aySat + ayT;
        az = azSat + azT;
    } else {
        // Pure-FP32 Saturn-radius-normalized inner loop; separate tidal accumulator prevents Saturn's
        // large monopole from swamping Sun direct/indirect tidal cancellation in FP32.
        const float rxf = static_cast<float>(rx * invRSat);
        const float ryf = static_cast<float>(ry * invRSat);
        const float rzf = static_cast<float>(rz * invRSat);

        float axSat = 0.0f, aySat = 0.0f, azSat = 0.0f;
        if (muSatN > 0.0f) {
            const float d2 = fmaf(rxf, rxf, fmaf(ryf, ryf, rzf * rzf));
            const float inv = rsqrtf(fmaxf(d2, 1.0e-15f));
            const float inv2 = inv * inv;
            const float sc = -muSatN * inv2 * inv;
            axSat = sc * rxf;
            aySat = sc * ryf;
            azSat = sc * rzf;

            if (j2SatF > 0.0f) {
                const float spX = sCurSpinF[0], spY = sCurSpinF[1], spZ = sCurSpinF[2];
                const float rDotS = fmaf(rxf, spX, fmaf(ryf, spY, rzf * spZ)) * inv;
                const float baseQ = 1.5f * muSatN * j2SatF * inv2 * inv2;
                const float coeff = baseQ * inv * fmaf(5.0f, rDotS * rDotS, -1.0f);
                const float pole  = -2.0f * rDotS * baseQ;
                axSat = fmaf(coeff, rxf, fmaf(pole, spX, axSat));
                aySat = fmaf(coeff, ryf, fmaf(pole, spY, aySat));
                azSat = fmaf(coeff, rzf, fmaf(pole, spZ, azSat));
            }
        }

        float axT = sSumIndN[0], ayT = sSumIndN[1], azT = sSumIndN[2];
        #pragma unroll 4
        for (int p = 0; p < numPert; ++p) {
            const float mp = sPertMuN[p];
            if (mp <= 0.0f) continue;
            const float dx = sPertPosN[3 * p + 0] - rxf;
            const float dy = sPertPosN[3 * p + 1] - ryf;
            const float dz = sPertPosN[3 * p + 2] - rzf;
            const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            const float inv = rsqrtf(fmaxf(d2, 1.0e-15f));
            const float sc = mp * inv * inv * inv;
            axT = fmaf(sc, dx, axT);
            ayT = fmaf(sc, dy, ayT);
            azT = fmaf(sc, dz, azT);
        }

        ax = (static_cast<double>(axSat) + static_cast<double>(axT)) * rSatM;
        ay = (static_cast<double>(aySat) + static_cast<double>(ayT)) * rSatM;
        az = (static_cast<double>(azSat) + static_cast<double>(azT)) * rSatM;
    }
}

template <bool MIXED>
__global__ void ringStepKernel(double* __restrict__ rp, size_t N,
                               const double* __restrict__ mu, const double* __restrict__ j2R2,
                               int saturnIdx, const int* __restrict__ ringPertIdx, int numRingPert,
                               const double* __restrict__ ringHist, const int* __restrict__ ringActualSubsteps,
                               double macroDt, double rSatM) {
    __shared__ double sFrame0[RING_FRAME_STRIDE];
    __shared__ double sFrame1[RING_FRAME_STRIDE];
    __shared__ double sPertMu[MAX_RING_PERT];
    __shared__ float  sPertMuN[MAX_RING_PERT];
    __shared__ double sCurSpin[3];
    __shared__ float  sCurSpinF[3];
    __shared__ double sCurPertPos[MAX_RING_PERT * 3];
    __shared__ float  sCurPertPosN[MAX_RING_PERT * 3];
    __shared__ double sSumInd[3];
    __shared__ float  sSumIndN[3];
    __shared__ double sMuSat, sJ2R2Sat;
    __shared__ float  sMuSatN, sJ2SatF;
    __shared__ int sActualSubsteps;

    const double safeRSat = fmax(rSatM, 1.0);
    const double invRSat = 1.0 / safeRSat;
    const double invRSat3 = invRSat * invRSat * invRSat;

    if (threadIdx.x == 0) {
        sMuSat = mu[saturnIdx];
        sJ2R2Sat = (sMuSat > 0.0) ? j2R2[saturnIdx] : 0.0;
        sMuSatN = static_cast<float>(sMuSat * invRSat3);
        sJ2SatF = static_cast<float>(sJ2R2Sat * invRSat * invRSat);
        sActualSubsteps = max(1, *ringActualSubsteps);
    }
    if (threadIdx.x < numRingPert) {
        const double mp = mu[ringPertIdx[threadIdx.x]];
        sPertMu[threadIdx.x] = mp;
        sPertMuN[threadIdx.x] = static_cast<float>(mp * invRSat3);
    }
    __syncthreads();

    const int actualSubsteps = sActualSubsteps;
    const double muSat = sMuSat;
    const double j2R2Sat = sJ2R2Sat;
    const float muSatN = sMuSatN;
    const float j2SatF = sJ2SatF;
    const double subDt = macroDt / static_cast<double>(actualSubsteps);
    // Guarantee <= RING_MAX_SUBDT micro-steps (~19 steps per inner C-ring orbit, 1 local step at dt=6h, sub=20)
    const int nLocal = max(1, static_cast<int>(ceil(fabs(subDt) / RING_MAX_SUBDT)));
    const double microDt = subDt / static_cast<double>(nLocal);
    const double halfMicro = 0.5 * microDt;

    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const bool active = (i < N);

    double rx = 0.0, ry = 0.0, rz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    if (active) {
        rx = rp[i];          ry = rp[N + i];      rz = rp[2 * N + i];
        vx = rp[3 * N + i];  vy = rp[4 * N + i];  vz = rp[5 * N + i];
    }

    // Warp-0 helper: interpolate relative perturber vectors in FP64 (subtracting Saturn world pos
    // before blending), precompute net indirect tidal acceleration once per micro-step, and publish.
    auto loadFrameState = [&](double oneMinusF, double frac) {
        if (threadIdx.x < 32) {
            const int p = threadIdx.x;
            double indX = 0.0, indY = 0.0, indZ = 0.0;
            if (p < numRingPert) {
                const int off = (2 + p) * 3;
                const double px = oneMinusF * (sFrame0[off + 0] - sFrame0[3]) + frac * (sFrame1[off + 0] - sFrame1[3]);
                const double py = oneMinusF * (sFrame0[off + 1] - sFrame0[4]) + frac * (sFrame1[off + 1] - sFrame1[4]);
                const double pz = oneMinusF * (sFrame0[off + 2] - sFrame0[5]) + frac * (sFrame1[off + 2] - sFrame1[5]);
                if constexpr (!MIXED) {
                    sCurPertPos[3 * p + 0] = px;
                    sCurPertPos[3 * p + 1] = py;
                    sCurPertPos[3 * p + 2] = pz;
                } else {
                    sCurPertPosN[3 * p + 0] = static_cast<float>(px * invRSat);
                    sCurPertPosN[3 * p + 1] = static_cast<float>(py * invRSat);
                    sCurPertPosN[3 * p + 2] = static_cast<float>(pz * invRSat);
                }
                const double mp = sPertMu[p];
                if (mp > 0.0) {
                    const double d2 = fma(px, px, fma(py, py, pz * pz));
                    const double inv = fastRsqrt64(fmax(d2, 1.0));
                    const double sc = -mp * inv * inv * inv;
                    indX = sc * px;
                    indY = sc * py;
                    indZ = sc * pz;
                }
            }
            // Unconditional warp-0 reduction of net indirect tidal acceleration
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                indX += __shfl_down_sync(0xFFFFFFFFu, indX, offset, 32);
                indY += __shfl_down_sync(0xFFFFFFFFu, indY, offset, 32);
                indZ += __shfl_down_sync(0xFFFFFFFFu, indZ, offset, 32);
            }
            if (p == 0) {
                if constexpr (!MIXED) {
                    sSumInd[0] = indX;
                    sSumInd[1] = indY;
                    sSumInd[2] = indZ;
                } else {
                    sSumIndN[0] = static_cast<float>(indX * invRSat);
                    sSumIndN[1] = static_cast<float>(indY * invRSat);
                    sSumIndN[2] = static_cast<float>(indZ * invRSat);
                }
                const double sx = oneMinusF * sFrame0[0] + frac * sFrame1[0];
                const double sy = oneMinusF * sFrame0[1] + frac * sFrame1[1];
                const double sz = oneMinusF * sFrame0[2] + frac * sFrame1[2];
                const double invS = fastRsqrt64(fmax(fma(sx, sx, fma(sy, sy, sz * sz)), 1e-30));
                const double nx = sx * invS, ny = sy * invS, nz = sz * invS;
                if constexpr (!MIXED) {
                    sCurSpin[0] = nx;
                    sCurSpin[1] = ny;
                    sCurSpin[2] = nz;
                } else {
                    sCurSpinF[0] = static_cast<float>(nx);
                    sCurSpinF[1] = static_cast<float>(ny);
                    sCurSpinF[2] = static_cast<float>(nz);
                }
            }
        }
    };

    // Load initial frame (k = 0) and evaluate start-of-step ring acceleration (FSAL Verlet)
    const int frameElems = (2 + numRingPert) * 3;
    if (threadIdx.x < frameElems) {
        const double v = ringHist[threadIdx.x];
        sFrame0[threadIdx.x] = v;
        sFrame1[threadIdx.x] = v;
    }
    __syncthreads();
    loadFrameState(0.0, 1.0);
    __syncthreads();

    if (active) {
        ringAccel<MIXED>(rx, ry, rz, muSat, j2R2Sat, safeRSat, invRSat, muSatN, j2SatF,
                         sCurSpin, sCurSpinF, sCurPertPos, sCurPertPosN,
                         sPertMu, sPertMuN, sSumInd, sSumIndN, numRingPert, ax, ay, az);
    }

    for (int k = 0; k < actualSubsteps; ++k) {
        const int k1 = min(k + 1, MAX_RING_HIST - 1);
        __syncthreads();
        if (threadIdx.x < frameElems) {
            sFrame0[threadIdx.x] = sFrame1[threadIdx.x];
            sFrame1[threadIdx.x] = ringHist[k1 * RING_FRAME_STRIDE + threadIdx.x];
        }
        __syncthreads();

        for (int m = 0; m < nLocal; ++m) {
            if (active) {
                vx = fma(halfMicro, ax, vx);
                vy = fma(halfMicro, ay, vy);
                vz = fma(halfMicro, az, vz);
                rx = fma(vx, microDt, rx);
                ry = fma(vy, microDt, ry);
                rz = fma(vz, microDt, rz);
            }

            const double frac = static_cast<double>(m + 1) / static_cast<double>(nLocal);
            const double oneMinusF = 1.0 - frac;
            __syncthreads();
            loadFrameState(oneMinusF, frac);
            __syncthreads();

            if (active) {
                ringAccel<MIXED>(rx, ry, rz, muSat, j2R2Sat, safeRSat, invRSat, muSatN, j2SatF,
                                 sCurSpin, sCurSpinF, sCurPertPos, sCurPertPosN,
                                 sPertMu, sPertMuN, sSumInd, sSumIndN, numRingPert, ax, ay, az);
                vx = fma(halfMicro, ax, vx);
                vy = fma(halfMicro, ay, vy);
                vz = fma(halfMicro, az, vz);
            }
        }
    }

    if (active) {
        rp[i]         = rx; rp[N + i]     = ry; rp[2 * N + i] = rz;
        rp[3 * N + i] = vx; rp[4 * N + i] = vy; rp[5 * N + i] = vz;
    }
}

__global__ void writeRingRenderKernel(const double* __restrict__ rp, size_t count,
                                      const double* __restrict__ hs, int M,
                                      int saturnIdx, int origin,
                                      float4* __restrict__ out) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    double dx, dy, dz;
    if (origin == saturnIdx) {
        dx = rp[i];
        dy = rp[count + i];
        dz = rp[2 * count + i];
    } else {
        const double sx = hs[saturnIdx];
        const double sy = hs[M + saturnIdx];
        const double sz = hs[2 * M + saturnIdx];
        double ox = 0.0, oy = 0.0, oz = 0.0;
        if (origin >= 0) {
            ox = hs[origin];
            oy = hs[M + origin];
            oz = hs[2 * M + origin];
        }
        dx = (sx - ox) + rp[i];
        dy = (sy - oy) + rp[count + i];
        dz = (sz - oz) + rp[2 * count + i];
    }
    out[i] = make_float4(static_cast<float>(dx * INV_AU),
                         static_cast<float>(dy * INV_AU),
                         static_cast<float>(dz * INV_AU), 0.0f);
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

// ---------------------------------------------------------------------------
// Zero-readback GPU Saturnian Ring Radial Density Profile Kernel
// ---------------------------------------------------------------------------
__global__ void ringStatsKernel(const double* __restrict__ rp, size_t N,
                                const double* __restrict__ mu, const double* __restrict__ j2R2,
                                const double* __restrict__ spinAxis, int M, int saturnIdx,
                                uint32_t* __restrict__ outBins, float* __restrict__ outSums) {
    __shared__ uint32_t sBins[TOTAL_RING_INT_BINS];
    __shared__ float sSumRadKm;
    __shared__ float sSumEcc;

    for (int b = threadIdx.x; b < TOTAL_RING_INT_BINS; b += blockDim.x) sBins[b] = 0u;
    if (threadIdx.x == 0) {
        sSumRadKm = 0.0f;
        sSumEcc = 0.0f;
    }
    __syncthreads();

    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < N) {
        const double muSat = mu[saturnIdx];
        const double j2R2Sat = (muSat > 0.0) ? j2R2[saturnIdx] : 0.0;
        const double sx = spinAxis[saturnIdx];
        const double sy = spinAxis[M + saturnIdx];
        const double sz = spinAxis[2 * M + saturnIdx];

        const double rx = rp[i];
        const double ry = rp[N + i];
        const double rz = rp[2 * N + i];
        const double vx = rp[3 * N + i];
        const double vy = rp[4 * N + i];
        const double vz = rp[5 * N + i];

        const double r = sqrt(fmax(rx * rx + ry * ry + rz * rz, 1.0));
        const double inv = 1.0 / r;
        const double v2 = vx * vx + vy * vy + vz * vz;
        const double rDotS = (rx * sx + ry * sy + rz * sz) * inv;
        const double uJ2 = (j2R2Sat > 0.0)
            ? (0.5 * muSat * j2R2Sat * inv * inv * inv * (3.0 * rDotS * rDotS - 1.0))
            : 0.0;
        const double eps = 0.5 * v2 - muSat * inv + uJ2;

        if (muSat <= 0.0 || eps >= 0.0) {
            atomicAdd(&sBins[RingStats::RING_BINS + 1], 1u); // escapedCount
        } else {
            atomicAdd(&sBins[RingStats::RING_BINS], 1u);     // boundCount

            const double hx = ry * vz - rz * vy;
            const double hy = rz * vx - rx * vz;
            const double hz = rx * vy - ry * vx;
            const double h2 = hx * hx + hy * hy + hz * hz;
            const double muEff = muSat * (1.0 + 1.5 * j2R2Sat * inv * inv);
            const double epsEff = 0.5 * v2 - muEff * inv;
            const double eccTerm = 1.0 + (2.0 * epsEff * h2) / (muEff * muEff);
            const double ecc = sqrt(fmax(eccTerm, 0.0));

            const double rKm = r * 1.0e-3;
            atomicAdd(&sSumRadKm, static_cast<float>(rKm));
            atomicAdd(&sSumEcc, static_cast<float>(ecc));

            if (rKm >= RingStats::RING_MIN_KM && rKm < RingStats::RING_MAX_KM) {
                constexpr double span = RingStats::RING_MAX_KM - RingStats::RING_MIN_KM;
                int bin = static_cast<int>((rKm - RingStats::RING_MIN_KM) / span * RingStats::RING_BINS);
                bin = min(max(bin, 0), RingStats::RING_BINS - 1);
                atomicAdd(&sBins[bin], 1u);
            }
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < TOTAL_RING_INT_BINS; b += blockDim.x) {
        if (sBins[b] > 0u) atomicAdd(&outBins[b], sBins[b]);
    }
    if (threadIdx.x == 0) {
        if (sSumRadKm > 0.0f) atomicAdd(&outSums[0], sSumRadKm);
        if (sSumEcc > 0.0f)   atomicAdd(&outSums[1], sSumEcc);
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
    CUDA_CHECK(cudaEventCreateWithFlags(&m_ringStatsEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&m_timerStart));
    CUDA_CHECK(cudaEventCreate(&m_timerStop));
}

SubstrateEngine::~SubstrateEngine() {
    release();
    cudaEventDestroy(m_timerStop);
    cudaEventDestroy(m_timerStart);
    cudaEventDestroy(m_ringStatsEvent);
    cudaEventDestroy(m_beltEvent);
    cudaEventDestroy(m_snapEvent);
    cudaStreamDestroy(m_stream);
}

void SubstrateEngine::release() {
    if (m_stream) cudaStreamSynchronize(m_stream);
    cudaFree(d_heavy);              d_heavy = nullptr;
    cudaFree(d_heavyPingPong);      d_heavyPingPong = nullptr;
    cudaFree(d_eihScratch);         d_eihScratch = nullptr;
    cudaFree(d_cflOmega2);          d_cflOmega2 = nullptr;
    cudaFree(d_bodyRingSlot);       d_bodyRingSlot = nullptr;
    cudaFree(d_mu);                 d_mu = nullptr;
    cudaFree(d_rc);                 d_rc = nullptr;
    cudaFree(d_j2R2);               d_j2R2 = nullptr;
    cudaFree(d_spinAxis);           d_spinAxis = nullptr;
    cudaFree(d_spinPrecCoeff);      d_spinPrecCoeff = nullptr;
    cudaFree(d_parentIdx);          d_parentIdx = nullptr;
    cudaFree(d_moonMask);           d_moonMask = nullptr;
    cudaFree(d_particles);          d_particles = nullptr;
    cudaFree(d_ringParticles);      d_ringParticles = nullptr;
    cudaFree(d_ringPertIdx);        d_ringPertIdx = nullptr;
    cudaFree(d_ringHist);           d_ringHist = nullptr;
    cudaFree(d_ringActualSubsteps); d_ringActualSubsteps = nullptr;
    cudaFree(d_beltBins);           d_beltBins = nullptr;
    cudaFree(d_beltSums);           d_beltSums = nullptr;
    cudaFree(d_ringBins);           d_ringBins = nullptr;
    cudaFree(d_ringSums);           d_ringSums = nullptr;
    cudaFreeHost(h_snapPinned);     h_snapPinned = nullptr;
    cudaFreeHost(h_beltBinsPinned); h_beltBinsPinned = nullptr;
    cudaFreeHost(h_beltSumsPinned); h_beltSumsPinned = nullptr;
    cudaFreeHost(h_ringBinsPinned); h_ringBinsPinned = nullptr;
    cudaFreeHost(h_ringSumsPinned); h_ringSumsPinned = nullptr;
    m_useCoopHeavy = false;
    m_numRingParticles = 0;
    m_saturnIdx = -1;
    m_numRingPerturbers = 0;
    m_oblateSunMask = 0ULL;
    m_deviceBytes = 0;
    m_snapPending = false;
    m_beltPending = false;
    m_ringStatsPending = false;
    m_timerPending = false;
}

void SubstrateEngine::refreshCachedAccelerations(bool relativistic) {
    if (m_numHeavy <= 0) return;
    // Exact multiple of 32 covering M bodies (1 thread/body)
    const int heavyBlockThreads = ((m_numHeavy + 31) / 32) * 32;
    heavyInitAccKernel<<<1, heavyBlockThreads, 0, m_stream>>>(d_heavy, d_mu, d_j2R2, d_spinAxis,
                                                              d_spinPrecCoeff, d_parentIdx, d_moonMask,
                                                              m_oblateSunMask, m_numHeavy, relativistic);
    if (m_numParticles > 0) {
        particleInitAccKernel<false><<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_particles, m_numParticles, d_heavy, d_mu, d_j2R2, d_spinAxis, m_numHeavy, relativistic);
    }
    CUDA_CHECK(cudaGetLastError());
}

void SubstrateEngine::load(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles,
                           bool correctBarycenter, const ParticleSet* saturnRings) {
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
    m_spinPrecCoeff.assign(M, 0.0);
    m_momentC.assign(M, 0.34);
    m_j2.assign(M, 0.0);
    m_j2R2.assign(M, 0.0);
    m_obliquityDeg.assign(M, 0.0);
    m_tidalLockMode.assign(M, 0);
    m_parentIdx.assign(M, -1);
    m_moonMask.assign(M, 0ULL);
    m_oblateSunMask = 0ULL;
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

        const double absOmega = std::abs(m_spinOmega[i]);
        m_j2[i] = deriveHydrostaticJ2(absOmega, m_radius[i], m_mu[i], m_momentC[i]);
        m_j2R2[i] = m_j2[i] * m_radius[i] * m_radius[i] * theta;
        m_spinPrecCoeff[i] = (m_tidalLockMode[i] == 0 && m_j2[i] > 0.0 && m_momentC[i] > 0.0 && absOmega > 1e-18 && theta > 0.5)
            ? (3.0 * m_j2[i] / (m_momentC[i] * absOmega))
            : 0.0;

        if (m_parentIdx[i] == 0 && m_j2R2[i] > 0.0 && i < 64) {
            m_oblateSunMask |= (1ULL << static_cast<unsigned int>(i));
        }

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
    CUDA_CHECK(cudaMalloc(&d_heavyPingPong, 2 * 6 * MAX_HEAVY * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eihScratch, 7 * MAX_HEAVY * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cflOmega2, MAX_HEAVY * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bodyRingSlot, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mu, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rc, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_j2R2, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spinAxis, spinSoA.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spinPrecCoeff, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_parentIdx, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moonMask, M * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_beltBins, TOTAL_INT_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_beltSums, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_heavy, hs.data(), hs.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, m_mu.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc, m_rc.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_j2R2, m_j2R2.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spinAxis, spinSoA.data(), spinSoA.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spinPrecCoeff, m_spinPrecCoeff.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parentIdx, m_parentIdx.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moonMask, m_moonMask.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&h_snapPinned, 9 * M * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_beltBinsPinned, TOTAL_INT_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_beltSumsPinned, 2 * sizeof(float)));
    m_deviceBytes = (hs.size() + 8 * M + 20 * MAX_HEAVY) * sizeof(double) + M * sizeof(int);

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

    // Dedicated passive Saturnian ring system buffers (strictly separate from heliocentric BeltStats)
    m_saturnIdx = findBodyIndex(heavy, "Saturn");
    std::vector<int> ringPert;
    if (m_saturnIdx >= 0) {
        const int sunIdx = findBodyIndex(heavy, "Sun");
        if (sunIdx >= 0 && sunIdx != m_saturnIdx) ringPert.push_back(sunIdx);
        for (int i = 0; i < M; ++i) {
            if (heavy[i].parent == "Saturn" && static_cast<int>(ringPert.size()) < MAX_RING_PERT) {
                ringPert.push_back(i);
            }
        }
    }
    m_numRingPerturbers = static_cast<int>(ringPert.size());

    std::vector<int> bodyRingSlot(M, -1);
    for (size_t s = 0; s < ringPert.size(); ++s) {
        const int idx = ringPert[s];
        if (idx >= 0 && idx < M) bodyRingSlot[idx] = static_cast<int>(s);
    }
    CUDA_CHECK(cudaMemcpy(d_bodyRingSlot, bodyRingSlot.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    const size_t NR = (saturnRings != nullptr && m_saturnIdx >= 0) ? saturnRings->count() : 0;
    m_numRingParticles = NR;
    if (NR > 0) {
        CUDA_CHECK(cudaMalloc(&d_ringParticles, 6 * NR * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ringPertIdx, MAX_RING_PERT * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ringHist, MAX_RING_HIST * RING_FRAME_STRIDE * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ringActualSubsteps, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ringBins, TOTAL_RING_INT_BINS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ringSums, 2 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_ringBinsPinned, TOTAL_RING_INT_BINS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_ringSumsPinned, 2 * sizeof(float)));
        m_deviceBytes += 6 * NR * sizeof(double) + MAX_RING_HIST * RING_FRAME_STRIDE * sizeof(double) +
                         TOTAL_RING_INT_BINS * sizeof(uint32_t) + 2 * sizeof(float);

        const std::vector<double>* rsrc[6] = {&saturnRings->px, &saturnRings->py, &saturnRings->pz,
                                              &saturnRings->vx, &saturnRings->vy, &saturnRings->vz};
        for (int k = 0; k < 6; ++k) {
            CUDA_CHECK(cudaMemcpy(d_ringParticles + k * NR, rsrc[k]->data(), NR * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        if (!ringPert.empty()) {
            CUDA_CHECK(cudaMemcpy(d_ringPertIdx, ringPert.data(), ringPert.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        const int initSub = 1;
        CUDA_CHECK(cudaMemcpy(d_ringActualSubsteps, &initSub, sizeof(int), cudaMemcpyHostToDevice));
    }

    // Query cooperative multi-SM launch eligibility (automatically falls back to 1-block if unsupported)
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    m_useCoopHeavy = false;
    if (prop.cooperativeLaunch) {
        int maxBlocksPerSM = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM, heavySubstepCoopKernel, 32, 0) == cudaSuccess) {
            m_useCoopHeavy = (M <= maxBlocksPerSM * prop.multiProcessorCount);
        }
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
    const double absOmega = std::abs(m_spinOmega[bodyIdx]);
    m_j2[bodyIdx] = deriveHydrostaticJ2(absOmega, m_radius[bodyIdx], m_mu[bodyIdx], m_momentC[bodyIdx]);
    m_j2R2[bodyIdx] = m_j2[bodyIdx] * m_radius[bodyIdx] * m_radius[bodyIdx] * theta;
    m_spinPrecCoeff[bodyIdx] = (m_tidalLockMode[bodyIdx] == 0 && m_j2[bodyIdx] > 0.0 && m_momentC[bodyIdx] > 0.0 && absOmega > 1e-18 && theta > 0.5)
        ? (3.0 * m_j2[bodyIdx] / (m_momentC[bodyIdx] * absOmega))
        : 0.0;

    if (m_mu[bodyIdx] != prevMu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu + bodyIdx, &m_mu[bodyIdx], sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_j2R2 + bodyIdx, &m_j2R2[bodyIdx], sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_spinPrecCoeff + bodyIdx, &m_spinPrecCoeff[bodyIdx], sizeof(double),
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
            const double absOmega = std::abs(m_spinOmega[i]);
            m_j2[i] = deriveHydrostaticJ2(absOmega, m_radius[i], m_mu[i], m_momentC[i]);
            m_j2R2[i] = m_j2[i] * m_radius[i] * m_radius[i] * theta;
            m_spinPrecCoeff[i] = (m_tidalLockMode[i] == 0 && m_j2[i] > 0.0 && m_momentC[i] > 0.0 && absOmega > 1e-18 && theta > 0.5)
                ? (3.0 * m_j2[i] / (m_momentC[i] * absOmega))
                : 0.0;
            anyChanged = true;
        }
    }
    if (anyChanged && d_mu) {
        CUDA_CHECK(cudaMemcpyAsync(d_mu, m_mu.data(), m_numHeavy * sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_j2R2, m_j2R2.data(), m_numHeavy * sizeof(double),
                                   cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_spinPrecCoeff, m_spinPrecCoeff.data(), m_numHeavy * sizeof(double),
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

void SubstrateEngine::requestRingStats() {
    if (m_numRingParticles == 0 || m_saturnIdx < 0 || !d_ringParticles ||
        !d_ringBins || !d_ringSums || !h_ringBinsPinned || !h_ringSumsPinned || m_ringStatsPending) {
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(d_ringBins, 0, TOTAL_RING_INT_BINS * sizeof(uint32_t), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_ringSums, 0, 2 * sizeof(float), m_stream));

    ringStatsKernel<<<blocksFor(m_numRingParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
        d_ringParticles, m_numRingParticles, d_mu, d_j2R2, d_spinAxis, m_numHeavy, m_saturnIdx,
        d_ringBins, d_ringSums);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_ringBinsPinned, d_ringBins, TOTAL_RING_INT_BINS * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_ringSumsPinned, d_ringSums, 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaEventRecord(m_ringStatsEvent, m_stream));
    m_ringStatsPending = true;
}

bool SubstrateEngine::pollRingStats(RingStats& out) {
    if (!m_ringStatsPending) return false;
    const cudaError_t q = cudaEventQuery(m_ringStatsEvent);
    if (q == cudaErrorNotReady) return false;
    CUDA_CHECK(q);

    out = RingStats{};
    for (int i = 0; i < RingStats::RING_BINS; ++i) {
        out.radialHist[i] = static_cast<float>(h_ringBinsPinned[i]);
    }
    out.boundCount   = h_ringBinsPinned[RingStats::RING_BINS];
    out.escapedCount = h_ringBinsPinned[RingStats::RING_BINS + 1];
    if (out.boundCount > 0) {
        out.meanRadiusKm = h_ringSumsPinned[0] / static_cast<float>(out.boundCount);
        out.meanEcc      = h_ringSumsPinned[1] / static_cast<float>(out.boundCount);
    }

    m_ringStatsPending = false;
    return true;
}

void SubstrateEngine::computeRingStats(RingStats& out) {
    if (m_ringStatsPending) {
        CUDA_CHECK(cudaEventSynchronize(m_ringStatsEvent));
        pollRingStats(out);
    }
    out = RingStats{};
    requestRingStats();
    if (m_ringStatsPending) {
        CUDA_CHECK(cudaEventSynchronize(m_ringStatsEvent));
        pollRingStats(out);
    }
}

void SubstrateEngine::step(const StepParams& p, bool captureSnapshot) {
    const int M = m_numHeavy;
    const int sub = std::max(1, p.substeps);
    const double dtSub = p.dtSeconds / static_cast<double>(sub);
    const bool hasRings = (m_numRingParticles > 0 && m_saturnIdx >= 0 && d_ringParticles != nullptr);
    double* ringHistPtr = hasRings ? d_ringHist : nullptr;
    int* ringSubPtr = hasRings ? d_ringActualSubsteps : nullptr;

    if (m_useCoopHeavy) {
        int mArg = M;
        bool relArg = p.relativistic;
        double dtArg = dtSub;
        int subArg = sub;
        uint64_t oblateMask = m_oblateSunMask;
        int satIdx = m_saturnIdx;
        void* args[] = {
            &d_heavy, &d_heavyPingPong, &d_eihScratch, &d_cflOmega2,
            &d_mu, &d_j2R2,
            &d_spinAxis, &d_spinPrecCoeff,
            &d_parentIdx, &d_moonMask, &oblateMask,
            &satIdx, &d_bodyRingSlot, &ringHistPtr, &ringSubPtr,
            &mArg, &relArg, &dtArg, &subArg
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(heavySubstepCoopKernel),
            dim3(static_cast<unsigned int>(M)), dim3(32), args, 0, m_stream));
    } else {
        const int heavyBlockThreads = ((M + 31) / 32) * 32;
        heavySubstepFallbackKernel<<<1, heavyBlockThreads, 0, m_stream>>>(
            d_heavy, d_mu, d_j2R2, d_spinAxis, d_spinPrecCoeff,
            d_parentIdx, d_moonMask, m_oblateSunMask,
            m_saturnIdx, d_bodyRingSlot, ringHistPtr, ringSubPtr,
            M, p.relativistic, dtSub, sub);
    }
    m_simTime += p.dtSeconds;
    if (captureSnapshot) requestSnapshot();

    if (hasRings) {
        const double rSatM = m_radius[m_saturnIdx];
        const unsigned int ringBlocks = blocksFor(m_numRingParticles, PARTICLE_BLOCK);
        if (p.precision == Precision::Mixed) {
            ringStepKernel<true><<<ringBlocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_ringParticles, m_numRingParticles, d_mu, d_j2R2,
                m_saturnIdx, d_ringPertIdx, m_numRingPerturbers,
                d_ringHist, d_ringActualSubsteps, p.dtSeconds, rSatM);
        } else {
            ringStepKernel<false><<<ringBlocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_ringParticles, m_numRingParticles, d_mu, d_j2R2,
                m_saturnIdx, d_ringPertIdx, m_numRingPerturbers,
                d_ringHist, d_ringActualSubsteps, p.dtSeconds, rSatM);
        }
    }

    if (m_numParticles > 0) {
        const unsigned int blocks = blocksFor(m_numParticles, PARTICLE_BLOCK);
        if (p.precision == Precision::Mixed)
            particleStepKernel<true><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_j2R2, d_spinAxis,
                M, p.relativistic, p.dtSeconds);
        else
            particleStepKernel<false><<<blocks, PARTICLE_BLOCK, 0, m_stream>>>(
                d_particles, m_numParticles, d_heavy, d_mu, d_j2R2, d_spinAxis,
                M, p.relativistic, p.dtSeconds);
    }
    CUDA_CHECK(cudaGetLastError());
}

void SubstrateEngine::requestSnapshot() {
    const int M = m_numHeavy;
    CUDA_CHECK(cudaMemcpyAsync(h_snapPinned, d_heavy, 6 * M * sizeof(double),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_snapPinned + 6 * M, d_spinAxis, 3 * M * sizeof(double),
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
        m_snapshot.spinAxis.resize(3 * M);
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < 3; ++k) {
                m_snapshot.pos[3 * i + k] = h_snapPinned[k * M + i];
                m_snapshot.vel[3 * i + k] = h_snapPinned[(3 + k) * M + i];
                m_snapshot.spinAxis[3 * i + k] = h_snapPinned[(6 + k) * M + i];
            }
            m_spinAxisX[i] = m_snapshot.spinAxis[3 * i + 0];
            m_spinAxisY[i] = m_snapshot.spinAxis[3 * i + 1];
            m_spinAxisZ[i] = m_snapshot.spinAxis[3 * i + 2];
        }
        m_snapshot.simTimeSeconds = m_snapTime;
        m_snapPending = false;
    }
    return m_snapshot;
}

void SubstrateEngine::writeRenderPositions(float4* heavyOut, float4* particleOut, int originIdx, float4* ringOut) {
    const int M = m_numHeavy;
    if (originIdx >= M) originIdx = -1;
    if (heavyOut)
        writeRenderKernel<<<1, MAX_HEAVY, 0, m_stream>>>(d_heavy, M, d_heavy, M, originIdx, heavyOut);
    if (particleOut && m_numParticles > 0)
        writeRenderKernel<<<blocksFor(m_numParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_particles, m_numParticles, d_heavy, M, originIdx, particleOut);
    if (ringOut && m_numRingParticles > 0 && m_saturnIdx >= 0 && d_ringParticles != nullptr)
        writeRingRenderKernel<<<blocksFor(m_numRingParticles, PARTICLE_BLOCK), PARTICLE_BLOCK, 0, m_stream>>>(
            d_ringParticles, m_numRingParticles, d_heavy, M, m_saturnIdx, originIdx, ringOut);
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

void SubstrateEngine::copyRingPositions(std::vector<double>& xyzOut) {
    const size_t N = m_numRingParticles;
    std::vector<double> soa(3 * N);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    if (N > 0 && d_ringParticles != nullptr)
        CUDA_CHECK(cudaMemcpy(soa.data(), d_ringParticles, 3 * N * sizeof(double), cudaMemcpyDeviceToHost));
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
    const bool hasLiveSpin = (s.spinAxis.size() == static_cast<size_t>(3 * M));
    auto spinAt = [&](int idx, double& sx, double& sy, double& sz) {
        if (hasLiveSpin) {
            sx = s.spinAxis[3 * idx + 0];
            sy = s.spinAxis[3 * idx + 1];
            sz = s.spinAxis[3 * idx + 2];
        } else {
            sx = m_spinAxisX[idx];
            sy = m_spinAxisY[idx];
            sz = m_spinAxisZ[idx];
        }
    };

    // Precompute scalar Newtonian potential phi[i] = sum_{j != i} mu_j / r_ij for 3-body EIH 1PN energy
    std::vector<double> phi(M, 0.0);
    if (relativistic) {
        for (int i = 0; i < M; ++i) {
            double pi = 0.0;
            for (int j = 0; j < M; ++j) {
                if (j == i || m_mu[j] <= 0.0) continue;
                const double dx = s.pos[3 * j + 0] - s.pos[3 * i + 0];
                const double dy = s.pos[3 * j + 1] - s.pos[3 * i + 1];
                const double dz = s.pos[3 * j + 2] - s.pos[3 * i + 2];
                const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
                pi += m_mu[j] / r;
            }
            phi[i] = pi;
        }
    }

    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* v = &s.vel[3 * i];
        const double vi2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        ke += 0.5 * m_mass[i] * vi2;
        if (relativistic) {
            ke += (0.375 * INV_C2) * m_mass[i] * vi2 * vi2;
            if (m_mu[i] > 0.0) {
                pe += (0.5 * INV_C2) * m_mass[i] * phi[i] * phi[i];
            }
        }
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const double dx = s.pos[3 * j + 0] - s.pos[3 * i + 0];
            const double dy = s.pos[3 * j + 1] - s.pos[3 * i + 1];
            const double dz = s.pos[3 * j + 2] - s.pos[3 * i + 2];
            const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
            const double inv = 1.0 / r;
            pe -= 0.5 * m_mass[i] * m_mu[j] * inv;
            if (relativistic && m_mu[j] > 0.0) {
                const double* vj = &s.vel[3 * j];
                const double vj2 = vj[0] * vj[0] + vj[1] * vj[1] + vj[2] * vj[2];
                const double viDotVj = v[0] * vj[0] + v[1] * vj[1] + v[2] * vj[2];
                const double nDotVi = (dx * v[0] + dy * v[1] + dz * v[2]) * inv;
                const double nDotVj = (dx * vj[0] + dy * vj[1] + dz * vj[2]) * inv;
                pe += (0.25 * INV_C2) * (m_mass[i] * m_mu[j] * inv) *
                      (3.0 * vi2 + 3.0 * vj2 - 7.0 * viDotVj - nDotVi * nDotVj);
            }
        }
        const int p = m_parentIdx[i];
        // Moon-Planet J2 potential (oblate planet p pulling on moon i)
        if (p > 0 && m_j2R2[p] > 0.0 && m_mu[p] > 0.0 && m_mu[i] > 0.0) {
            const double dx = s.pos[3 * p] - s.pos[3 * i];
            const double dy = s.pos[3 * p + 1] - s.pos[3 * i + 1];
            const double dz = s.pos[3 * p + 2] - s.pos[3 * i + 2];
            const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
            const double inv = 1.0 / r;
            double spX, spY, spZ;
            spinAt(p, spX, spY, spZ);
            const double rDotS = (-dx * spX - dy * spY - dz * spZ) * inv;
            pe += 0.5 * m_mass[i] * m_mu[p] * m_j2R2[p] * inv * inv * inv * (3.0 * rDotS * rDotS - 1.0);
        }
        // Sun-Planet J2 potential (oblate planet i coupled to Sun j = 0)
        if (p == 0 && m_j2R2[i] > 0.0 && m_mu[0] > 0.0 && m_mu[i] > 0.0) {
            const double dx = s.pos[0] - s.pos[3 * i];
            const double dy = s.pos[1] - s.pos[3 * i + 1];
            const double dz = s.pos[2] - s.pos[3 * i + 2];
            const double r = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
            const double inv = 1.0 / r;
            double spX, spY, spZ;
            spinAt(i, spX, spY, spZ);
            const double rDotS = (dx * spX + dy * spY + dz * spZ) * inv;
            pe += 0.5 * m_mass[i] * m_mu[0] * m_j2R2[i] * inv * inv * inv * (3.0 * rDotS * rDotS - 1.0);
        }
    }
    return ke + pe;
}

void SubstrateEngine::canonicalMomentum(const HeavySnapshot& s, bool relativistic,
                                        double& pxOut, double& pyOut, double& pzOut) const {
    const int M = m_numHeavy;
    double Px = 0.0, Py = 0.0, Pz = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* r = &s.pos[3 * i];
        const double* v = &s.vel[3 * i];
        const double m = m_mass[i];
        double pix = m * v[0], piy = m * v[1], piz = m * v[2];
        if (relativistic) {
            const double vi2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            const double kinFac = 1.0 + 0.5 * vi2 * INV_C2;
            pix = m * v[0] * kinFac;
            piy = m * v[1] * kinFac;
            piz = m * v[2] * kinFac;
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int j = 0; j < M; ++j) {
                if (j == i || m_mu[j] <= 0.0) continue;
                const double dx = s.pos[3 * j + 0] - r[0];
                const double dy = s.pos[3 * j + 1] - r[1];
                const double dz = s.pos[3 * j + 2] - r[2];
                const double inv = 1.0 / std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
                const double nx = dx * inv, ny = dy * inv, nz = dz * inv;
                const double* vj = &s.vel[3 * j];
                const double nDotVj = nx * vj[0] + ny * vj[1] + nz * vj[2];
                const double w = m * m_mu[j] * inv;
                cx += w * (6.0 * v[0] - 7.0 * vj[0] - nDotVj * nx);
                cy += w * (6.0 * v[1] - 7.0 * vj[1] - nDotVj * ny);
                cz += w * (6.0 * v[2] - 7.0 * vj[2] - nDotVj * nz);
            }
            pix += (0.5 * INV_C2) * cx;
            piy += (0.5 * INV_C2) * cy;
            piz += (0.5 * INV_C2) * cz;
        }
        Px += pix;
        Py += piy;
        Pz += piz;
    }
    pxOut = Px; pyOut = Py; pzOut = Pz;
}

double SubstrateEngine::angularMomentumMag(const HeavySnapshot& s, bool relativistic) const {
    const int M = m_numHeavy;
    const bool hasLiveSpin = (s.spinAxis.size() == static_cast<size_t>(3 * M));
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;
    for (int i = 0; i < M; ++i) {
        const double* r = &s.pos[3 * i];
        const double* v = &s.vel[3 * i];
        const double m = m_mass[i];
        double pix = m * v[0], piy = m * v[1], piz = m * v[2];
        if (relativistic) {
            const double vi2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            const double kinFac = 1.0 + 0.5 * vi2 * INV_C2;
            pix = m * v[0] * kinFac;
            piy = m * v[1] * kinFac;
            piz = m * v[2] * kinFac;
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int j = 0; j < M; ++j) {
                if (j == i || m_mu[j] <= 0.0) continue;
                const double dx = s.pos[3 * j + 0] - r[0];
                const double dy = s.pos[3 * j + 1] - r[1];
                const double dz = s.pos[3 * j + 2] - r[2];
                const double inv = 1.0 / std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
                const double nx = dx * inv, ny = dy * inv, nz = dz * inv;
                const double* vj = &s.vel[3 * j];
                const double nDotVj = nx * vj[0] + ny * vj[1] + nz * vj[2];
                const double w = m * m_mu[j] * inv;
                cx += w * (6.0 * v[0] - 7.0 * vj[0] - nDotVj * nx);
                cy += w * (6.0 * v[1] - 7.0 * vj[1] - nDotVj * ny);
                cz += w * (6.0 * v[2] - 7.0 * vj[2] - nDotVj * nz);
            }
            pix += (0.5 * INV_C2) * cx;
            piy += (0.5 * INV_C2) * cy;
            piz += (0.5 * INV_C2) * cz;
        }
        Lx += r[1] * piz - r[2] * piy;
        Ly += r[2] * pix - r[0] * piz;
        Lz += r[0] * piy - r[1] * pix;

        // Add live precessing axial spin angular momentum S_i = C_i * M_i * R_i^2 * |omega_i| * s_hat_i(t)
        const double spinMag = m_momentC[i] * m * m_radius[i] * m_radius[i] * std::abs(m_spinOmega[i]);
        const double spX = hasLiveSpin ? s.spinAxis[3 * i + 0] : m_spinAxisX[i];
        const double spY = hasLiveSpin ? s.spinAxis[3 * i + 1] : m_spinAxisY[i];
        const double spZ = hasLiveSpin ? s.spinAxis[3 * i + 2] : m_spinAxisZ[i];
        Lx += spinMag * spX;
        Ly += spinMag * spY;
        Lz += spinMag * spZ;
    }
    return std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz);
}

double SubstrateEngine::barycenterDriftSpeed(const HeavySnapshot& s, bool relativistic) const {
    const int M = m_numHeavy;
    double mTot = 0.0;
    for (int i = 0; i < M; ++i) mTot += m_mass[i];
    if (mTot <= 0.0) return 0.0;
    double Px = 0.0, Py = 0.0, Pz = 0.0;
    canonicalMomentum(s, relativistic, Px, Py, Pz);
    return std::sqrt(Px * Px + Py * Py + Pz * Pz) / mTot;
}

} // namespace substrate
