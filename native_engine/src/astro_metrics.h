#pragma once

#include <vector>

#include "payload.h"
#include "substrate_engine.h"

namespace substrate {

struct KeplerElements {
    bool valid = false;
    bool bound = false;
    double r_m = 0.0;
    double v_ms = 0.0;
    double a_m = 0.0;
    double a_AU = 0.0;
    double e = 0.0;
    double inc_deg = 0.0;
    double raan_deg = 0.0;
    double argPeri_deg = 0.0;
    double trueAnom_deg = 0.0;
    double periapsis_m = 0.0;
    double apoapsis_m = 0.0;
    double period_days = 0.0;
    double specificEnergy = 0.0;
};

struct DomainRadii {
    double core1R_m = 0.0;         // 1 * R_body (Bare-metal core boundary)
    double vector3R_m = 0.0;       // 3 * R_body (3-channel directional mass/axial extension / Plasmapause delta)
    double vector4R_m = 0.0;       // 4 * R_body (4-channel unspooled EM / Van Allen boundary)
    double hillRadius_m = 0.0;     // Hill sphere r_H
    double soiRadius_m = 0.0;      // Laplace sphere of influence r_SOI
    double rocheFluid_m = 0.0;     // Fluid Roche limit (around this planet, or host planet's Roche limit for moon)
    double rocheRigid_m = 0.0;     // Rigid-body Roche limit
};

struct SubstrateBodyAudit {
    // Hardware node & compilation properties
    double nodesN = 0.0;
    double radiusM = 0.0;
    double volumeM3 = 0.0;
    double rhoNodes = 0.0;
    double thetaGC = 1.0;

    // Macro-scale quantization (Paper 1, Sec. 8)
    double lambdaComptonBody_m = 0.0;
    double quantumTopDownRadius_m = 0.0;

    // Discretized River Metric & Spatial Pixel Purge (Paper 1, Sec. 3)
    double gSurface_ms2 = 0.0;
    double vInflowSurface_ms = 0.0;
    double volDeletedSurface_m3s = 0.0;
    double pixelPurgeSurface_s = 0.0;

    double gParentAtOrbit_ms2 = 0.0;
    double vInflowOrbit_ms = 0.0;
    double pixelPurgeParentAtOrbit_s = 0.0;

    // Pythagorean Bandwidth Invariant & CPU Clock Throttling (Paper 1, Sec. 4-5)
    double schwarzschildRadius_m = 0.0;  // Kernel Lock horizon R_EH = 2 * N * lambda_p / T_Omega^2
    double safetyRadiusReq_m = 0.0;      // Bekenstein saturation floor R_req = R_EH / 2
    double safetyFactor = 0.0;           // R_body / R_req
    double omegaSurface = 0.0;           // Bandwidth utilization at body surface
    double clockLagSurface_ppb = 0.0;    // (1 - u_Omega) * 1e9
    double clockLagSurface_usDay = 0.0;  // Microseconds/day slower than deep-void clock
    double omegaOrbit = 0.0;             // Bandwidth utilization in orbit (parent g_Omega^2 + v_Omega^2)
    double clockLagOrbit_ppb = 0.0;
    double clockLagOrbit_usDay = 0.0;
    double precessionArcsecCentury = 0.0; // 1PN periapsis precession rate (arcsec / Julian century)
};

struct CoarseGrainAudit {
    double probeRadiusAU = 121.54;
    double probeVolumeM3 = 0.0;
    double enclosedNodes = 0.0;
    double enclosedCompiledNodes = 0.0;
    double meanRhoNodes = 0.0;
    bool exceedsGCCeiling = false;
    double localDeltaH_kmsMpc = 0.0;     // Pointer-recycling expansion boost (Paper 2, Sec. 5)
    double gravFieldEnergyDensity_Jm3 = 0.0; // u_g(R) = m_p * g_Omega^2 / (8*pi*K_Omega) vs. Sigma_0
};

KeplerElements computeKeplerElements(int bodyIdx, int parentIdx,
                                     const HeavySnapshot& snap,
                                     const SubstrateEngine& engine);

DomainRadii computeDomainRadii(int bodyIdx, int parentIdx,
                               const std::vector<HeavyBodyDesc>& descs,
                               const KeplerElements& kep,
                               const SubstrateEngine& engine);

SubstrateBodyAudit computeSubstrateAudit(int bodyIdx, int parentIdx,
                                         const HeavySnapshot& snap,
                                         const KeplerElements& kep,
                                         const SubstrateEngine& engine,
                                         bool relativisticEnabled);

CoarseGrainAudit computeCoarseGrainAudit(double probeRadiusAU,
                                         const HeavySnapshot& snap,
                                         const SubstrateEngine& engine);

} // namespace substrate

