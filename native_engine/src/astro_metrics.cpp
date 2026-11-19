#include "astro_metrics.h"
#include "substrate_constants.h"

#include <algorithm>
#include <cmath>

namespace substrate {

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 6.28318530717958647692;
constexpr double RAD2DEG = 180.0 / PI;
constexpr double ARCSEC_PER_RAD = (180.0 * 3600.0) / PI;

// Numerically stable 1 - sqrt(1 - x) for tiny x ~ 1e-10 without FP64 cancellation.
double oneMinusSqrtOneMinus(double x) {
    const double xc = std::clamp(x, 0.0, 1.0);
    return xc / (1.0 + std::sqrt(1.0 - xc));
}
} // namespace

KeplerElements computeKeplerElements(int bodyIdx, int parentIdx,
                                     const HeavySnapshot& snap,
                                     const SubstrateEngine& engine) {
    KeplerElements k{};
    if (parentIdx < 0 || bodyIdx < 0 || bodyIdx >= engine.numHeavy() || parentIdx >= engine.numHeavy()) {
        return k;
    }

    const double rx = snap.pos[3 * bodyIdx + 0] - snap.pos[3 * parentIdx + 0];
    const double ry = snap.pos[3 * bodyIdx + 1] - snap.pos[3 * parentIdx + 1];
    const double rz = snap.pos[3 * bodyIdx + 2] - snap.pos[3 * parentIdx + 2];

    const double vx = snap.vel[3 * bodyIdx + 0] - snap.vel[3 * parentIdx + 0];
    const double vy = snap.vel[3 * bodyIdx + 1] - snap.vel[3 * parentIdx + 1];
    const double vz = snap.vel[3 * bodyIdx + 2] - snap.vel[3 * parentIdx + 2];

    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    const double v2 = vx * vx + vy * vy + vz * vz;
    k.r_m = r;
    k.v_ms = std::sqrt(v2);

    const double muPair = engine.mu()[parentIdx] + engine.mu()[bodyIdx];
    if (r < 1.0 || muPair <= 0.0) {
        return k;
    }

    // Specific angular momentum h = r x v
    const double hx = ry * vz - rz * vy;
    const double hy = rz * vx - rx * vz;
    const double hz = rx * vy - ry * vx;
    const double h = std::sqrt(hx * hx + hy * hy + hz * hz);
    if (h < 1e-12) return k;

    // Include parent body's J2 oblate quadrupole enhancement in effective central gravity mu:
    double mu = muPair;
    const double j2Parent = engine.j2()[parentIdx];
    const double rParent  = engine.radius()[parentIdx];
    if (j2Parent > 0.0 && rParent > 0.0 && engine.mu()[parentIdx] > 0.0) {
        const double sx = engine.spinAxisX()[parentIdx];
        const double sy = engine.spinAxisY()[parentIdx];
        const double sz = engine.spinAxisZ()[parentIdx];
        const double cosIncEq = std::clamp((hx * sx + hy * sy + hz * sz) / h, -1.0, 1.0);
        const double sin2IncEq = 1.0 - cosIncEq * cosIncEq;
        const double ratio = rParent / r;
        mu = muPair * (1.0 + 1.5 * j2Parent * ratio * ratio * (1.0 - 1.5 * sin2IncEq));
    }

    k.valid = true;
    k.specificEnergy = 0.5 * v2 - mu / r;

    // Eccentricity vector e_vec = (v x h)/mu - r/r
    const double vxh_x = vy * hz - vz * hy;
    const double vxh_y = vz * hx - vx * hz;
    const double vxh_z = vx * hy - vy * hx;
    const double ex = vxh_x / mu - rx / r;
    const double ey = vxh_y / mu - ry / r;
    const double ez = vxh_z / mu - rz / r;
    k.e = std::sqrt(ex * ex + ey * ey + ez * ez);

    // Inclination
    k.inc_deg = std::acos(std::clamp(hz / h, -1.0, 1.0)) * RAD2DEG;

    // Node vector n = z_hat x h = (-hy, hx, 0)
    const double nx = -hy;
    const double ny = hx;
    const double n = std::sqrt(nx * nx + ny * ny);

    if (n > 1e-12 * h) {
        double raan = std::acos(std::clamp(nx / n, -1.0, 1.0));
        if (ny < 0.0) raan = TWO_PI - raan;
        k.raan_deg = raan * RAD2DEG;

        if (k.e > 1e-9) {
            double argP = std::acos(std::clamp((nx * ex + ny * ey) / (n * k.e), -1.0, 1.0));
            if (ez < 0.0) argP = TWO_PI - argP;
            k.argPeri_deg = argP * RAD2DEG;
        }
    } else if (k.e > 1e-9) {
        double argP = std::atan2(ey, ex);
        if (argP < 0.0) argP += TWO_PI;
        k.argPeri_deg = argP * RAD2DEG;
    }

    if (k.e > 1e-9) {
        double nu = std::acos(std::clamp((ex * rx + ey * ry + ez * rz) / (k.e * r), -1.0, 1.0));
        if ((rx * vx + ry * vy + rz * vz) < 0.0) nu = TWO_PI - nu;
        k.trueAnom_deg = nu * RAD2DEG;
    } else {
        double nu = std::atan2(ry, rx);
        if (nu < 0.0) nu += TWO_PI;
        k.trueAnom_deg = nu * RAD2DEG;
    }

    if (k.specificEnergy < -1e-12 && k.e < 1.0) {
        k.bound = true;
        k.a_m = -mu / (2.0 * k.specificEnergy);
        k.a_AU = k.a_m / AU;
        k.periapsis_m = k.a_m * (1.0 - k.e);
        k.apoapsis_m = k.a_m * (1.0 + k.e);
        k.period_days = (TWO_PI * std::sqrt((k.a_m * k.a_m * k.a_m) / mu)) / 86400.0;
    }
    return k;
}

DomainRadii computeDomainRadii(int bodyIdx, int parentIdx,
                               const std::vector<HeavyBodyDesc>& descs,
                               const KeplerElements& kep,
                               const SubstrateEngine& engine) {
    DomainRadii d{};
    if (bodyIdx < 0 || bodyIdx >= engine.numHeavy()) return d;

    const double R = engine.radius()[bodyIdx];
    d.core1R_m = R;
    d.vector3R_m = 3.0 * R;
    d.vector4R_m = 4.0 * R;
    d.rDeltaStagnation_m = C_DELTA_INV_FOURTH_ROOT * R;

    // Hill & Laplace SOI relative to parent
    if (parentIdx >= 0 && kep.bound && kep.a_m > 0.0) {
        const double mBody = engine.mass()[bodyIdx];
        const double mParent = engine.mass()[parentIdx];
        if (mParent > 0.0 && mBody > 0.0) {
            d.hillRadius_m = kep.a_m * (1.0 - kep.e) * std::cbrt(mBody / (3.0 * mParent));
            d.soiRadius_m = kep.a_m * std::pow(mBody / mParent, 0.4);
        }
    }

    // Roche limit:
    //  * If inspecting a Moon, compute its host planet's Roche limit for THIS moon's density!
    //  * Otherwise (Planet/Dwarf/Star), compute this body's Roche limit for its innermost moon
    //    (or a reference silicate/icy satellite of 2,000 kg/m^3 if moonless).
    if (descs[bodyIdx].category == BodyCategory::Moon && parentIdx >= 0) {
        const double rHost = engine.radius()[parentIdx];
        const double rhoHost = engine.baseDensity(parentIdx) * M_P;
        const double rhoMoon = std::max(engine.baseDensity(bodyIdx) * M_P, 1.0);
        const double ratio = std::cbrt(rhoHost / rhoMoon);
        d.rocheFluid_m = 2.44 * rHost * ratio;
        d.rocheRigid_m = 1.26 * rHost * ratio;
    } else {
        const double rhoPrimary = engine.baseDensity(bodyIdx) * M_P;
        double rhoSat = 2000.0;  // reference satellite density (kg/m^3)
        double minDist = 1e30;
        for (int j = 0; j < static_cast<int>(descs.size()); ++j) {
            if (descs[j].parent == descs[bodyIdx].name) {
                const double dx = descs[j].pos[0] - descs[bodyIdx].pos[0];
                const double dy = descs[j].pos[1] - descs[bodyIdx].pos[1];
                const double dz = descs[j].pos[2] - descs[bodyIdx].pos[2];
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < minDist) {
                    minDist = dist;
                    rhoSat = std::max(engine.baseDensity(j) * M_P, 1.0);
                }
            }
        }
        const double ratio = std::cbrt(rhoPrimary / rhoSat);
        d.rocheFluid_m = 2.44 * R * ratio;
        d.rocheRigid_m = 1.26 * R * ratio;
    }
    return d;
}

SubstrateBodyAudit computeSubstrateAudit(int bodyIdx, int parentIdx,
                                         const HeavySnapshot& snap,
                                         const KeplerElements& kep,
                                         const SubstrateEngine& engine,
                                         bool relativisticEnabled) {
    SubstrateBodyAudit a{};
    if (bodyIdx < 0 || bodyIdx >= engine.numHeavy()) return a;

    a.nodesN = engine.nodes()[bodyIdx];
    a.radiusM = engine.radius()[bodyIdx];
    a.volumeM3 = (4.0 / 3.0) * PI * a.radiusM * a.radiusM * a.radiusM;
    a.rhoNodes = engine.density()[bodyIdx];
    a.thetaGC = (a.rhoNodes >= RHO_GC) ? 1.0 : 0.0;

    // Collective Compton floor (Paper 1, Sec. 8.2)
    a.lambdaComptonBody_m = (a.nodesN > 0.0) ? (LAMBDA_BAR_P / a.nodesN) : 0.0;

    // Surface River Metric & Discrete Spatial Pixel Purge (Paper 1, Sec. 3)
    const double muSelf = engine.mu()[bodyIdx];  // N * K_Omega * Theta
    a.gSurface_ms2 = (a.radiusM > 0.0) ? (muSelf / (a.radiusM * a.radiusM)) : 0.0;
    a.vInflowSurface_ms = std::sqrt(2.0 * a.gSurface_ms2 * a.radiusM);
    a.volDeletedSurface_m3s = 4.0 * PI * a.radiusM * a.radiusM * a.vInflowSurface_ms;
    a.pixelPurgeSurface_s = a.volDeletedSurface_m3s / V_NODE;

    // Top-Down Quantum Boundary Radius verification (Paper 1, Eq. 455)
    if (a.lambdaComptonBody_m > 0.0 && a.gSurface_ms2 > 0.0) {
        const double waveFootprint = (C_LIGHT * LAMBDA_BAR_P) / T_OMEGA;
        const double inflowFloor = std::sqrt(a.lambdaComptonBody_m * a.gSurface_ms2);
        a.quantumTopDownRadius_m = waveFootprint / inflowFloor;
    }

    // Bekenstein / Kernel Lock Horizon (Paper 1, Sec. 4 & Sec. 9)
    a.safetyRadiusReq_m = (a.nodesN * LAMBDA_BAR_P) / T_OMEGA_SQ;
    a.schwarzschildRadius_m = 2.0 * a.safetyRadiusReq_m;
    a.safetyFactor = (a.safetyRadiusReq_m > 0.0) ? (a.radiusM / a.safetyRadiusReq_m) : 0.0;

    // Surface Bandwidth Utilization & Clock Dephasing (Paper 1, Sec. 5)
    const double c2 = C_LIGHT * C_LIGHT;
    a.omegaSurface = (a.radiusM > 0.0) ? ((2.0 * muSelf) / (c2 * a.radiusM)) : 0.0;
    const double lagSurfFrac = oneMinusSqrtOneMinus(a.omegaSurface);
    a.clockLagSurface_ppb = lagSurfFrac * 1.0e9;
    a.clockLagSurface_usDay = lagSurfFrac * 86400.0 * 1.0e6;

    // Axial Rotation, Emergent Tidal Locking (1:1 Moons & 3:2 Mercury), River-Metric Flattening & Darwin-Radau J2
    a.tidalLockMode = engine.tidalLockMode()[bodyIdx];
    a.obliquity_deg = engine.obliquityDeg()[bodyIdx];
    double omegaRot = engine.spinOmega()[bodyIdx];
    const double signRot = (omegaRot < 0.0) ? -1.0 : 1.0;
    if (a.tidalLockMode == 1 && kep.bound && kep.period_days > 0.0) {
        // 1:1 Synchronous Tidal Lock: spin period emerges dynamically from live orbital period!
        a.spinPeriodHours = signRot * kep.period_days * 24.0;
        omegaRot = signRot * (TWO_PI / (kep.period_days * 86400.0));
    } else if (a.tidalLockMode == 2 && kep.bound && kep.period_days > 0.0) {
        // 3:2 Spin-Orbit Resonance (Mercury): omega_rot = 1.5 * n_orb
        a.spinPeriodHours = signRot * (2.0 / 3.0) * kep.period_days * 24.0;
        omegaRot = signRot * 1.5 * (TWO_PI / (kep.period_days * 86400.0));
    } else {
        a.spinPeriodHours = (std::abs(omegaRot) > 1e-18) ? (TWO_PI / (omegaRot * 3600.0)) : 0.0;
    }

    const double absOmegaRot = std::abs(omegaRot);
    a.vEquatorial_ms = absOmegaRot * a.radiusM;
    a.riverCentrifugalQ = (a.vInflowSurface_ms > 0.0)
        ? (2.0 * (a.vEquatorial_ms / a.vInflowSurface_ms) * (a.vEquatorial_ms / a.vInflowSurface_ms))
        : 0.0;
    const double momC = engine.momentOfInertiaC()[bodyIdx];
    a.derivedJ2 = deriveHydrostaticJ2(absOmegaRot, a.radiusM, muSelf, momC);
    a.polarFlatteningF = 1.5 * a.derivedJ2 + 0.5 * a.riverCentrifugalQ;
    a.spinAngMom_kgm2s = momC * engine.mass()[bodyIdx] * a.radiusM * a.radiusM * absOmegaRot;

    const double eqKinLoad = (a.vEquatorial_ms * a.vEquatorial_ms) / c2;
    const double eqLagFrac = oneMinusSqrtOneMinus(eqKinLoad);
    a.clockLagEquatorExtra_ppb = eqLagFrac * 1.0e9;
    a.clockLagEquatorExtra_usDay = eqLagFrac * 86400.0 * 1.0e6;

    // Orbital River Metric, Bandwidth Load, 1PN Precession & Parent J2 Quadrupole Precession
    if (parentIdx >= 0 && kep.r_m > 1.0) {
        const double muParent = engine.mu()[parentIdx];
        a.gParentAtOrbit_ms2 = muParent / (kep.r_m * kep.r_m);
        a.vInflowOrbit_ms = std::sqrt(2.0 * a.gParentAtOrbit_ms2 * kep.r_m);
        const double volOrb = 4.0 * PI * kep.r_m * kep.r_m * a.vInflowOrbit_ms;
        a.pixelPurgeParentAtOrbit_s = volOrb / V_NODE;

        const double gLoadSq = (2.0 * muParent) / (c2 * kep.r_m);
        const double vLoadSq = (kep.v_ms * kep.v_ms) / c2;
        a.omegaOrbit = gLoadSq + vLoadSq;
        const double lagOrbFrac = oneMinusSqrtOneMinus(a.omegaOrbit);
        a.clockLagOrbit_ppb = lagOrbFrac * 1.0e9;
        a.clockLagOrbit_usDay = lagOrbFrac * 86400.0 * 1.0e6;

        if (kep.bound && kep.period_days > 0.0 && muParent > 0.0) {
            const double pSemi = kep.a_m * (1.0 - kep.e * kep.e);
            if (relativisticEnabled && pSemi > 1.0) {
                const double radPerOrbit = (6.0 * PI * muParent) / (c2 * pSemi);
                const double orbitsPerCentury = 36525.0 / kep.period_days;
                a.precessionArcsecCentury = radPerOrbit * ARCSEC_PER_RAD * orbitsPerCentury;
            }
            const double j2Parent = engine.j2()[parentIdx];
            const double rParent  = engine.radius()[parentIdx];
            if (j2Parent > 0.0 && rParent > 0.0 && pSemi > 1.0) {
                const double rx = snap.pos[3 * bodyIdx + 0] - snap.pos[3 * parentIdx + 0];
                const double ry = snap.pos[3 * bodyIdx + 1] - snap.pos[3 * parentIdx + 1];
                const double rz = snap.pos[3 * bodyIdx + 2] - snap.pos[3 * parentIdx + 2];
                const double vx = snap.vel[3 * bodyIdx + 0] - snap.vel[3 * parentIdx + 0];
                const double vy = snap.vel[3 * bodyIdx + 1] - snap.vel[3 * parentIdx + 1];
                const double vz = snap.vel[3 * bodyIdx + 2] - snap.vel[3 * parentIdx + 2];
                const double hx = ry * vz - rz * vy;
                const double hy = rz * vx - rx * vz;
                const double hz = rx * vy - ry * vx;
                const double hMag = std::max(std::sqrt(hx * hx + hy * hy + hz * hz), 1e-12);
                const double sx = engine.spinAxisX()[parentIdx];
                const double sy = engine.spinAxisY()[parentIdx];
                const double sz = engine.spinAxisZ()[parentIdx];
                const double cosIncEq = std::clamp((hx * sx + hy * sy + hz * sz) / hMag, -1.0, 1.0);
                const double ratio = rParent / pSemi;
                const double orbitsPerYr = 365.25 / kep.period_days;
                const double baseDegYr = 1.5 * 360.0 * orbitsPerYr * j2Parent * ratio * ratio;
                a.j2NodalRegressionDegYr   = -baseDegYr * cosIncEq;
                a.j2ApsidalPrecessionDegYr = 0.5 * baseDegYr * (5.0 * cosIncEq * cosIncEq - 1.0);
            }
        }
    } else {
        // Root body (Sun): kinematic drift relative to barycenter
        const double vx = snap.vel[3 * bodyIdx + 0];
        const double vy = snap.vel[3 * bodyIdx + 1];
        const double vz = snap.vel[3 * bodyIdx + 2];
        a.omegaOrbit = (vx * vx + vy * vy + vz * vz) / c2;
        const double lagOrbFrac = oneMinusSqrtOneMinus(a.omegaOrbit);
        a.clockLagOrbit_ppb = lagOrbFrac * 1.0e9;
        a.clockLagOrbit_usDay = lagOrbFrac * 86400.0 * 1.0e6;
    }

    return a;
}

CoarseGrainAudit computeCoarseGrainAudit(double probeRadiusAU,
                                         const HeavySnapshot& snap,
                                         const SubstrateEngine& engine) {
    CoarseGrainAudit cg{};
    cg.probeRadiusAU = std::max(probeRadiusAU, 1e-4);
    const double R = cg.probeRadiusAU * AU;
    cg.probeVolumeM3 = (4.0 / 3.0) * PI * R * R * R;

    const double sx = snap.pos[0], sy = snap.pos[1], sz = snap.pos[2];
    for (int i = 0; i < engine.numHeavy(); ++i) {
        const double dx = snap.pos[3 * i + 0] - sx;
        const double dy = snap.pos[3 * i + 1] - sy;
        const double dz = snap.pos[3 * i + 2] - sz;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist <= R) {
            cg.enclosedNodes += engine.nodes()[i];
            if (engine.density()[i] >= RHO_GC) {
                cg.enclosedCompiledNodes += engine.nodes()[i];
            }
        }
    }

    cg.meanRhoNodes = cg.enclosedNodes / cg.probeVolumeM3;
    cg.exceedsGCCeiling = (cg.meanRhoNodes >= RHO_GC);

    // Local pointer-recycling expansion boost Delta H (Paper 2, Eq. 125)
    const double rhoCompiled = cg.enclosedCompiledNodes / cg.probeVolumeM3;
    if (cg.exceedsGCCeiling && rhoCompiled > 0.0) {
        const double deltaH_sInv = std::sqrt((8.0 * PI * K_OMEGA * rhoCompiled) / 3.0);
        cg.localDeltaH_kmsMpc = (deltaH_sInv * MPC_METERS) / 1000.0;
    }

    // Gravitational field energy density at sphere boundary: u_g = m_p * g_Omega^2 / (8*pi*K_Omega)
    const double gProbe = (cg.enclosedCompiledNodes * K_OMEGA) / (R * R);
    cg.gravFieldEnergyDensity_Jm3 = (M_P * gProbe * gProbe) / (8.0 * PI * K_OMEGA);

    return cg;
}

LagrangePoints computeLagrangePoints(int bodyIdx, int parentIdx,
                                     const HeavySnapshot& snap,
                                     const SubstrateEngine& engine) {
    LagrangePoints lp{};
    if (parentIdx < 0 || bodyIdx < 0 || bodyIdx >= engine.numHeavy() || parentIdx >= engine.numHeavy()) {
        return lp;
    }

    const double muParent = engine.mu()[parentIdx];
    const double muBody   = engine.mu()[bodyIdx];
    // Require both bodies to be GC-compiled (Theta = 1): if either decompiles to Theta = 0, Lagrange points vanish!
    if (muParent <= 0.0 || muBody <= 0.0) {
        return lp;
    }

    const double px = snap.pos[3 * parentIdx + 0];
    const double py = snap.pos[3 * parentIdx + 1];
    const double pz = snap.pos[3 * parentIdx + 2];

    const double rx = snap.pos[3 * bodyIdx + 0] - px;
    const double ry = snap.pos[3 * bodyIdx + 1] - py;
    const double rz = snap.pos[3 * bodyIdx + 2] - pz;

    const double vx = snap.vel[3 * bodyIdx + 0] - snap.vel[3 * parentIdx + 0];
    const double vy = snap.vel[3 * bodyIdx + 1] - snap.vel[3 * parentIdx + 1];
    const double vz = snap.vel[3 * bodyIdx + 2] - snap.vel[3 * parentIdx + 2];

    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (r < 1.0) return lp;

    const double ux = rx / r, uy = ry / r, uz = rz / r;

    // Orbital angular momentum normal h_hat = normalize(r_vec x v_vec)
    const double hx = ry * vz - rz * vy;
    const double hy = rz * vx - rx * vz;
    const double hz = rx * vy - ry * vx;
    const double hLen = std::sqrt(hx * hx + hy * hy + hz * hz);
    if (hLen < 1e-12) return lp;

    const double hnx = hx / hLen, hny = hy / hLen, hnz = hz / hLen;

    // Transverse unit vector t_hat = h_hat x u_hat (points +90 deg ahead in the orbital plane)
    const double tx = hny * uz - hnz * uy;
    const double ty = hnz * ux - hnx * uz;
    const double tz = hnx * uy - hny * ux;

    const double alpha = muBody / (muParent + muBody);
    const double oneMinusAlpha = 1.0 - alpha;
    lp.alpha = alpha;

    const double rH = std::cbrt(alpha / 3.0);
    const double eps = std::max(1.0e-14, 1.0e-6 * rH);

    // Exact Euler quintic root finder:
    //   f(x) = x - (1 - alpha)*(x + alpha)/|x + alpha|^3 - alpha*(x - (1 - alpha))/|x - (1 - alpha)|^3 = 0
    //   f'(x) = 1 + 2*(1 - alpha)/|x + alpha|^3 + 2*alpha/|x - (1 - alpha)|^3  (> 1 everywhere)
    auto solveCollinear = [&](double xInit, double xMin, double xMax) -> double {
        double x = std::clamp(xInit, xMin, xMax);
        for (int iter = 0; iter < 12; ++iter) {
            const double d1 = x + alpha;
            const double d2 = x - oneMinusAlpha;
            const double a1 = std::abs(d1);
            const double a2 = std::abs(d2);
            if (a1 < 1.0e-15 || a2 < 1.0e-15) break;
            const double a1_3 = a1 * a1 * a1;
            const double a2_3 = a2 * a2 * a2;
            const double f  = x - oneMinusAlpha * (d1 / a1_3) - alpha * (d2 / a2_3);
            const double df = 1.0 + 2.0 * oneMinusAlpha / a1_3 + 2.0 * alpha / a2_3;
            const double step = f / df;
            if (!std::isfinite(step)) break;
            x = std::clamp(x - step, xMin, xMax);
            if (std::abs(step) < 1.0e-14) break;
        }
        return x;
    };

    const double x1 = solveCollinear(oneMinusAlpha - rH * (1.0 - rH / 3.0), -alpha + eps, oneMinusAlpha - eps);
    const double x2 = solveCollinear(oneMinusAlpha + rH * (1.0 + rH / 3.0), oneMinusAlpha + eps, 10.0);
    const double x3 = solveCollinear(-1.0 + (7.0 / 12.0) * alpha, -10.0, -alpha - eps);

    auto setCollinear = [&](int idx, double x) {
        const double s = x + alpha;
        lp.pos_m[idx][0] = px + s * rx;
        lp.pos_m[idx][1] = py + s * ry;
        lp.pos_m[idx][2] = pz + s * rz;
    };
    setCollinear(0, x1); // L1
    setCollinear(1, x2); // L2
    setCollinear(2, x3); // L3

    // Exact equilateral triangular points L4 (+60 deg ahead) and L5 (-60 deg behind)
    constexpr double HALF_SQRT3 = 0.86602540378443864676;
    lp.pos_m[3][0] = px + 0.5 * rx + HALF_SQRT3 * r * tx;
    lp.pos_m[3][1] = py + 0.5 * ry + HALF_SQRT3 * r * ty;
    lp.pos_m[3][2] = pz + 0.5 * rz + HALF_SQRT3 * r * tz;

    lp.pos_m[4][0] = px + 0.5 * rx - HALF_SQRT3 * r * tx;
    lp.pos_m[4][1] = py + 0.5 * ry - HALF_SQRT3 * r * ty;
    lp.pos_m[4][2] = pz + 0.5 * rz - HALF_SQRT3 * r * tz;

    lp.valid = true;
    return lp;
}

} // namespace substrate

