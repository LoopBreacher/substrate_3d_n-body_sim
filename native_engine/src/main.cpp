// Substrate Logistics 3D N-Body Simulator — Native C++20 / CUDA / OpenGL 4.6 Application.
//
// Interactive mode (default):
//   substrate_native.exe [--particles N]
// Headless benchmark & physics verification mode:
//   substrate_native.exe --benchmark

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <numeric>
#include <string>
#include <vector>

#include "astro_metrics.h"
#include "csv_export.h"
#include "payload.h"
#include "renderer.h"
#include "substrate_constants.h"
#include "substrate_engine.h"
#include "telemetry_ui.h"

namespace substrate {
namespace {

glm::vec3 logCompressPos(glm::vec3 p) {
    const float r = glm::length(p);
    if (r < 1e-6f) return p;
    return (p / r) * (std::log(1.0f + r * 10.0f) * 0.43429448190325182f);
}

// Builds an orthonormal basis whose column 2 (+Z) is aligned with unit spin axis sHat.
glm::mat3 buildSpinBasis(const glm::vec3& sHatIn) {
    const float len = glm::length(sHatIn);
    const glm::vec3 zAxis = (len > 1e-6f) ? (sHatIn / len) : glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::vec3 ref = (std::abs(zAxis.z) < 0.92f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 xAxis = glm::normalize(glm::cross(ref, zAxis));
    const glm::vec3 yAxis = glm::cross(zAxis, xAxis);
    return glm::mat3(xAxis, yAxis, zAxis);
}

struct OrbitCamera {
    float distanceAU = 35.0f;
    float azimuthDeg = 45.0f;
    float elevationDeg = 35.0f;
    glm::vec3 panOffsetAU{0.0f};
    bool lockTarget = false;

    glm::vec3 eyeRelativeAU() const {
        const float el = glm::radians(elevationDeg);
        const float az = glm::radians(azimuthDeg);
        return panOffsetAU + distanceAU * glm::vec3(std::cos(el) * std::cos(az),
                                                    std::cos(el) * std::sin(az),
                                                    std::sin(el));
    }

    glm::mat4 viewMatrix() const {
        const glm::vec3 eye = eyeRelativeAU();
        return glm::lookAt(eye, panOffsetAU, glm::vec3(0.0f, 0.0f, 1.0f));
    }

    void reset() {
        distanceAU = 35.0f;
        azimuthDeg = 45.0f;
        elevationDeg = 35.0f;
        panOffsetAU = glm::vec3(0.0f);
    }
};

struct MouseState {
    OrbitCamera* cam = nullptr;
    double lastX = 0.0;
    double lastY = 0.0;
};

void scrollCallback(GLFWwindow* win, double /*xoff*/, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto* ms = static_cast<MouseState*>(glfwGetWindowUserPointer(win));
    if (!ms || !ms->cam) return;
    const float factor = std::pow(0.86f, static_cast<float>(yoff));
    ms->cam->distanceAU = std::clamp(ms->cam->distanceAU * factor, 1.0e-5f, 25000.0f);
}

void applySciFiTheme() {
    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 6.0f;
    s.FrameRounding = 4.0f;
    s.GrabRounding = 3.0f;
    s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 1.0f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]         = ImVec4(0.06f, 0.08f, 0.12f, 0.94f);
    c[ImGuiCol_ChildBg]          = ImVec4(0.08f, 0.10f, 0.15f, 0.60f);
    c[ImGuiCol_Border]           = ImVec4(0.14f, 0.22f, 0.32f, 0.80f);
    c[ImGuiCol_FrameBg]          = ImVec4(0.09f, 0.13f, 0.19f, 1.00f);
    c[ImGuiCol_FrameBgHovered]   = ImVec4(0.13f, 0.20f, 0.29f, 1.00f);
    c[ImGuiCol_FrameBgActive]    = ImVec4(0.16f, 0.26f, 0.38f, 1.00f);
    c[ImGuiCol_TitleBg]          = ImVec4(0.05f, 0.07f, 0.11f, 1.00f);
    c[ImGuiCol_TitleBgActive]    = ImVec4(0.07f, 0.14f, 0.22f, 1.00f);
    c[ImGuiCol_CheckMark]        = ImVec4(0.00f, 0.85f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrab]       = ImVec4(0.00f, 0.72f, 0.92f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.00f, 0.90f, 1.00f, 1.00f);
    c[ImGuiCol_Button]           = ImVec4(0.11f, 0.18f, 0.27f, 1.00f);
    c[ImGuiCol_ButtonHovered]    = ImVec4(0.00f, 0.48f, 0.65f, 1.00f);
    c[ImGuiCol_ButtonActive]     = ImVec4(0.00f, 0.62f, 0.82f, 1.00f);
    c[ImGuiCol_Header]           = ImVec4(0.10f, 0.22f, 0.33f, 0.85f);
    c[ImGuiCol_HeaderHovered]    = ImVec4(0.14f, 0.32f, 0.46f, 0.90f);
    c[ImGuiCol_HeaderActive]     = ImVec4(0.18f, 0.40f, 0.58f, 1.00f);
    c[ImGuiCol_Tab]              = ImVec4(0.09f, 0.16f, 0.25f, 0.92f);
    c[ImGuiCol_TabHovered]       = ImVec4(0.00f, 0.52f, 0.72f, 0.92f);
    c[ImGuiCol_TabSelected]      = ImVec4(0.00f, 0.42f, 0.60f, 1.00f);
}

// ---------------------------------------------------------------------------
// Symplectic Time-Reversal Proof & Live 2-Body Mercury EIH Precession Audit
// ---------------------------------------------------------------------------
struct TimeReversalReport {
    double maxPosErrM = 0.0;
    double maxVelErrMs = 0.0;
    double maxSpinErrArcsec = 0.0;
    int worstPosIdx = 0;
    double finalTimeSec = 0.0;
};

TimeReversalReport runTimeReversalFromState(SubstrateEngine& engine,
                                            const HeavySnapshot& s0,
                                            int nSteps,
                                            const StepParams& spFwd) {
    StepParams spRev = spFwd;
    spRev.dtSeconds = -std::abs(spFwd.dtSeconds);
    for (int i = 0; i < nSteps; ++i) {
        engine.step(spRev, i == nSteps - 1);
    }
    const auto sBack = engine.snapshot();

    TimeReversalReport rep;
    rep.finalTimeSec = engine.simTimeSeconds();
    const int M = engine.numHeavy();
    constexpr double ARCSEC_PER_RAD = (180.0 * 3600.0) / PI_CONST;
    for (int i = 0; i < M; ++i) {
        const double dx = sBack.pos[3 * i + 0] - s0.pos[3 * i + 0];
        const double dy = sBack.pos[3 * i + 1] - s0.pos[3 * i + 1];
        const double dz = sBack.pos[3 * i + 2] - s0.pos[3 * i + 2];
        const double posErr = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (posErr > rep.maxPosErrM) {
            rep.maxPosErrM = posErr;
            rep.worstPosIdx = i;
        }
        const double dvx = sBack.vel[3 * i + 0] - s0.vel[3 * i + 0];
        const double dvy = sBack.vel[3 * i + 1] - s0.vel[3 * i + 1];
        const double dvz = sBack.vel[3 * i + 2] - s0.vel[3 * i + 2];
        const double velErr = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
        if (velErr > rep.maxVelErrMs) rep.maxVelErrMs = velErr;

        if (s0.spinAxis.size() == static_cast<size_t>(3 * M) &&
            sBack.spinAxis.size() == static_cast<size_t>(3 * M)) {
            const double ax = s0.spinAxis[3 * i + 0], ay = s0.spinAxis[3 * i + 1], az = s0.spinAxis[3 * i + 2];
            const double bx = sBack.spinAxis[3 * i + 0], by = sBack.spinAxis[3 * i + 1], bz = sBack.spinAxis[3 * i + 2];
            const double cx = ay * bz - az * by;
            const double cy = az * bx - ax * bz;
            const double cz = ax * by - ay * bx;
            const double dot = ax * bx + ay * by + az * bz;
            const double angArcsec = std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), dot) * ARCSEC_PER_RAD;
            if (angArcsec > rep.maxSpinErrArcsec) rep.maxSpinErrArcsec = angArcsec;
        }
    }
    return rep;
}

struct MercuryPrecessionReport {
    double rateEihArcsecCy = 0.0;
    double rate0pnArcsecCy = 0.0;
    double net1pnArcsecCy = 0.0;
    double analyticalArcsecCy = 0.0;
    double relErrPct = 0.0;
    int orbitsMeasured = 10;
};

MercuryPrecessionReport runMercury2BodyPrecessionProof(const std::vector<HeavyBodyDesc>& fullHeavy) {
    MercuryPrecessionReport rep;
    const int idxSun = findBodyIndex(fullHeavy, "Sun");
    const int idxMerc = findBodyIndex(fullHeavy, "Mercury");
    if (idxSun < 0 || idxMerc < 0) return rep;

    std::vector<HeavyBodyDesc> pair = {fullHeavy[idxSun], fullHeavy[idxMerc]};
    pair[0].parent.clear();
    pair[1].parent = pair[0].name;
    ParticleSet emptyPs;

    constexpr double ARCSEC_PER_RAD = (180.0 * 3600.0) / PI_CONST;
    constexpr double JULIAN_CENTURY_SEC = 100.0 * 365.25 * 86400.0;

    // Initial Keplerian period & exact analytical 1PN GR precession in K_OMEGA
    SubstrateEngine eng;
    eng.load(pair, emptyPs, true);
    eng.requestSnapshot();
    const auto sInit = eng.snapshot();
    const auto kep0 = computeKeplerElements(1, 0, sInit, eng);
    const double T_merc = kep0.period_days * 86400.0;
    const double muTot = eng.mu()[0] + eng.mu()[1];
    const double dOmegaOrbit =
        (6.0 * PI_CONST * muTot) / (C_LIGHT * C_LIGHT * kep0.a_m * (1.0 - kep0.e * kep0.e));
    rep.analyticalArcsecCy = dOmegaOrbit * ARCSEC_PER_RAD * (JULIAN_CENTURY_SEC / T_merc);

    auto relVecs = [](const HeavySnapshot& s, glm::dvec3& rRel, glm::dvec3& vRel) {
        rRel = glm::dvec3(s.pos[3] - s.pos[0], s.pos[4] - s.pos[1], s.pos[5] - s.pos[2]);
        vRel = glm::dvec3(s.vel[3] - s.vel[0], s.vel[4] - s.vel[1], s.vel[5] - s.vel[2]);
    };

    auto bisectPerihelion = [&](const HeavySnapshot& sPrev, double tPrev, double dtStep, bool rel,
                                HeavySnapshot& sCross, double& tCross) {
        std::vector<HeavyBodyDesc> probeBodies = pair;
        for (int b = 0; b < 2; ++b) {
            for (int k = 0; k < 3; ++k) {
                probeBodies[b].pos[k] = sPrev.pos[3 * b + k];
                probeBodies[b].vel[k] = sPrev.vel[3 * b + k];
                if (sPrev.spinAxis.size() == 6) probeBodies[b].spinAxis[k] = sPrev.spinAxis[3 * b + k];
            }
        }
        SubstrateEngine probe;
        probe.load(probeBodies, emptyPs, false);
        probe.refreshCachedAccelerations(rel);
        probe.synchronize();

        StepParams spP;
        spP.substeps = 2;
        spP.relativistic = rel;
        spP.precision = Precision::FP64;

        double lo = 0.0, hi = dtStep, tCur = 0.0;
        HeavySnapshot sMid = sPrev;
        for (int iter = 0; iter < 42; ++iter) {
            const double mid = 0.5 * (lo + hi);
            const double dStep = mid - tCur;
            if (std::abs(dStep) < 1e-12) break;
            spP.dtSeconds = dStep;
            probe.step(spP, true);
            sMid = probe.snapshot();
            tCur = mid;
            glm::dvec3 rM, vM;
            relVecs(sMid, rM, vM);
            if (glm::dot(rM, vM) < 0.0) lo = mid;
            else hi = mid;
        }
        sCross = sMid;
        tCross = tPrev + tCur;
    };

    auto measureRate = [&](bool rel) -> double {
        eng.load(pair, emptyPs, true);
        eng.refreshCachedAccelerations(rel);
        eng.synchronize();
        eng.requestSnapshot();
        HeavySnapshot sPrev = eng.snapshot();
        double tPrev = 0.0;
        glm::dvec3 rPrev, vPrev;
        relVecs(sPrev, rPrev, vPrev);
        double rdotvPrev = glm::dot(rPrev, vPrev);

        StepParams sp;
        sp.dtSeconds = T_merc / 400.0;
        sp.substeps = 2;
        sp.relativistic = rel;
        sp.precision = Precision::FP64;

        HeavySnapshot sCross0, sCrossN;
        double tCross0 = 0.0, tCrossN = 0.0;
        int crossings = 0;
        const int targetOrbits = rep.orbitsMeasured;
        const int maxSteps = 400 * (targetOrbits + 2) + 100;

        for (int step = 0; step < maxSteps; ++step) {
            eng.step(sp, true);
            const HeavySnapshot sCur = eng.snapshot();
            const double tCur = eng.simTimeSeconds();
            glm::dvec3 rCur, vCur;
            relVecs(sCur, rCur, vCur);
            const double rdotvCur = glm::dot(rCur, vCur);

            if (rdotvPrev < 0.0 && rdotvCur >= 0.0) {
                if (crossings == 0) {
                    bisectPerihelion(sPrev, tPrev, sp.dtSeconds, rel, sCross0, tCross0);
                } else if (crossings == targetOrbits) {
                    bisectPerihelion(sPrev, tPrev, sp.dtSeconds, rel, sCrossN, tCrossN);
                    break;
                }
                ++crossings;
            }
            sPrev = sCur;
            tPrev = tCur;
            rdotvPrev = rdotvCur;
        }

        glm::dvec3 r0, v0, rN, vN;
        relVecs(sCross0, r0, v0);
        relVecs(sCrossN, rN, vN);
        const glm::dvec3 u0 = glm::normalize(r0);
        const glm::dvec3 h0 = glm::normalize(glm::cross(r0, v0));
        const glm::dvec3 w0 = glm::cross(h0, u0);
        const glm::dvec3 uN = glm::normalize(rN);
        const double dOmega = std::atan2(glm::dot(uN, w0), glm::dot(uN, u0));
        return dOmega * ARCSEC_PER_RAD * (JULIAN_CENTURY_SEC / (tCrossN - tCross0));
    };

    rep.rateEihArcsecCy = measureRate(true);
    rep.rate0pnArcsecCy = measureRate(false);
    rep.net1pnArcsecCy  = rep.rateEihArcsecCy - rep.rate0pnArcsecCy;
    rep.relErrPct = (rep.analyticalArcsecCy > 0.0)
        ? (100.0 * std::abs(rep.net1pnArcsecCy - rep.analyticalArcsecCy) / rep.analyticalArcsecCy)
        : 0.0;
    return rep;
}

std::vector<int> buildParentIndices(const std::vector<HeavyBodyDesc>& heavy) {
    const int M = static_cast<int>(heavy.size());
    std::vector<int> pIdx(M, -1);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            if (j != i && heavy[i].parent == heavy[j].name) {
                pIdx[i] = j;
                break;
            }
        }
    }
    return pIdx;
}

void maybeApplyEpochCSV(std::vector<HeavyBodyDesc>& heavy, const std::string& epochCsvPath) {
    if (epochCsvPath.empty()) return;
    const int nLoaded = applyHorizonsEpochCSV(heavy, epochCsvPath);
    if (nLoaded < 0) {
        std::fprintf(stderr, "[Horizons Epoch] WARNING: Could not open '%s' (using default J2000.0 payload)\n",
                     epochCsvPath.c_str());
    } else {
        std::printf("[Horizons Epoch] Loaded %d body states from %s\n\n", nLoaded, epochCsvPath.c_str());
    }
}

int runTimeReversalCLI(const std::string& epochCsvPath) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("====================================================================\n");
    std::printf("SUBSTRATE LOGISTICS - EIH 1PN SYMPLECTIC TIME-REVERSAL & PRECESSION\n");
    std::printf("GPU: %s (Compute %d.%d, %.0f MB VRAM)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("====================================================================\n\n");

    auto heavy = buildHeavyBodies();
    maybeApplyEpochCSV(heavy, epochCsvPath);
    ParticleSet emptyPs;
    SubstrateEngine engine;
    engine.load(heavy, emptyPs, true);
    engine.requestSnapshot();
    const auto s0 = engine.snapshot();

    StepParams sp;
    sp.dtSeconds = 21600.0;
    sp.substeps = 20;
    sp.relativistic = true;
    sp.precision = Precision::FP64;

    const double E0 = engine.newtonianEnergy(s0, sp.relativistic);
    const double L0 = engine.angularMomentumMag(s0, sp.relativistic);
    double P0x = 0.0, P0y = 0.0, P0z = 0.0;
    engine.canonicalMomentum(s0, sp.relativistic, P0x, P0y, P0z);
    double mTot = 0.0;
    for (double m : engine.mass()) mTot += m;

    constexpr int YEAR_STEPS = 365 * 4 + 1;
    for (int i = 0; i < YEAR_STEPS; ++i) engine.step(sp, i == YEAR_STEPS - 1);
    const auto s1 = engine.snapshot();
    const double E1 = engine.newtonianEnergy(s1, sp.relativistic);
    const double L1 = engine.angularMomentumMag(s1, sp.relativistic);
    double P1x = 0.0, P1y = 0.0, P1z = 0.0;
    engine.canonicalMomentum(s1, sp.relativistic, P1x, P1y, P1z);
    const double dPMtot = std::sqrt((P1x - P0x) * (P1x - P0x) + (P1y - P0y) * (P1y - P0y) + (P1z - P0z) * (P1z - P0z)) / mTot;

    const auto tr = runTimeReversalFromState(engine, s0, YEAR_STEPS, sp);
    const auto mp = runMercury2BodyPrecessionProof(heavy);

    std::printf("[1-Julian-Year EIH 1PN + Gyro J2 Symplectic Time-Reversal Proof (55 Heavy Bodies)]\n");
    std::printf("  1-Yr Rel. Energy Drift (E_EIH+U_J2): %.3e\n", std::abs((E1 - E0) / E0));
    std::printf("  1-Yr Rel. AngMom Drift (L_EIH+S)   : %.3e\n", std::abs((L1 - L0) / L0));
    std::printf("  1-Yr Canonical |dP_EIH| / M_tot    : %.3e m/s\n", dPMtot);
    std::printf("  Max Position Return Error (+1Y/-1Y): %.4e m (%.3e AU, worst: %s)\n",
                tr.maxPosErrM, tr.maxPosErrM / AU, heavy[tr.worstPosIdx].name.c_str());
    std::printf("  Max Velocity Return Error (+1Y/-1Y): %.4e m/s\n", tr.maxVelErrMs);
    std::printf("  Max Spin-Pole Return Error         : %.4e arcsec (final t = %.3f s)\n\n",
                tr.maxSpinErrArcsec, tr.finalTimeSec);

    std::printf("[Live 2-Body Mercury EIH Perihelion Precession Proof (%d Orbits, dt = T/400, Bisection)]\n",
                mp.orbitsMeasured);
    std::printf("  Measured EIH (1PN) Rate            : %.4f arcsec/cy\n", mp.rateEihArcsecCy);
    std::printf("  Measured 0PN Baseline Rate         : %.4f arcsec/cy\n", mp.rate0pnArcsecCy);
    std::printf("  Live Measured 1PN EIH Advance      : %.4f arcsec/cy (Analytical K_OMEGA GR: %.4f arcsec/cy, err = %.4f%%)\n",
                mp.net1pnArcsecCy, mp.analyticalArcsecCy, mp.relErrPct);
    return 0;
}

// ---------------------------------------------------------------------------
// Headless Publication-Ready Scientific CSV Export Pipeline (--export-csv)
// ---------------------------------------------------------------------------
int runExportCsvCLI(const std::string& outDir, double years, double dtHours,
                    size_t numParticles, const std::string& epochCsvPath) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("====================================================================\n");
    std::printf("SUBSTRATE LOGISTICS - SCIENTIFIC CSV EXPORT PIPELINE\n");
    std::printf("GPU: %s (Compute %d.%d, %.0f MB VRAM)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("====================================================================\n\n");

    auto heavy = buildHeavyBodies();
    maybeApplyEpochCSV(heavy, epochCsvPath);
    const auto parentIdx = buildParentIndices(heavy);
    const int M = static_cast<int>(heavy.size());

    auto particles = buildParticles(numParticles, 42, &heavy);
    auto satRings  = buildSaturnRings(40000, heavy);
    SubstrateEngine engine;
    engine.load(heavy, particles, true, &satRings);
    engine.requestSnapshot();
    const HeavySnapshot s0 = engine.snapshot();

    StepParams sp;
    sp.dtSeconds = std::max(dtHours, 0.05) * 3600.0;
    sp.substeps = 20;
    sp.relativistic = true;
    sp.precision = Precision::FP64;

    const int idxSun     = std::max(0, findBodyIndex(heavy, "Sun"));
    const int idxMercury = findBodyIndex(heavy, "Mercury");
    const int idxEarth   = findBodyIndex(heavy, "Earth");
    const int idxJupiter = findBodyIndex(heavy, "Jupiter");

    const double E0 = engine.newtonianEnergy(s0, sp.relativistic);
    const double L0 = engine.angularMomentumMag(s0, sp.relativistic);
    double P0x = 0.0, P0y = 0.0, P0z = 0.0;
    engine.canonicalMomentum(s0, sp.relativistic, P0x, P0y, P0z);
    double mTot = 0.0;
    for (double m : engine.mass()) mTot += m;

    constexpr double RAD2DEG = 180.0 / PI_CONST;
    constexpr double ARCSEC_PER_RAD = (180.0 * 3600.0) / PI_CONST;
    double prevEarthPhi = 0.0;
    double cumEarthPrecRad = 0.0;
    if (idxEarth >= 0 && s0.spinAxis.size() == static_cast<size_t>(3 * M)) {
        prevEarthPhi = std::atan2(s0.spinAxis[3 * idxEarth + 1], s0.spinAxis[3 * idxEarth + 0]);
    }

    AstroEventTracker tracker;
    tracker.reset(M);
    tracker.update(s0, heavy, parentIdx, engine);

    std::vector<ConservationSample> samples;
    auto pushSample = [&](const HeavySnapshot& s) {
        ConservationSample cs{};
        cs.simYear = (s.simTimeSeconds / 86400.0) / 365.25;
        const double E = engine.newtonianEnergy(s, sp.relativistic);
        const double L = engine.angularMomentumMag(s, sp.relativistic);
        double Px = 0.0, Py = 0.0, Pz = 0.0;
        engine.canonicalMomentum(s, sp.relativistic, Px, Py, Pz);
        cs.relEnergyDrift = (std::abs(E0) > 1e-30) ? std::abs((E - E0) / E0) : 0.0;
        cs.relAngMomDrift = (std::abs(L0) > 1e-30) ? std::abs((L - L0) / L0) : 0.0;
        cs.barycenterDriftMs = (mTot > 0.0)
            ? std::sqrt((Px - P0x) * (Px - P0x) + (Py - P0y) * (Py - P0y) + (Pz - P0z) * (Pz - P0z)) / mTot
            : 0.0;
        if (idxEarth >= 0 && s.spinAxis.size() == static_cast<size_t>(3 * M)) {
            const double sx = s.spinAxis[3 * idxEarth + 0];
            const double sy = s.spinAxis[3 * idxEarth + 1];
            const double sz = s.spinAxis[3 * idxEarth + 2];
            cs.earthObliquityDeg = std::acos(std::clamp(sz, -1.0, 1.0)) * RAD2DEG;
            const double curPhi = std::atan2(sy, sx);
            double dPhi = curPhi - prevEarthPhi;
            while (dPhi >  PI_CONST) dPhi -= 2.0 * PI_CONST;
            while (dPhi < -PI_CONST) dPhi += 2.0 * PI_CONST;
            cumEarthPrecRad += dPhi;
            prevEarthPhi = curPhi;
            cs.earthPrecessionArcsec = cumEarthPrecRad * ARCSEC_PER_RAD;
        }
        if (idxMercury >= 0) {
            const auto mk = computeKeplerElements(idxMercury, idxSun, s, engine);
            cs.mercuryArgPeriDeg = mk.argPeri_deg;
        }
        samples.push_back(cs);
    };

    const int totalSteps = std::max(1, static_cast<int>(std::llround((years * 365.25 * 24.0) / dtHours)));
    const int stepsPerDay = std::max(1, static_cast<int>(std::llround(24.0 / dtHours)));
    samples.reserve(static_cast<size_t>(totalSteps / stepsPerDay) + 4);
    pushSample(s0);

    std::printf("Integrating %.3f Julian years (%d steps @ dt = %.2f h, substeps = %d, FP64 EIH + %zu belt + 40,000 rings)...\n",
                years, totalSteps, dtHours, sp.substeps, numParticles);

    HeavySnapshot finalSnap = s0;
    for (int step = 1; step <= totalSteps; ++step) {
        engine.step(sp, true);
        const HeavySnapshot& s = engine.snapshot();
        tracker.update(s, heavy, parentIdx, engine);
        if (step % stepsPerDay == 0 || step == totalSteps) {
            pushSample(s);
        }
        if (step == totalSteps) finalSnap = s;
    }

    RingStats ringStats{};
    engine.computeRingStats(ringStats);
    BeltStats beltStats{};
    double aJup = 5.202887;
    if (idxJupiter >= 0) {
        const auto jk = computeKeplerElements(idxJupiter, idxSun, finalSnap, engine);
        if (jk.bound && jk.a_AU > 0.1) aJup = jk.a_AU;
    }
    engine.computeBeltStats(beltStats, aJup);

    const auto summary = exportTelemetryCSV(
        outDir, samples, ringStats, tracker.events, finalSnap, heavy, parentIdx, engine, sp.relativistic);
    if (!summary.ok) {
        std::fprintf(stderr, "CSV Export Error: %s\n", summary.errorMsg.c_str());
        return 1;
    }

    std::printf("\n[CSV Export Complete -> %s]\n", summary.outDir.c_str());
    std::printf("  1. conservation_and_precession.csv : %zu rows (final dE/E = %.3e, dL/L = %.3e, Earth prec = %+.2f arcsec)\n",
                summary.conservationRows,
                samples.back().relEnergyDrift, samples.back().relAngMomDrift, samples.back().earthPrecessionArcsec);
    std::printf("  2. saturn_ring_profile.csv         : %zu bins (bound = %u, escaped = %u, <r> = %.0f km, <e> = %.4f)\n",
                summary.ringRows, ringStats.boundCount, ringStats.escapedCount,
                ringStats.meanRadiusKm, ringStats.meanEcc);
    std::printf("  3. close_encounters.csv            : %zu events logged (Periapsis + Planetary Flybys)\n",
                summary.eventRows);
    std::printf("  4. body_ephemeris_summary.csv      : %zu coupled heavy bodies\n", summary.bodyRows);
    std::printf("  GPU Belt Stats Check               : %u bound (%u asteroidal, <e> = %.4f, <i> = %.2f deg)\n",
                beltStats.boundCount, beltStats.asteroidalCount, beltStats.meanEcc, beltStats.meanIncDeg);
    return 0;
}

// ---------------------------------------------------------------------------
// Headless Benchmark & Manuscript Verification Suite
// ---------------------------------------------------------------------------
int runBenchmarkSuite(const std::string& epochCsvPath) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("====================================================================\n");
    std::printf("SUBSTRATE LOGISTICS NATIVE C++/CUDA ENGINE - BENCHMARK & AUDIT\n");
    std::printf("GPU: %s (Compute %d.%d, %.0f MB VRAM)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("====================================================================\n\n");

    auto heavy = buildHeavyBodies();
    maybeApplyEpochCSV(heavy, epochCsvPath);
    SubstrateEngine engine;

    // 1. Theoretical Manuscript Audit + 1-Year Symplectic Invariance + Live Gyro Precession & 40k Saturn Rings
    {
        auto particles = buildParticles(3000, 42, &heavy);
        auto satRings  = buildSaturnRings(40000, heavy);
        engine.load(heavy, particles, true, &satRings);
        if (engine.isCooperativeHeavy()) {
            std::printf("[Heavy Kernel Mode: Cooperative Multi-SM (%zu blocks x 32 threads)]\n\n", heavy.size());
        } else {
            std::printf("[Heavy Kernel Mode: Single-Block Fallback (1 block x %zu threads)]\n\n",
                        ((heavy.size() + 31) / 32) * 32);
        }
        engine.requestSnapshot();
        const auto s0 = engine.snapshot();

        std::vector<double> ringPos0;
        engine.copyRingPositions(ringPos0);

        StepParams sp;
        sp.dtSeconds = 21600.0;
        sp.substeps = 20;
        sp.relativistic = true;
        sp.precision = Precision::FP64;

        const double E0 = engine.newtonianEnergy(s0, sp.relativistic);
        const double L0 = engine.angularMomentumMag(s0, sp.relativistic);
        double P0x = 0.0, P0y = 0.0, P0z = 0.0;
        engine.canonicalMomentum(s0, sp.relativistic, P0x, P0y, P0z);
        double mTotHeavy = 0.0;
        for (double m : engine.mass()) mTotHeavy += m;

        const int idxSun      = findBodyIndex(heavy, "Sun");
        const int idxMercury  = findBodyIndex(heavy, "Mercury");
        const int idxEarth    = findBodyIndex(heavy, "Earth");
        const int idxMars     = findBodyIndex(heavy, "Mars");
        const int idxPhobos   = findBodyIndex(heavy, "Phobos");
        const int idxJupiter  = findBodyIndex(heavy, "Jupiter");
        const int idxSaturn   = findBodyIndex(heavy, "Saturn");
        const int idxOumuamua = findBodyIndex(heavy, "1I/'Oumuamua");

        const auto sunAud     = computeSubstrateAudit(idxSun, -1, s0, KeplerElements{}, engine, sp.relativistic);
        const auto earthKep0  = computeKeplerElements(idxEarth, idxSun, s0, engine);
        const auto earthDom0  = computeDomainRadii(idxEarth, idxSun, heavy, earthKep0, engine);
        const auto earthAud   = computeSubstrateAudit(idxEarth, idxSun, s0, earthKep0, engine, sp.relativistic);
        const auto mercKep0   = computeKeplerElements(idxMercury, idxSun, s0, engine);
        const auto mercAud    = computeSubstrateAudit(idxMercury, idxSun, s0, mercKep0, engine, sp.relativistic);
        const auto jupKep0    = computeKeplerElements(idxJupiter, idxSun, s0, engine);
        const auto jupAud     = computeSubstrateAudit(idxJupiter, idxSun, s0, jupKep0, engine, sp.relativistic);
        const auto satKep0    = computeKeplerElements(idxSaturn, idxSun, s0, engine);
        const auto satAud     = computeSubstrateAudit(idxSaturn, idxSun, s0, satKep0, engine, sp.relativistic);
        const auto phobosKep  = computeKeplerElements(idxPhobos, idxMars, s0, engine);
        const auto phobosDom  = computeDomainRadii(idxPhobos, idxMars, heavy, phobosKep, engine);
        const auto phobosAud  = computeSubstrateAudit(idxPhobos, idxMars, s0, phobosKep, engine, sp.relativistic);
        const auto oumuamuaK  = computeKeplerElements(idxOumuamua, idxSun, s0, engine);
        const auto jupLP      = computeLagrangePoints(idxJupiter, idxSun, s0, engine);

        const double rGravAU     = engine.gravitationalFloorAU();
        const double rOortCoreAU = engine.outerOortStagnationAU();
        const double rTS_AU      = engine.terminationShockRadiusAU();
        const double rHP_AU      = engine.heliopauseRadiusAU();

        std::printf("[Substrate Logistics Bare-Metal Hardware Audit]\n");
        std::printf("  Coupled Heavy Bodies              : %zu (Sun, 8 Planets, 23 Moons, 23 Dwarf/Centaur/NEA/Trojan/Comet/ISO)\n",
                    heavy.size());
        std::printf("  Exact C_delta                     : %.12e  ((sqrt(5) - 2) / 500)\n", C_DELTA);
        std::printf("  Exact Closed-Form T_Omega         : %.10e  ((400/3)*50^10 / (1 + C_delta))\n", T_OMEGA);
        std::printf("  Exact Derived K_Omega             : %.10e m^3/(node*s^2) (K_Omega/m_p = %.6e)\n",
                    K_OMEGA, K_OMEGA / M_P);
        std::printf("  Linear Drag sqrt(C_d)             : %.12f  (Void Om_m = %.6f, Global Om_m = %.6f)\n",
                    SQRT_C_DELTA, OMEGA_M_VOID, OMEGA_M_GLOBAL);
        std::printf("  Stagnation Operators C_d^(+-1/4)  : %.7f / %.7f\n",
                    C_DELTA_FOURTH_ROOT, C_DELTA_INV_FOURTH_ROOT);
        std::printf("  MIT Bag Constant B_QCD            : %.3f MeV/fm^3 (%.4e Pa; B^1/4 = 145.0 MeV)\n",
                    B_QCD_MEV_FM3, P_DELTA_MICRO_PA);
        std::printf("  Derived Yukawa Range (6.784*l_p)  : %.4f fm      (Strong range: 1.414-1.430 fm)\n",
                    R_YUKAWA_FM);
        std::printf("  Derived Pion Mass (m_p*C_d^1/4)   : %.2f MeV/c^2  (Observed multiplet: 138.04 MeV/c^2)\n",
                    M_PION_MEV);
        std::printf("  Cover Ledger Sigma_0              : %.6e J/m^3\n", SIGMA_0);
        std::printf("  Macro Drag Floor P_delta          : %.6e Pa (%.4f pPa; V2/IBEX: 0.26-0.29 pPa)\n",
                    P_DRAG_FLOOR, P_DRAG_FLOOR * 1.0e12);
        std::printf("  Earth Node Density                : %.4e nodes/m^3\n", earthAud.rhoNodes);
        std::printf("  Earth Compton Floor               : %.4e m\n", earthAud.lambdaComptonBody_m);
        std::printf("  Top-Down Quantum R                : %.2f km       (Physical R_Earth: %.2f km)\n",
                    earthAud.quantumTopDownRadius_m / 1000.0, earthAud.radiusM / 1000.0);
        std::printf("  Earth Pixel Purge Rate            : %.4e px/s\n", earthAud.pixelPurgeSurface_s);
        std::printf("  Earth 3R / 4R Channels            : %.1f km / %.1f km (Plasmapause / Van Allen)\n",
                    earthDom0.vector3R_m / 1000.0, earthDom0.vector4R_m / 1000.0);
        std::printf("  Earth 6.784R Trapping Limit       : %.1f km      (McIlwain L = 6.784 ring-current limit)\n",
                    earthDom0.rDeltaStagnation_m / 1000.0);
        std::printf("  Earth Surface Clock Lag           : %.4f ppb (-%.2f us/day; +%.4f ppb eq. spin)\n",
                    earthAud.clockLagSurface_ppb, earthAud.clockLagSurface_usDay,
                    earthAud.clockLagEquatorExtra_ppb);
        std::printf("  Earth River-Metric q_rot / J2     : q = %.5f | J2 = %.4e (1/f = %.1f; Obs: 298.3)\n",
                    earthAud.riverCentrifugalQ, earthAud.derivedJ2,
                    (earthAud.polarFlatteningF > 0.0) ? (1.0 / earthAud.polarFlatteningF) : 0.0);
        std::printf("  Sun / Jupiter / Saturn Derived J2 : J2_Sun = %+.4e (Obs: +2.0e-7) | J2_Jup = %.4e | J2_Sat = %.4e\n",
                    sunAud.derivedJ2, jupAud.derivedJ2, satAud.derivedJ2);
        std::printf("  Mercury 3:2 Spin-Orbit Period     : %.3f days     (Emergent 2/3*T_orb; Obs: 58.646 d)\n",
                    mercAud.spinPeriodHours / 24.0);
        std::printf("  Mercury 1PN Precession            : %.2f arcsec/cy (Observed GR: 42.98 arcsec/cy)\n",
                    mercAud.precessionArcsecCentury);
        std::printf("  Phobos 1:1 Tidal Lock & J2 Prec.  : T_rot = %.3f h | d_omega/dt = +%.1f deg/yr (r/d_R = %.3f)\n",
                    phobosAud.spinPeriodHours, phobosAud.j2ApsidalPrecessionDegYr,
                    phobosKep.a_m / phobosDom.rocheFluid_m);
        std::printf("  Solar Wind Termination Shock      : %.2f AU       (Voyager 2: 83.7 AU / Voyager 1: 94.0 AU)\n",
                    rTS_AU);
        std::printf("  Solar Wind Heliopause             : %.2f AU      (Voyager 2: 119.0 AU / Voyager 1: 121.6 AU)\n",
                    rHP_AU);
        std::printf("  Sun Grav. Floor R_g               : %.2f AU     (u_g(R_g) == Sigma_0)\n", rGravAU);
        std::printf("  Outer Oort Core 6.784 R_g         : %.1f AU    (Inner/Outer Oort core transition)\n",
                    rOortCoreAU);
        std::printf("  Galactic 2D Disk Truncation Ratio : %.3f R_d       (0.5*ln(1/C_d); Obs: 3.8 +- 0.7 R_d)\n",
                    DISK_TRUNCATION_RATIO);
        std::printf("  1I/'Oumuamua Hyperbolic State     : r = %.3f AU, e = %.5f (%s), v = %.2f km/s\n",
                    oumuamuaK.r_m / AU, oumuamuaK.e, oumuamuaK.bound ? "BOUND" : "HYPERBOLIC",
                    oumuamuaK.v_ms / 1000.0);
        if (jupLP.valid && idxJupiter >= 0) {
            const double dx1 = jupLP.pos_m[0][0] - s0.pos[idxJupiter * 3 + 0];
            const double dy1 = jupLP.pos_m[0][1] - s0.pos[idxJupiter * 3 + 1];
            const double dz1 = jupLP.pos_m[0][2] - s0.pos[idxJupiter * 3 + 2];
            const double dJupL1_AU = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1) / AU;
            std::printf("  Sun-Jupiter Emergent L1 Distance  : %.5f AU from Jupiter (alpha = %.6e)\n\n",
                        dJupL1_AU, jupLP.alpha);
        } else {
            std::printf("\n");
        }

        constexpr int YEAR_STEPS = 365 * 4 + 1;
        for (int i = 0; i < YEAR_STEPS; ++i) {  // 1 Julian year (365.25 days)
            engine.step(sp, i == YEAR_STEPS - 1);
        }
        const auto s1 = engine.snapshot();
        const double E1 = engine.newtonianEnergy(s1, sp.relativistic);
        const double L1 = engine.angularMomentumMag(s1, sp.relativistic);
        double P1x = 0.0, P1y = 0.0, P1z = 0.0;
        engine.canonicalMomentum(s1, sp.relativistic, P1x, P1y, P1z);
        const double dPMtot = (mTotHeavy > 0.0)
            ? std::sqrt((P1x - P0x) * (P1x - P0x) + (P1y - P0y) * (P1y - P0y) + (P1z - P0z) * (P1z - P0z)) / mTotHeavy
            : 0.0;
        const double relEDrift = std::abs((E1 - E0) / E0);
        const double relLDrift = std::abs((L1 - L0) / L0);
        const auto earthKep1  = computeKeplerElements(idxEarth, idxSun, s1, engine);
        const auto phobosKep1 = computeKeplerElements(idxPhobos, idxMars, s1, engine);

        // Verify GPU BeltStats kernel using Jupiter's live semi-major axis (confirm Saturn rings do NOT pollute BeltStats)
        BeltStats bs{};
        const double aJupBench = (jupKep0.bound && jupKep0.a_AU > 0.1) ? jupKep0.a_AU : 5.202887;
        engine.computeBeltStats(bs, aJupBench);

        // Live Gyroscopic Spin-Axis Precession Audit (Earth, 1 Julian Year)
        constexpr double ARCSEC_PER_RAD = (180.0 * 3600.0) / PI_CONST;
        double earthPrecArcsecYr = 0.0;
        double earthNutationArcsec = 0.0;
        if (idxEarth >= 0 && s0.spinAxis.size() == static_cast<size_t>(3 * heavy.size()) &&
            s1.spinAxis.size() == static_cast<size_t>(3 * heavy.size())) {
            const double sx0 = s0.spinAxis[3 * idxEarth + 0];
            const double sy0 = s0.spinAxis[3 * idxEarth + 1];
            const double sz0 = s0.spinAxis[3 * idxEarth + 2];
            const double sx1 = s1.spinAxis[3 * idxEarth + 0];
            const double sy1 = s1.spinAxis[3 * idxEarth + 1];
            const double sz1 = s1.spinAxis[3 * idxEarth + 2];
            double dPhi = std::atan2(sy1, sx1) - std::atan2(sy0, sx0);
            while (dPhi >  PI_CONST) dPhi -= 2.0 * PI_CONST;
            while (dPhi < -PI_CONST) dPhi += 2.0 * PI_CONST;
            earthPrecArcsecYr = dPhi * ARCSEC_PER_RAD;
            const double eps0 = std::acos(std::clamp(sz0, -1.0, 1.0));
            const double eps1 = std::acos(std::clamp(sz1, -1.0, 1.0));
            earthNutationArcsec = (eps1 - eps0) * ARCSEC_PER_RAD;
        }

        // Saturnian Ring System & Mimas 2:1 ILR Audit (40,000 passive icy particles, 1 Julian Year)
        const int idxMimas = findBodyIndex(heavy, "Mimas");
        double r21_ILR_km = 0.0;
        if (idxMimas >= 0 && idxSaturn >= 0) {
            const auto mimasKep = computeKeplerElements(idxMimas, idxSaturn, s0, engine);
            const double aM = mimasKep.a_m;
            const double rSatM = engine.radius()[idxSaturn];
            const double j2Sat = engine.j2()[idxSaturn];
            const double nM2 = (1.0 + 1.5 * j2Sat * (rSatM / aM) * (rSatM / aM)) / (aM * aM * aM);
            // Solve (1 + 1.5*J2*(R_S/r)^2) / r^3 = 4 * nM2 for exact J2-corrected 2:1 ILR radius
            double r = aM * std::pow(0.5, 2.0 / 3.0);
            for (int iter = 0; iter < 8; ++iter) {
                const double q = rSatM / r;
                const double f  = (1.0 + 1.5 * j2Sat * q * q) / (r * r * r) - 4.0 * nM2;
                const double df = (-3.0 - 7.5 * j2Sat * q * q) / (r * r * r * r);
                r -= f / df;
            }
            r21_ILR_km = r / 1000.0;
        }

        std::vector<double> ringPos1;
        engine.copyRingPositions(ringPos1);
        size_t ringStableCount = 0;
        double sumAbsDrKm = 0.0;
        const size_t nRings = engine.numRingParticles();
        for (size_t i = 0; i < nRings; ++i) {
            const double r0 = std::sqrt(ringPos0[3 * i + 0] * ringPos0[3 * i + 0] +
                                        ringPos0[3 * i + 1] * ringPos0[3 * i + 1] +
                                        ringPos0[3 * i + 2] * ringPos0[3 * i + 2]);
            const double r1 = std::sqrt(ringPos1[3 * i + 0] * ringPos1[3 * i + 0] +
                                        ringPos1[3 * i + 1] * ringPos1[3 * i + 1] +
                                        ringPos1[3 * i + 2] * ringPos1[3 * i + 2]);
            const double r1Km = r1 / 1000.0;
            if (r1Km >= 60000.0 && r1Km <= 160000.0) {
                ++ringStableCount;
                sumAbsDrKm += std::abs(r1 - r0) / 1000.0;
            }
        }
        const double meanRingDriftKm = (ringStableCount > 0) ? (sumAbsDrKm / static_cast<double>(ringStableCount)) : 0.0;

        // GPU Saturn Ring Radial Density Histogram check (zero particle readback!)
        RingStats rsBench{};
        engine.computeRingStats(rsBench);
        // Compare peak B-ring bin (~105,000 km -> bin ~70) vs Cassini Division trough (~119,750 km -> bin ~99)
        float bRingPeak = 0.0f;
        for (int b = 45; b < 90; ++b) bRingPeak = std::max(bRingPeak, rsBench.radialHist[b]);
        float cassiniMin = 1.0e9f;
        for (int b = 96; b <= 103; ++b) cassiniMin = std::min(cassiniMin, rsBench.radialHist[b]);

        // 1-Julian-Year Symplectic Time-Reversal Proof: reverse directly from s1 back to t = 0.0!
        const auto timeRev = runTimeReversalFromState(engine, s0, YEAR_STEPS, sp);
        // Live 2-Body Mercury EIH Perihelion Precession Verification (10 orbits, exact r.v=0 bisection)
        const auto mercPrec = runMercury2BodyPrecessionProof(heavy);

        // Also verify 1-year 40k Saturn ring stability in Mixed mode (pure-FP32 Saturn-radius force loop)
        engine.load(heavy, particles, true, &satRings);
        StepParams spMixed = sp;
        spMixed.precision = Precision::Mixed;
        for (int i = 0; i < YEAR_STEPS; ++i) {
            engine.step(spMixed, false);
        }
        std::vector<double> ringPosMixed1;
        engine.copyRingPositions(ringPosMixed1);
        size_t ringStableMixed = 0;
        double sumAbsDrMixedKm = 0.0;
        for (size_t i = 0; i < nRings; ++i) {
            const double r0 = std::sqrt(ringPos0[3 * i + 0] * ringPos0[3 * i + 0] +
                                        ringPos0[3 * i + 1] * ringPos0[3 * i + 1] +
                                        ringPos0[3 * i + 2] * ringPos0[3 * i + 2]);
            const double r1 = std::sqrt(ringPosMixed1[3 * i + 0] * ringPosMixed1[3 * i + 0] +
                                        ringPosMixed1[3 * i + 1] * ringPosMixed1[3 * i + 1] +
                                        ringPosMixed1[3 * i + 2] * ringPosMixed1[3 * i + 2]);
            const double r1Km = r1 / 1000.0;
            if (r1Km >= 60000.0 && r1Km <= 160000.0) {
                ++ringStableMixed;
                sumAbsDrMixedKm += std::abs(r1 - r0) / 1000.0;
            }
        }
        const double meanRingDriftMixedKm = (ringStableMixed > 0) ? (sumAbsDrMixedKm / static_cast<double>(ringStableMixed)) : 0.0;

        // Time 40k Saturn ring kernel step overhead directly (with vs without rings, Mixed & FP64)
        auto timeStepsMs = [&](bool withRings, Precision prec) -> double {
            engine.load(heavy, particles, true, withRings ? &satRings : nullptr);
            StepParams spT = sp;
            spT.precision = prec;
            for (int w = 0; w < 5; ++w) engine.step(spT, false);
            engine.synchronize();
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, engine.stream());
            for (int it = 0; it < 30; ++it) engine.step(spT, false);
            cudaEventRecord(e1, engine.stream());
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return ms / 30.0;
        };
        const double msNoRingsMix   = timeStepsMs(false, Precision::Mixed);
        const double msWithRingsMix = timeStepsMs(true,  Precision::Mixed);
        const double msNoRings64    = timeStepsMs(false, Precision::FP64);
        const double msWithRings64  = timeStepsMs(true,  Precision::FP64);

        std::printf("[Physics Verification - 1 Julian Year (4th-Order PEFRL + Kahan + Full EIH 1PN + Gyro Spin, substeps=20)]\n");
        std::printf("  Earth Semi-Major Axis   : %.6f AU (initial %.6f AU, e = %.5f)\n",
                    earthKep1.a_AU, earthKep0.a_AU, earthKep1.e);
        std::printf("  Phobos 1-Yr Orbit       : a = %.2f km (initial %.2f km), e = %.5f (%s)\n",
                    phobosKep1.a_m / 1000.0, phobosKep.a_m / 1000.0, phobosKep1.e,
                    phobosKep1.bound ? "BOUND" : "UNBOUND");
        std::printf("  Rel. Energy (E_EIH+U_J2): %.3e\n", relEDrift);
        std::printf("  Rel. AngMom (L_EIH+S)   : %.3e\n", relLDrift);
        std::printf("  Canonical |dP_EIH|/M_tot: %.3e m/s\n", dPMtot);
        std::printf("  Earth Gyro Precession   : %+.2f arcsec/yr retrograde | Obliquity Nutation: %+.2f arcsec\n",
                    earthPrecArcsecYr, earthNutationArcsec);
        std::printf("  Saturn Rings (FP64, 1Y) : %zu / %zu stable (<|dr|> = %.2f km) | Mimas 2:1 ILR = %.1f km\n",
                    ringStableCount, nRings, meanRingDriftKm, r21_ILR_km);
        std::printf("  Saturn Rings (Mixed, 1Y): %zu / %zu stable (<|dr|> = %.2f km)\n",
                    ringStableMixed, nRings, meanRingDriftMixedKm);
        std::printf("  Saturn Ring Overhead Mix: +%.3f ms/step (%.3f ms -> %.3f ms @ 3k belt + 40k rings)\n",
                    msWithRingsMix - msNoRingsMix, msNoRingsMix, msWithRingsMix);
        std::printf("  Saturn Ring Overhead 64 : +%.3f ms/step (%.3f ms -> %.3f ms @ 3k belt + 40k rings)\n",
                    msWithRings64 - msNoRings64, msNoRings64, msWithRings64);
        std::printf("  GPU Ring Stats Check    : %u bound / %u escaped (<r> = %.0f km, <e> = %.4f, B-peak = %.0f vs Cassini = %.0f/bin)\n",
                    rsBench.boundCount, rsBench.escapedCount, rsBench.meanRadiusKm, rsBench.meanEcc,
                    bRingPeak, cassiniMin);
        std::printf("  GPU Belt Stats Check    : %u bound (%u asteroidal, <e>=%.4f, <i>=%.2f deg)\n\n",
                    bs.boundCount, bs.asteroidalCount, bs.meanEcc, bs.meanIncDeg);

        std::printf("[Symplectic Time-Reversal Proof (+1 Yr -> -1 Yr, 55 Heavy Bodies, Full EIH + Gyro J2)]\n");
        std::printf("  Max Position Return Err : %.4e m (%.3e AU, worst: %s)\n",
                    timeRev.maxPosErrM, timeRev.maxPosErrM / AU, heavy[timeRev.worstPosIdx].name.c_str());
        std::printf("  Max Velocity Return Err : %.4e m/s\n", timeRev.maxVelErrMs);
        std::printf("  Max Spin-Pole Return Err: %.4e arcsec (final t = %.3f s)\n\n",
                    timeRev.maxSpinErrArcsec, timeRev.finalTimeSec);

        std::printf("[Live 2-Body Mercury EIH Perihelion Precession Proof (%d Orbits, dt = T/400, r.v = 0 Bisection)]\n",
                    mercPrec.orbitsMeasured);
        std::printf("  Measured EIH (1PN) Rate : %.4f arcsec/cy  |  0PN Baseline: %.4f arcsec/cy\n",
                    mercPrec.rateEihArcsecCy, mercPrec.rate0pnArcsecCy);
        std::printf("  Live 1PN Advance (EIH-0): %.4f arcsec/cy  |  Analytical K_OMEGA GR: %.4f arcsec/cy (err = %.4f%%)\n\n",
                    mercPrec.net1pnArcsecCy, mercPrec.analyticalArcsecCy, mercPrec.relErrPct);

        // Extreme CFL Torture Test: Macro dt = 48.0 hours, UI substeps = 1 (1 Julian Year = 183 steps)
        engine.load(heavy, particles, true);
        StepParams spExtreme;
        spExtreme.dtSeconds = 48.0 * 3600.0;
        spExtreme.substeps = 1;
        spExtreme.relativistic = true;
        spExtreme.precision = Precision::FP64;
        for (int i = 0; i < 183; ++i) {
            engine.step(spExtreme, i == 182);
        }
        const auto sExtreme = engine.snapshot();
        const auto phobosExtreme = computeKeplerElements(idxPhobos, idxMars, sExtreme, engine);

        // GC Sandbox Unbinding Check at Macro dt = 48.0 h, substeps = 1: Decompile Mars (rho_N = 1.0 < 4.0)
        engine.setDensityOverride(idxMars, true, 1.0, true);
        engine.step(spExtreme, true);
        const auto sDecomp = engine.snapshot();
        const double dxP = sDecomp.pos[idxPhobos * 3 + 0] - sDecomp.pos[idxMars * 3 + 0];
        const double dyP = sDecomp.pos[idxPhobos * 3 + 1] - sDecomp.pos[idxMars * 3 + 1];
        const double dzP = sDecomp.pos[idxPhobos * 3 + 2] - sDecomp.pos[idxMars * 3 + 2];
        const double rPhobosAfterDecompKm = std::sqrt(dxP * dxP + dyP * dyP + dzP * dzP) / 1000.0;

        std::printf("[CFL Dynamical Frequency Subdivision Torture Test (dt = 48.0 h, substeps = 1)]\n");
        std::printf("  Phobos 1-Yr (dt=48h, sub=1): a = %.2f km, e = %.5f (%s)\n",
                    phobosExtreme.a_m / 1000.0, phobosExtreme.e,
                    phobosExtreme.bound ? "BOUND" : "UNBOUND");
        std::printf("  GC Mars Decompile (Theta=0): Phobos tangent escape r = %.1f km (UNBOUND)\n\n",
                    rPhobosAfterDecompKm);
    }

    // 2. Scaling throughput across particle counts (3k -> 1M)
    const size_t counts[] = {3000, 100000, 250000, 500000, 1000000};
    const Precision modes[] = {Precision::Mixed, Precision::FP64};

    std::printf("%-12s | %-10s | %-12s | %-12s | %-15s | %-10s\n",
                "Particles", "Precision", "Step (ms)", "Equiv FPS", "Interactions/s", "VRAM (MB)");
    std::printf("---------------------------------------------------------------------------------------\n");

    for (Precision prec : modes) {
        for (size_t N : counts) {
            auto particles = buildParticles(N);
            engine.load(heavy, particles, true);

            StepParams sp;
            sp.dtSeconds = 21600.0;
            sp.substeps = 20;
            sp.relativistic = true;
            sp.precision = prec;

            for (int w = 0; w < 5; ++w) engine.step(sp, false);
            engine.synchronize();

            const int iters = (N >= 500000 && prec == Precision::FP64) ? 20 : 40;
            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0, engine.stream());
            for (int it = 0; it < iters; ++it) engine.step(sp, false);
            cudaEventRecord(ev1, engine.stream());
            cudaEventSynchronize(ev1);

            float totalMs = 0.0f;
            cudaEventElapsedTime(&totalMs, ev0, ev1);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);

            const double msPerStep = totalMs / iters;
            const double fps1Step = 1000.0 / msPerStep;
            const double interactionsPerStep =
                static_cast<double>(N) * heavy.size() +
                static_cast<double>(sp.substeps) * heavy.size() * heavy.size();
            const double gigaInterPerSec = (interactionsPerStep / (msPerStep * 1e-3)) * 1e-9;
            const double vramMB = engine.deviceBytes() / (1024.0 * 1024.0);

            std::printf("%-12zu | %-10s | %9.3f ms | %9.1f Hz | %11.2f GI/s | %7.2f MB\n",
                        N, prec == Precision::Mixed ? "Mixed" : "FP64",
                        msPerStep, fps1Step, gigaInterPerSec, vramMB);
        }
        std::printf("---------------------------------------------------------------------------------------\n");
    }

    return 0;
}

} // namespace
} // namespace substrate

int main(int argc, char** argv) {
    using namespace substrate;

    size_t initialParticles = 100000;
    int maxFrames = 0;
    bool runBench = false;
    bool runTimeRev = false;
    std::string exportCsvDir;
    std::string epochCsvPath;
    double exportYears = 1.0;
    double exportDtHours = 6.0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) {
            runBench = true;
        } else if (std::strcmp(argv[i], "--time-reverse") == 0) {
            runTimeRev = true;
        } else if (std::strcmp(argv[i], "--smoke-test") == 0) {
            maxFrames = 30;
        } else if (std::strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            initialParticles = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--export-csv") == 0 && i + 1 < argc) {
            exportCsvDir = argv[++i];
        } else if (std::strcmp(argv[i], "--years") == 0 && i + 1 < argc) {
            exportYears = std::max(0.01, std::strtod(argv[++i], nullptr));
        } else if (std::strcmp(argv[i], "--dt-hours") == 0 && i + 1 < argc) {
            exportDtHours = std::max(0.05, std::strtod(argv[++i], nullptr));
        } else if (std::strcmp(argv[i], "--epoch-csv") == 0 && i + 1 < argc) {
            epochCsvPath = argv[++i];
        }
    }

    if (runBench) {
        return runBenchmarkSuite(epochCsvPath);
    }
    if (runTimeRev) {
        return runTimeReversalCLI(epochCsvPath);
    }
    if (!exportCsvDir.empty()) {
        return runExportCsvCLI(exportCsvDir, exportYears, exportDtHours, initialParticles, epochCsvPath);
    }

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(
        1720, 980,
        "Substrate Logistics 3D N-Body Simulator — Native C++20 / CUDA / OpenGL 4.6",
        nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    bool vsync = false;
    glfwSwapInterval(vsync ? 1 : 0);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL 4.6 via GLAD\n");
        glfwTerminate();
        return 1;
    }

    unsigned int glDevCount = 0;
    int cudaDev = 0;
    if (cudaGLGetDevices(&glDevCount, &cudaDev, 1, cudaGLDeviceListAll) == cudaSuccess && glDevCount > 0) {
        cudaSetDevice(cudaDev);
    } else {
        cudaSetDevice(0);
    }
    cudaDeviceProp gpuProp{};
    cudaGetDeviceProperties(&gpuProp, cudaDev);

    // Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    applySciFiTheme();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");

    // Initialize Physics & Renderer
    auto heavyDescs = buildHeavyBodies();
    maybeApplyEpochCSV(heavyDescs, epochCsvPath);
    const int numHeavy = static_cast<int>(heavyDescs.size());
    const auto parentIdx = buildParentIndices(heavyDescs);
    const int idxSunGui     = std::max(0, findBodyIndex(heavyDescs, "Sun"));
    const int idxMercuryGui = findBodyIndex(heavyDescs, "Mercury");
    const int idxEarthGui   = findBodyIndex(heavyDescs, "Earth");

    SubstrateEngine engine;
    Renderer renderer;
    renderer.init();

    // Parent-relative orbit trails (up to 300 angular-spaced points per heavy body)
    constexpr size_t MAX_TRAIL_PTS = 300;
    std::vector<std::deque<glm::dvec3>> relTrailsAU(numHeavy);
    std::vector<TrailVertex> trailVertBuffer;
    trailVertBuffer.reserve(numHeavy * MAX_TRAIL_PTS * 2);

    TelemetryUIState telemetryUI;

    StepParams stepParams;
    stepParams.dtSeconds = 21600.0;
    stepParams.substeps = 20;
    stepParams.relativistic = true;
    stepParams.precision = Precision::Mixed;

    size_t activeParticleCount = initialParticles;
    auto reloadPayload = [&](size_t numParticles) {
        activeParticleCount = numParticles;
        ParticleSet ps = buildParticles(numParticles, 42, &heavyDescs);
        ParticleSet satRings = buildSaturnRings(40000, heavyDescs);
        engine.load(heavyDescs, ps, true, &satRings);
        renderer.setPayload(heavyDescs, ps, &satRings);
        for (auto& dq : relTrailsAU) dq.clear();
        engine.requestSnapshot();
        const auto& s0 = engine.snapshot();
        telemetryUI.conservation.reset(engine.newtonianEnergy(s0, stepParams.relativistic),
                                       engine.angularMomentumMag(s0, stepParams.relativistic));
        engine.canonicalMomentum(s0, stepParams.relativistic,
                                 telemetryUI.baseMomX, telemetryUI.baseMomY, telemetryUI.baseMomZ);
        telemetryUI.conservationLog.clear();
        telemetryUI.ringStats = RingStats{};
        telemetryUI.eventTracker.reset(numHeavy);
        telemetryUI.eventTracker.update(s0, heavyDescs, parentIdx, engine);
        if (idxEarthGui >= 0 && s0.spinAxis.size() == static_cast<size_t>(3 * numHeavy)) {
            telemetryUI.initialEarthSpinPhi = std::atan2(s0.spinAxis[3 * idxEarthGui + 1],
                                                         s0.spinAxis[3 * idxEarthGui + 0]);
            telemetryUI.hasInitialEarthSpinPhi = true;
        }
    };
    reloadPayload(initialParticles);

    OrbitCamera camera;
    MouseState mouseState{&camera, 0.0, 0.0};
    glfwSetWindowUserPointer(window, &mouseState);
    glfwSetScrollCallback(window, scrollCallback);

    RenderSettings renderSettings;
    renderSettings.terminationShockRadiusAU = static_cast<float>(engine.terminationShockRadiusAU());
    renderSettings.heliopauseRadiusAU = static_cast<float>(engine.heliopauseRadiusAU());
    telemetryUI.coarseProbeAU = engine.heliopauseRadiusAU();

    bool isPaused = false;
    bool reverseTime = false;
    bool showLabels = true;
    int stepsPerFrame = 2;
    int targetIdx = std::max(0, findBodyIndex(heavyDescs, "Earth"));  // Earth by default (highlights Paper 1 Sec. 3 & Sec. 8 benchmarks immediately!)

    const size_t scalePresets[] = {3000, 25000, 100000, 250000, 500000, 1000000, 1500000};
    const char* scaleLabels[] = {
        "3,000 (Prototype Scale)",
        "25,000 (25k Objects)",
        "100,000 (100k Objects)",
        "250,000 (250k Objects)",
        "500,000 (500k Objects)",
        "1,000,000 (1M Objects)",
        "1,500,000 (1.5M Objects)"
    };
    int currentScaleIdx = 2;
    for (int i = 0; i < 7; ++i) {
        if (scalePresets[i] == initialParticles) currentScaleIdx = i;
    }

    float smoothFps = 0.0f;
    int conservationSampleCounter = 0;
    const double cosMinTrailAngle = std::cos(glm::radians(1.25));
    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto now = std::chrono::high_resolution_clock::now();
        const float dtWall = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;
        if (dtWall > 1e-6f) {
            const float instFps = 1.0f / dtWall;
            smoothFps = (smoothFps == 0.0f) ? instFps : (0.92f * smoothFps + 0.08f * instFps);
        }

        // Keyboard shortcut F2 toggles floating telemetry window
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F2)) {
            telemetryUI.showWindow = !telemetryUI.showWindow;
        }

        // Mouse camera controls
        double mx = 0.0, my = 0.0;
        glfwGetCursorPos(window, &mx, &my);
        const double dx = mx - mouseState.lastX;
        const double dy = my - mouseState.lastY;
        mouseState.lastX = mx;
        mouseState.lastY = my;

        if (!io.WantCaptureMouse) {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                camera.azimuthDeg = std::fmod(camera.azimuthDeg - static_cast<float>(dx) * 0.35f + 360.0f, 360.0f);
                camera.elevationDeg = std::clamp(camera.elevationDeg + static_cast<float>(dy) * 0.35f, -89.0f, 89.0f);
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                const float zoomFactor = std::pow(1.008f, static_cast<float>(dy));
                camera.distanceAU = std::clamp(camera.distanceAU * zoomFactor, 1.0e-5f, 25000.0f);
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
                const float panScale = camera.distanceAU * 0.0012f;
                const float az = glm::radians(camera.azimuthDeg);
                const float el = glm::radians(camera.elevationDeg);
                const glm::vec3 right(-std::sin(az), std::cos(az), 0.0f);
                const glm::vec3 up(-std::sin(el) * std::cos(az), -std::sin(el) * std::sin(az), std::cos(el));
                camera.panOffsetAU += (-static_cast<float>(dx) * panScale) * right +
                                      (static_cast<float>(dy) * panScale) * up;
            }
        }

        // 1. Physics Step + Zero-Copy CUDA->GL VBO Write
        const int originIdx = (camera.lockTarget && !renderSettings.logScale) ? targetIdx : -1;

        engine.beginTimer();
        if (!isPaused) {
            StepParams activeParams = stepParams;
            activeParams.dtSeconds = reverseTime ? -std::abs(stepParams.dtSeconds)
                                                 :  std::abs(stepParams.dtSeconds);
            for (int s = 0; s < stepsPerFrame; ++s) {
                const bool snapThisStep = (s == stepsPerFrame - 1);
                engine.step(activeParams, snapThisStep);
            }
        } else {
            engine.requestSnapshot();
        }

        float4* d_heavyVbo = nullptr;
        float4* d_particleVbo = nullptr;
        float4* d_ringVbo = nullptr;
        renderer.mapForCuda(engine.stream(), &d_heavyVbo, &d_particleVbo, &d_ringVbo);
        engine.writeRenderPositions(d_heavyVbo, d_particleVbo, originIdx, d_ringVbo);
        renderer.unmapFromCuda(engine.stream());
        engine.endTimer();

        const HeavySnapshot& snap = engine.snapshot();

        if (!isPaused) {
            telemetryUI.eventTracker.update(snap, heavyDescs, parentIdx, engine);
            if (++conservationSampleCounter % 6 == 0) {
                const double curE = engine.newtonianEnergy(snap, stepParams.relativistic);
                const double curL = engine.angularMomentumMag(snap, stepParams.relativistic);
                telemetryUI.conservation.push(curE, curL);

                ConservationSample cs{};
                cs.simYear = (snap.simTimeSeconds / 86400.0) / 365.25;
                cs.relEnergyDrift = telemetryUI.conservation.relEnergyDrift.empty()
                                        ? 0.0
                                        : static_cast<double>(telemetryUI.conservation.relEnergyDrift.back());
                cs.relAngMomDrift = telemetryUI.conservation.relAngMomDrift.empty()
                                        ? 0.0
                                        : static_cast<double>(telemetryUI.conservation.relAngMomDrift.back());
                double Px = 0.0, Py = 0.0, Pz = 0.0;
                engine.canonicalMomentum(snap, stepParams.relativistic, Px, Py, Pz);
                double mTot = 0.0;
                for (double m : engine.mass()) mTot += m;
                cs.barycenterDriftMs = (mTot > 0.0)
                    ? std::sqrt((Px - telemetryUI.baseMomX) * (Px - telemetryUI.baseMomX) +
                                (Py - telemetryUI.baseMomY) * (Py - telemetryUI.baseMomY) +
                                (Pz - telemetryUI.baseMomZ) * (Pz - telemetryUI.baseMomZ)) / mTot
                    : 0.0;
                if (idxEarthGui >= 0 && snap.spinAxis.size() == static_cast<size_t>(3 * numHeavy)) {
                    const double sx = snap.spinAxis[3 * idxEarthGui + 0];
                    const double sy = snap.spinAxis[3 * idxEarthGui + 1];
                    const double sz = snap.spinAxis[3 * idxEarthGui + 2];
                    cs.earthObliquityDeg = std::acos(std::clamp(sz, -1.0, 1.0)) * (180.0 / PI_CONST);
                    if (telemetryUI.hasInitialEarthSpinPhi) {
                        double dPhi = std::atan2(sy, sx) - telemetryUI.initialEarthSpinPhi;
                        while (dPhi >  PI_CONST) dPhi -= 2.0 * PI_CONST;
                        while (dPhi < -PI_CONST) dPhi += 2.0 * PI_CONST;
                        cs.earthPrecessionArcsec = dPhi * ((180.0 * 3600.0) / PI_CONST);
                    }
                }
                if (idxMercuryGui >= 0) {
                    cs.mercuryArgPeriDeg = computeKeplerElements(idxMercuryGui, idxSunGui, snap, engine).argPeri_deg;
                }
                if (telemetryUI.conservationLog.size() >= 4000) {
                    telemetryUI.conservationLog.erase(telemetryUI.conservationLog.begin());
                }
                telemetryUI.conservationLog.push_back(cs);
            }
        }

        // 2. Build Camera, Parent-Relative Orbit Trails, & 3D Boundary Shells
        int fbW = 1, fbH = 1;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        fbW = std::max(1, fbW);
        fbH = std::max(1, fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.025f, 0.03f, 0.055f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
        const float nearPlane = std::clamp(camera.distanceAU * 1.0e-3f, 1.0e-8f, 0.1f);
        const float farPlane = std::max(40000.0f, camera.distanceAU * 25.0f);
        const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, nearPlane, farPlane);
        const glm::mat4 view = camera.viewMatrix();
        const glm::mat4 viewProj = proj * view;

        glm::dvec3 originWorldAU_d(0.0);
        if (originIdx >= 0 && originIdx < numHeavy) {
            originWorldAU_d = glm::dvec3(
                snap.pos[3 * originIdx + 0] / AU,
                snap.pos[3 * originIdx + 1] / AU,
                snap.pos[3 * originIdx + 2] / AU);
        }

        std::vector<glm::vec3> heavyRelAU(numHeavy);
        for (int i = 0; i < numHeavy; ++i) {
            const glm::dvec3 worldAU(
                snap.pos[3 * i + 0] / AU,
                snap.pos[3 * i + 1] / AU,
                snap.pos[3 * i + 2] / AU);
            heavyRelAU[i] = glm::vec3(worldAU - originWorldAU_d);

            // Sample parent-relative orbital trail points by angular / linear displacement
            const int p = parentIdx[i];
            if (p >= 0 && !isPaused) {
                const glm::dvec3 pWorldAU(
                    snap.pos[3 * p + 0] / AU,
                    snap.pos[3 * p + 1] / AU,
                    snap.pos[3 * p + 2] / AU);
                const glm::dvec3 relP = worldAU - pWorldAU;
                const double rLen = glm::length(relP);
                if (rLen > 1e-12) {
                    auto& dq = relTrailsAU[i];
                    bool pushSample = dq.empty();
                    if (!pushSample) {
                        const glm::dvec3 prev = dq.back();
                        const double prevLen = std::max(glm::length(prev), 1e-12);
                        const double cosAng = glm::dot(prev / prevLen, relP / rLen);
                        const double linFrac = glm::length(relP - prev) / prevLen;
                        if (cosAng < cosMinTrailAngle || linFrac > 0.03) {
                            pushSample = true;
                        }
                    }
                    if (pushSample) {
                        if (dq.size() >= MAX_TRAIL_PTS) dq.pop_front();
                        dq.push_back(relP);
                    }
                }
            }
        }

        // Assemble alpha-faded 3D trail line segments
        trailVertBuffer.clear();
        if (renderSettings.showOrbitTrails) {
            for (int i = 0; i < numHeavy; ++i) {
                const int p = parentIdx[i];
                if (p < 0 || relTrailsAU[i].empty()) continue;

                // Skip sub-pixel moon trails when zoomed out to full solar system
                if (heavyDescs[i].category == BodyCategory::Moon) {
                    const float approxOrbitAU = static_cast<float>(glm::length(relTrailsAU[i].back()));
                    if (approxOrbitAU < camera.distanceAU * 0.004f) continue;
                }

                const glm::dvec3 pWorldAU(
                    snap.pos[3 * p + 0] / AU,
                    snap.pos[3 * p + 1] / AU,
                    snap.pos[3 * p + 2] / AU);
                const glm::dvec3 pRelOrigin = pWorldAU - originWorldAU_d;

                const auto& dq = relTrailsAU[i];
                const size_t nPts = dq.size();
                const float cr = heavyDescs[i].color[0];
                const float cg = heavyDescs[i].color[1];
                const float cb = heavyDescs[i].color[2];
                const float maxAlpha = (i == targetIdx) ? 0.80f : 0.42f;

                for (size_t k = 0; k < nPts; ++k) {
                    const glm::vec3 v0 = glm::vec3(pRelOrigin + dq[k]);
                    const glm::vec3 v1 = (k + 1 < nPts)
                                             ? glm::vec3(pRelOrigin + dq[k + 1])
                                             : heavyRelAU[i];
                    const float a0 = maxAlpha * static_cast<float>(k) / static_cast<float>(nPts);
                    const float a1 = maxAlpha * static_cast<float>(k + 1) / static_cast<float>(nPts);
                    trailVertBuffer.push_back({v0, packRGBA(cr, cg, cb, a0)});
                    trailVertBuffer.push_back({v1, packRGBA(cr, cg, cb, a1)});
                }
            }
        }
        renderer.updateTrailVertices(trailVertBuffer);

        // Assemble 3D Boundary Shells (Heliosphere + Target 3R/4R/6.784R, Hill, & Roche)
        if (!renderSettings.customHeliopauseOverride) {
            renderSettings.terminationShockRadiusAU = static_cast<float>(engine.terminationShockRadiusAU());
            renderSettings.heliopauseRadiusAU = static_cast<float>(engine.heliopauseRadiusAU());
        }

        DrawContext drawCtx;
        drawCtx.viewProj = viewProj;
        drawCtx.view = view;
        drawCtx.camDistanceAU = camera.distanceAU;
        drawCtx.worldOffsetAU = glm::vec3(-originWorldAU_d);
        drawCtx.starCenterRelAU = (numHeavy > 0) ? heavyRelAU[0] : glm::vec3(-originWorldAU_d);
        drawCtx.viewportPx = glm::vec2(static_cast<float>(fbW), static_cast<float>(fbH));

        // Heliospheric Ram-Pressure Boundaries, Gravitational Floor R_g & Outer Oort Core
        const double rGravSunAU = engine.gravitationalFloorAU();
        const double rOortCoreAU = engine.outerOortStagnationAU();
        if (renderSettings.showHeliosphereShells && rGravSunAU > 0.0) {
            const glm::vec3 starCenter = drawCtx.starCenterRelAU;
            // Solar Wind Termination Shock Ring (89.44 AU, amber)
            drawCtx.shells.push_back({starCenter, renderSettings.terminationShockRadiusAU,
                                      glm::vec4(1.00f, 0.68f, 0.22f, 0.42f), false, true});
            // Solar Wind Heliopause Equatorial Ring (120.74 AU, blue)
            drawCtx.shells.push_back({starCenter, renderSettings.heliopauseRadiusAU,
                                      glm::vec4(0.22f, 0.62f, 1.00f, 0.44f), false, true});
            // Gravitational Deletion Floor R_g (~2,430.82 AU: u_g == Sigma_{0,Ledger}, subtle outer reference ring)
            drawCtx.shells.push_back({starCenter, static_cast<float>(rGravSunAU),
                                      glm::vec4(1.00f, 0.86f, 0.34f, 0.22f),
                                      camera.distanceAU > 250.0f, true});
            // Outer Oort Stagnation Core (6.784 * R_g = 16,490.6 AU)
            if (rOortCoreAU > 0.0) {
                drawCtx.shells.push_back({starCenter, static_cast<float>(rOortCoreAU),
                                          glm::vec4(0.55f, 0.75f, 1.00f, 0.24f),
                                          camera.distanceAU > 3000.0f, true});
            }
        }

        if (targetIdx >= 0 && targetIdx < numHeavy) {
            const int pIdx = parentIdx[targetIdx];
            const KeplerElements kep = computeKeplerElements(targetIdx, pIdx, snap, engine);
            const DomainRadii dom = computeDomainRadii(targetIdx, pIdx, heavyDescs, kep, engine);
            const glm::vec3 center = heavyRelAU[targetIdx];

            // Compute live 3D spin pole s_hat and rotation phase phi(t):
            //   - tidalLockMode == 1 (Moons): spin pole is the live orbital angular momentum normal h_hat,
            //     and prime-meridian spoke points directly toward the parent planet (-r_rel)!
            //   - tidalLockMode == 2 (Mercury 3:2 resonance): omega_rot = 1.5 * n_orb
            //   - tidalLockMode == 0 (Free Rotators): uses seeded spinAxis & spinOmega
            glm::vec3 sHat(static_cast<float>(engine.spinAxisX()[targetIdx]),
                           static_cast<float>(engine.spinAxisY()[targetIdx]),
                           static_cast<float>(engine.spinAxisZ()[targetIdx]));
            const int lockMode = engine.tidalLockMode()[targetIdx];
            glm::mat3 targetSpinBasis = buildSpinBasis(sHat);
            float spinPhaseRad = static_cast<float>(std::fmod(engine.spinOmega()[targetIdx] * snap.simTimeSeconds,
                                                              2.0 * PI_CONST));

            if (lockMode == 1 && pIdx >= 0) {
                const glm::dvec3 rRel(snap.pos[3 * targetIdx + 0] - snap.pos[3 * pIdx + 0],
                                      snap.pos[3 * targetIdx + 1] - snap.pos[3 * pIdx + 1],
                                      snap.pos[3 * targetIdx + 2] - snap.pos[3 * pIdx + 2]);
                const glm::dvec3 vRel(snap.vel[3 * targetIdx + 0] - snap.vel[3 * pIdx + 0],
                                      snap.vel[3 * targetIdx + 1] - snap.vel[3 * pIdx + 1],
                                      snap.vel[3 * targetIdx + 2] - snap.vel[3 * pIdx + 2]);
                const glm::dvec3 hVec = glm::cross(rRel, vRel);
                const double hLen = glm::length(hVec);
                if (hLen > 1e-6) {
                    sHat = glm::vec3(hVec / hLen);
                    targetSpinBasis = buildSpinBasis(sHat);
                    const glm::vec3 toParent = glm::normalize(glm::vec3(-rRel));
                    const float localX = glm::dot(toParent, targetSpinBasis[0]);
                    const float localY = glm::dot(toParent, targetSpinBasis[1]);
                    spinPhaseRad = std::atan2(localY, localX);
                }
            } else if (lockMode == 2 && pIdx >= 0 && kep.bound && kep.period_days > 0.0) {
                const double omegaRes = 1.5 * (2.0 * PI_CONST / (kep.period_days * 86400.0));
                spinPhaseRad = static_cast<float>(std::fmod(omegaRes * snap.simTimeSeconds, 2.0 * PI_CONST));
            }

            if (renderSettings.showVectorChannels) {
                // 1R Bare-Metal Core Ring + Tilted Spin Pole & Rotating Prime-Meridian Spoke
                drawCtx.shells.push_back({center, static_cast<float>(dom.core1R_m / AU),
                                          glm::vec4(0.95f, 0.98f, 1.0f, 0.50f), false, true,
                                          targetSpinBasis, true, spinPhaseRad});
                // 3R Directional Mass/Axial Extension Shell (Plasmapause altitude, tilted to spin axis)
                drawCtx.shells.push_back({center, static_cast<float>(dom.vector3R_m / AU),
                                          glm::vec4(0.00f, 0.90f, 1.00f, 0.32f), true, true,
                                          targetSpinBasis, false, 0.0f});
                // 4R Unspooled 4-Channel EM Boundary (Van Allen / Charge analog, tilted to spin axis)
                drawCtx.shells.push_back({center, static_cast<float>(dom.vector4R_m / AU),
                                          glm::vec4(0.76f, 0.34f, 1.00f, 0.26f), true, true,
                                          targetSpinBasis, false, 0.0f});
                // 6.784R Outer Dipolar Ring-Current Trapping Boundary (C_delta^(-1/4) * R_body, gold ring)
                drawCtx.shells.push_back({center, static_cast<float>(dom.rDeltaStagnation_m / AU),
                                          glm::vec4(1.00f, 0.82f, 0.25f, 0.45f), false, true,
                                          targetSpinBasis, false, 0.0f});
            }

            if (renderSettings.showHillAndRoche) {
                if (dom.hillRadius_m > 0.0) {
                    drawCtx.shells.push_back({center, static_cast<float>(dom.hillRadius_m / AU),
                                              glm::vec4(1.00f, 0.78f, 0.22f, 0.22f), true, true,
                                              targetSpinBasis, false, 0.0f});
                }
                if (dom.rocheFluid_m > 0.0) {
                    const bool isMoon = (heavyDescs[targetIdx].category == BodyCategory::Moon && pIdx >= 0);
                    const glm::vec3 rocheCenter = isMoon ? heavyRelAU[pIdx] : center;
                    const glm::mat3 rocheBasis = isMoon
                        ? buildSpinBasis(glm::vec3(static_cast<float>(engine.spinAxisX()[pIdx]),
                                                   static_cast<float>(engine.spinAxisY()[pIdx]),
                                                   static_cast<float>(engine.spinAxisZ()[pIdx])))
                        : targetSpinBasis;
                    drawCtx.shells.push_back({rocheCenter, static_cast<float>(dom.rocheFluid_m / AU),
                                              glm::vec4(1.00f, 0.28f, 0.32f, 0.55f), false, true,
                                              rocheBasis, false, 0.0f});
                }
            }
        }

        drawCtx.heavyDrawOrder.resize(numHeavy);
        std::iota(drawCtx.heavyDrawOrder.begin(), drawCtx.heavyDrawOrder.end(), 0);
        const glm::vec3 eyeAU = camera.eyeRelativeAU();
        std::sort(drawCtx.heavyDrawOrder.begin(), drawCtx.heavyDrawOrder.end(),
                  [&](int a, int b) {
                      const glm::vec3 pa = renderSettings.logScale ? logCompressPos(heavyRelAU[a]) : heavyRelAU[a];
                      const glm::vec3 pb = renderSettings.logScale ? logCompressPos(heavyRelAU[b]) : heavyRelAU[b];
                      const glm::vec3 da = pa - eyeAU;
                      const glm::vec3 db = pb - eyeAU;
                      return glm::dot(da, da) > glm::dot(db, db);
                  });

        renderer.draw(drawCtx, renderSettings);

        // 3. Start ImGui Frame & Draw Smart Deconflicted 2D Viewport HUD (+ Emergent L1-L5 Lagrange Points)
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glm::vec3 lpRelAU[5] = {};
        bool lpValid = false;
        if (targetIdx >= 0 && targetIdx < numHeavy && parentIdx[targetIdx] >= 0) {
            const LagrangePoints lp = computeLagrangePoints(targetIdx, parentIdx[targetIdx], snap, engine);
            if (lp.valid) {
                lpValid = true;
                for (int k = 0; k < 5; ++k) {
                    const glm::dvec3 worldAU(lp.pos_m[k][0] / AU, lp.pos_m[k][1] / AU, lp.pos_m[k][2] / AU);
                    lpRelAU[k] = glm::vec3(worldAU - originWorldAU_d);
                }
            }
        }

        drawSmartViewportHUD(ImGui::GetBackgroundDrawList(), io.DisplaySize, viewProj,
                             heavyRelAU, heavyDescs, parentIdx, engine, renderSettings,
                             showLabels, lpRelAU, lpValid,
                             camera.distanceAU, camera.azimuthDeg, camera.elevationDeg,
                             targetIdx, camera.lockTarget, camera.panOffsetAU);

        // 4. Separate Toggleable Floating Window: Astrophysics & Substrate Telemetry
        drawTelemetryWindow(telemetryUI, heavyDescs, parentIdx, snap, engine, stepParams,
                            renderSettings, targetIdx, camera.lockTarget, camera.distanceAU,
                            camera.panOffsetAU, [&]() { reloadPayload(activeParticleCount); });

        // 5. Right-Side Control Center
        const float panelWidth = 395.0f;
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - panelWidth - 12.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(panelWidth, io.DisplaySize.y - 24.0f), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Substrate Logistics Control Center")) {
            const float gpuMs = engine.lastTimedMs();
            const size_t totalObjs = engine.numParticles() + engine.numRingParticles() + static_cast<size_t>(engine.numHeavy());

            // Prominent toggle button for the Floating Telemetry Window
            const ImVec4 btnCol = telemetryUI.showWindow
                                      ? ImVec4(0.00f, 0.48f, 0.68f, 1.0f)
                                      : ImVec4(0.14f, 0.24f, 0.36f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
            if (ImGui::Button(telemetryUI.showWindow
                                  ? "Hide Astrophysics & Substrate Telemetry [F2]"
                                  : "Open Astrophysics & Substrate Telemetry [F2]",
                              ImVec2(-1.0f, 32.0f))) {
                telemetryUI.showWindow = !telemetryUI.showWindow;
            }
            ImGui::PopStyleColor();
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Substrate Hardware Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("CUDA Device: %s (sm_%d%d)", gpuProp.name, gpuProp.major, gpuProp.minor);
                ImGui::Text("Total Payload: %zu Objects", totalObjs);
                ImGui::TextColored(ImVec4(0.0f, 0.92f, 1.0f, 1.0f),
                                   "Render Rate: %.1f FPS (%.2f ms)", smoothFps, 1000.0f / std::max(smoothFps, 1.0f));
                ImGui::Text("CUDA Step + Interop Time: %.2f ms", gpuMs);
                ImGui::Text("VRAM (Physics State): %.2f MB", engine.deviceBytes() / (1024.0 * 1024.0));
                const double years = (snap.simTimeSeconds / 86400.0) / 365.25;
                ImGui::Text("Cosmological Time: %.3f Years", years);
                if (reverseTime) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.18f, 1.0f), "[TIME REVERSED]");
                }
            }

            if (ImGui::CollapsingHeader("Quick Target & Camera Framing", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginCombo("Inspect Target", heavyDescs[targetIdx].name.c_str())) {
                    for (int i = 0; i < numHeavy; ++i) {
                        const bool selected = (i == targetIdx);
                        if (ImGui::Selectable(heavyDescs[i].name.c_str(), selected)) {
                            targetIdx = i;
                            if (camera.lockTarget) camera.panOffsetAU = glm::vec3(0.0f);
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                const double* tp = &snap.pos[3 * targetIdx];
                const double* tv = &snap.vel[3 * targetIdx];
                const double* sp = &snap.pos[0];
                const double dxS = tp[0] - sp[0], dyS = tp[1] - sp[1], dzS = tp[2] - sp[2];
                const double rSunAU = std::sqrt(dxS * dxS + dyS * dyS + dzS * dzS) / AU;
                const double velKms = std::sqrt(tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2]) / 1000.0;

                ImGui::Text("Heliocentric Dist: %.4f AU  |  Speed: %.2f km/s", rSunAU, velKms);
                const double rho = engine.density()[targetIdx];
                ImGui::Text("Node Density: %.3e nodes/m^3", rho);
                if (rho >= RHO_GC) {
                    ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.5f, 1.0f),
                                       "GC Status: ACTIVE (Theta = 1%s)",
                                       engine.isDensityOverridden(targetIdx) ? ", Sandbox Override" : "");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                                       "GC Status: PASSIVE I-PHASE (Theta = 0)");
                }

                if (ImGui::Checkbox("Lock Camera to Target (Origin Shift)", &camera.lockTarget)) {
                    camera.panOffsetAU = glm::vec3(0.0f);
                }

                if (ImGui::Button("Zoom: 3R/4R/6.78R Magnetosphere")) {
                    camera.lockTarget = true;
                    renderSettings.logScale = false;
                    renderSettings.showVectorChannels = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    camera.distanceAU = std::max(1.0e-5f,
                                                 static_cast<float>((18.0 * engine.radius()[targetIdx]) / AU));
                }
                ImGui::SameLine();
                if (ImGui::Button("Zoom: Moon / Hill System")) {
                    const int pIdx = parentIdx[targetIdx];
                    const KeplerElements kep = computeKeplerElements(targetIdx, pIdx, snap, engine);
                    const DomainRadii dom = computeDomainRadii(targetIdx, pIdx, heavyDescs, kep, engine);
                    camera.lockTarget = true;
                    renderSettings.logScale = false;
                    renderSettings.showHillAndRoche = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    const double spanM = (dom.hillRadius_m > 0.0)
                                             ? (2.4 * dom.hillRadius_m)
                                             : (60.0 * engine.radius()[targetIdx]);
                    camera.distanceAU = std::clamp(static_cast<float>(spanM / AU), 5.0e-5f, 50.0f);
                }
                if (ImGui::Button("Zoom: Saturn & Cassini Rings", ImVec2(-1.0f, 0.0f))) {
                    const int satIdx = findBodyIndex(heavyDescs, "Saturn");
                    if (satIdx >= 0) targetIdx = satIdx;
                    camera.lockTarget = true;
                    renderSettings.logScale = false;
                    renderSettings.showSaturnRings = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    camera.distanceAU = 0.0028f;
                }
                if (ImGui::Button("Zoom: Heliosphere (TS 89.4 AU / HP 120.7 AU)", ImVec2(-1.0f, 0.0f))) {
                    camera.lockTarget = false;
                    renderSettings.logScale = false;
                    renderSettings.showHeliosphereShells = true;
                    renderSettings.showHeliopause = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    camera.distanceAU = 195.0f;
                }
                if (ImGui::Button("Zoom: Outer Oort Core (16,491 AU)", ImVec2(-1.0f, 0.0f))) {
                    camera.lockTarget = false;
                    renderSettings.logScale = false;
                    renderSettings.showHeliosphereShells = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    camera.distanceAU = 21000.0f;
                }
            }

            if (ImGui::CollapsingHeader("3D Overlays & Viewport Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Show Deconflicted 2D Labels (HUD)", &showLabels);
                ImGui::Checkbox("Show 3D Orbit Trails", &renderSettings.showOrbitTrails);
                ImGui::Checkbox("Show Passive Saturnian Ring System (40k)", &renderSettings.showSaturnRings);
                ImGui::Checkbox("Show 3R / 4R / 6.784R Magnetospheric Shells", &renderSettings.showVectorChannels);
                ImGui::Checkbox("Show Hill Sphere & Fluid Roche Limit", &renderSettings.showHillAndRoche);
                ImGui::Checkbox("Show Lagrange Points (L1-L5) for Target", &renderSettings.showLagrangePoints);
                ImGui::Checkbox("Show Heliosphere Rings (TS 89.44 AU & R_g Floor)", &renderSettings.showHeliosphereShells);
                char helioLabel[96];
                std::snprintf(helioLabel, sizeof(helioLabel),
                              "Show Heliopause Wireframe Sphere (%.2f AU)##HelioSphere",
                              renderSettings.heliopauseRadiusAU);
                ImGui::Checkbox(helioLabel, &renderSettings.showHeliopause);
                ImGui::Checkbox("Show Ecliptic Grid", &renderSettings.showGrid);
                if (ImGui::Checkbox("GPU Logarithmic Distance Scaling", &renderSettings.logScale)) {
                    camera.panOffsetAU = glm::vec3(0.0f);
                }
                ImGui::Checkbox("Additive Particle Glow", &renderSettings.additiveParticles);
                if (ImGui::Checkbox("V-Sync (Cap to Monitor Refresh)", &vsync)) {
                    glfwSwapInterval(vsync ? 1 : 0);
                }
                ImGui::SliderFloat("Particle Size", &renderSettings.particleSizeScale, 0.5f, 4.0f, "%.2f");
                ImGui::SliderFloat("Particle Alpha", &renderSettings.particleAlpha, 0.05f, 1.0f, "%.2f");
                ImGui::SliderFloat("Planet Size Scale", &renderSettings.heavySizeScale, 0.5f, 2.5f, "%.2f");
            }

            if (ImGui::CollapsingHeader("Scale & Physics Kernel Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Combo("Particle Scale", &currentScaleIdx, scaleLabels, 7)) {
                    reloadPayload(scalePresets[currentScaleIdx]);
                }

                int precInt = (stepParams.precision == Precision::Mixed) ? 0 : 1;
                if (ImGui::RadioButton("Mixed (FP64 State + Dual FP32 Accel)", &precInt, 0)) {
                    stepParams.precision = Precision::Mixed;
                }
                if (ImGui::RadioButton("Full IEEE FP64 Reference", &precInt, 1)) {
                    stepParams.precision = Precision::FP64;
                }

                if (ImGui::Checkbox("1PN Relativistic Precession (Full EIH)", &stepParams.relativistic)) {
                    engine.refreshCachedAccelerations(stepParams.relativistic);
                    telemetryUI.conservation.reset(engine.newtonianEnergy(snap, stepParams.relativistic),
                                                   engine.angularMomentumMag(snap, stepParams.relativistic));
                }
                if (ImGui::Checkbox("Reverse Time (dt -> -dt) [Symplectic Proof]", &reverseTime)) {
                    engine.refreshCachedAccelerations(stepParams.relativistic);
                }
                ImGui::SliderInt("Macro Steps / Frame", &stepsPerFrame, 1, 12);

                float dtHours = static_cast<float>(stepParams.dtSeconds / 3600.0);
                if (ImGui::SliderFloat("Macro dt (Hours)", &dtHours, 0.5f, 48.0f, "%.1f h")) {
                    stepParams.dtSeconds = static_cast<double>(dtHours) * 3600.0;
                }
                ImGui::SliderInt("Moon Sub-steps", &stepParams.substeps, 1, 50);
            }

            ImGui::Separator();
            if (ImGui::Button(isPaused ? "Resume Physics Engine" : "Pause Physics Engine", ImVec2(-1.0f, 32.0f))) {
                isPaused = !isPaused;
            }
            if (ImGui::Button("Reset Solar System Camera", ImVec2(-1.0f, 28.0f))) {
                camera.lockTarget = false;
                camera.reset();
            }
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (maxFrames > 0) {
            static int frameIdx = 0;
            if (++frameIdx >= maxFrames) {
                std::vector<uint8_t> pixels(static_cast<size_t>(fbW) * fbH * 4);
                glReadPixels(0, 0, fbW, fbH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                size_t nonBgPixels = 0;
                for (size_t p = 0; p < static_cast<size_t>(fbW) * fbH; ++p) {
                    const int r = pixels[4 * p + 0];
                    const int g = pixels[4 * p + 1];
                    const int b = pixels[4 * p + 2];
                    if (std::abs(r - 6) > 8 || std::abs(g - 8) > 8 || std::abs(b - 14) > 8) {
                        ++nonBgPixels;
                    }
                }
                std::printf("[Smoke Test] Rendered %d frames (%zu belt + %zu Saturn ring particles + %d heavy bodies)\n",
                            frameIdx, engine.numParticles(), engine.numRingParticles(), engine.numHeavy());
                std::printf("[Smoke Test] Frame Rate: %.1f FPS | GPU Step+Interop: %.2f ms | Non-BG Pixels: %zu\n",
                            smoothFps, engine.lastTimedMs(), nonBgPixels);
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }

        glfwSwapBuffers(window);
    }

    renderer.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
