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
    ms->cam->distanceAU = std::clamp(ms->cam->distanceAU * factor, 1.0e-5f, 3500.0f);
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
// Headless Benchmark & Manuscript Verification Suite
// ---------------------------------------------------------------------------
int runBenchmarkSuite() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("====================================================================\n");
    std::printf("SUBSTRATE LOGISTICS NATIVE C++/CUDA ENGINE - BENCHMARK & AUDIT\n");
    std::printf("GPU: %s (Compute %d.%d, %.0f MB VRAM)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("====================================================================\n\n");

    const auto heavy = buildHeavyBodies();
    SubstrateEngine engine;

    // 1. Theoretical Manuscript Audit + 1-Year Symplectic Invariance
    {
        auto particles = buildParticles(3000);
        engine.load(heavy, particles, true);
        engine.requestSnapshot();
        const auto s0 = engine.snapshot();
        const double E0 = engine.newtonianEnergy(s0);
        const double L0 = engine.angularMomentumMag(s0);

        // Earth = 3, Mercury = 1, Sun = 0, Jupiter = 8, Saturn = 13
        const auto earthKep0  = computeKeplerElements(3, 0, s0, engine);
        const auto earthDom0  = computeDomainRadii(3, 0, heavy, earthKep0, engine);
        const auto earthAud   = computeSubstrateAudit(3, 0, s0, earthKep0, engine, true);
        const auto mercKep0   = computeKeplerElements(1, 0, s0, engine);
        const auto mercAud    = computeSubstrateAudit(1, 0, s0, mercKep0, engine, true);
        const auto jupKep0    = computeKeplerElements(8, 0, s0, engine);
        const auto jupAud     = computeSubstrateAudit(8, 0, s0, jupKep0, engine, true);
        const auto satKep0    = computeKeplerElements(13, 0, s0, engine);
        const auto satAud     = computeSubstrateAudit(13, 0, s0, satKep0, engine, true);
        // Phobos = 6, Mars = 5
        const auto phobosKep  = computeKeplerElements(6, 5, s0, engine);
        const auto phobosDom  = computeDomainRadii(6, 5, heavy, phobosKep, engine);
        const auto phobosAud  = computeSubstrateAudit(6, 5, s0, phobosKep, engine, true);

        const double rGravAU     = engine.gravitationalFloorAU();
        const double rOortCoreAU = engine.outerOortStagnationAU();
        const double rTS_AU      = engine.terminationShockRadiusAU();
        const double rHP_AU      = engine.heliopauseRadiusAU();

        std::printf("[Substrate Logistics Bare-Metal Hardware Audit]\n");
        std::printf("  Coupled Heavy Bodies              : %zu (Sun, 8 Planets, 21 Moons, 13 Dwarf/Trojan/Comet)\n",
                    heavy.size());
        std::printf("  Exact C_delta                     : %.12e  ((sqrt(5) - 2) / 500)\n", C_DELTA);
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
        std::printf("  Jupiter / Saturn Derived J2       : J2_Jup = %.4e (Obs: 1.470e-2) | J2_Sat = %.4e (Obs: 1.629e-2)\n",
                    jupAud.derivedJ2, satAud.derivedJ2);
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
        std::printf("  Galactic 2D Disk Truncation Ratio : %.3f R_d       (0.5*ln(1/C_d); Obs: 3.8 +- 0.7 R_d)\n\n",
                    DISK_TRUNCATION_RATIO);

        StepParams sp;
        sp.dtSeconds = 21600.0;
        sp.substeps = 20;
        sp.relativistic = true;
        sp.precision = Precision::FP64;

        for (int i = 0; i < 365 * 4 + 1; ++i) {  // 1 Julian year (365.25 days)
            engine.step(sp, i == 365 * 4);
        }
        const auto s1 = engine.snapshot();
        const double E1 = engine.newtonianEnergy(s1);
        const double L1 = engine.angularMomentumMag(s1);
        const double relEDrift = std::abs((E1 - E0) / E0);
        const double relLDrift = std::abs((L1 - L0) / L0);
        const auto earthKep1  = computeKeplerElements(3, 0, s1, engine);
        const auto phobosKep1 = computeKeplerElements(6, 5, s1, engine);

        // Verify GPU BeltStats kernel
        BeltStats bs{};
        engine.computeBeltStats(bs, 5.2044);

        std::printf("[Physics Verification - 1 Julian Year (4th-Order PEFRL + Kahan, substeps=20)]\n");
        std::printf("  Earth Semi-Major Axis  : %.6f AU (initial %.6f AU, e = %.5f)\n",
                    earthKep1.a_AU, earthKep0.a_AU, earthKep1.e);
        std::printf("  Phobos 1-Yr Orbit      : a = %.2f km (initial %.2f km), e = %.5f (%s)\n",
                    phobosKep1.a_m / 1000.0, phobosKep.a_m / 1000.0, phobosKep1.e,
                    phobosKep1.bound ? "BOUND" : "UNBOUND");
        std::printf("  Relative Energy Drift  : %.3e\n", relEDrift);
        std::printf("  Relative AngMom Drift  : %.3e\n", relLDrift);
        std::printf("  GPU Belt Stats Check   : %u bound (%u asteroidal, <e>=%.4f, <i>=%.2f deg)\n\n",
                    bs.boundCount, bs.asteroidalCount, bs.meanEcc, bs.meanIncDeg);

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
        const auto phobosExtreme = computeKeplerElements(6, 5, sExtreme, engine);

        // GC Sandbox Unbinding Check at Macro dt = 48.0 h, substeps = 1: Decompile Mars (rho_N = 1.0 < 4.0)
        engine.setDensityOverride(5, true, 1.0, true);
        engine.step(spExtreme, true);
        const auto sDecomp = engine.snapshot();
        const double dxP = sDecomp.pos[6 * 3 + 0] - sDecomp.pos[5 * 3 + 0];
        const double dyP = sDecomp.pos[6 * 3 + 1] - sDecomp.pos[5 * 3 + 1];
        const double dzP = sDecomp.pos[6 * 3 + 2] - sDecomp.pos[5 * 3 + 2];
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
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) {
            return runBenchmarkSuite();
        }
        if (std::strcmp(argv[i], "--smoke-test") == 0) {
            maxFrames = 30;
        }
        if (std::strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            initialParticles = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        }
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
    const auto heavyDescs = buildHeavyBodies();
    const int numHeavy = static_cast<int>(heavyDescs.size());

    std::vector<int> parentIdx(numHeavy, -1);
    for (int i = 0; i < numHeavy; ++i) {
        for (int j = 0; j < numHeavy; ++j) {
            if (heavyDescs[i].parent == heavyDescs[j].name) {
                parentIdx[i] = j;
                break;
            }
        }
    }

    SubstrateEngine engine;
    Renderer renderer;
    renderer.init();

    // Parent-relative orbit trails (up to 300 angular-spaced points per heavy body)
    constexpr size_t MAX_TRAIL_PTS = 300;
    std::vector<std::deque<glm::dvec3>> relTrailsAU(numHeavy);
    std::vector<TrailVertex> trailVertBuffer;
    trailVertBuffer.reserve(numHeavy * MAX_TRAIL_PTS * 2);

    TelemetryUIState telemetryUI;

    size_t activeParticleCount = initialParticles;
    auto reloadPayload = [&](size_t numParticles) {
        activeParticleCount = numParticles;
        ParticleSet ps = buildParticles(numParticles);
        engine.load(heavyDescs, ps, true);
        renderer.setPayload(heavyDescs, ps);
        for (auto& dq : relTrailsAU) dq.clear();
        engine.requestSnapshot();
        const auto& s0 = engine.snapshot();
        telemetryUI.conservation.reset(engine.newtonianEnergy(s0), engine.angularMomentumMag(s0));
    };
    reloadPayload(initialParticles);

    OrbitCamera camera;
    MouseState mouseState{&camera, 0.0, 0.0};
    glfwSetWindowUserPointer(window, &mouseState);
    glfwSetScrollCallback(window, scrollCallback);

    StepParams stepParams;
    stepParams.dtSeconds = 21600.0;
    stepParams.substeps = 20;
    stepParams.relativistic = true;
    stepParams.precision = Precision::Mixed;

    RenderSettings renderSettings;
    renderSettings.terminationShockRadiusAU = static_cast<float>(engine.terminationShockRadiusAU());
    renderSettings.heliopauseRadiusAU = static_cast<float>(engine.heliopauseRadiusAU());
    telemetryUI.coarseProbeAU = engine.heliopauseRadiusAU();

    bool isPaused = false;
    bool showLabels = true;
    int stepsPerFrame = 2;
    int targetIdx = 3;  // Earth by default (highlights Paper 1 Sec. 3 & Sec. 8 benchmarks immediately!)

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
                camera.distanceAU = std::clamp(camera.distanceAU * zoomFactor, 1.0e-5f, 3500.0f);
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
            for (int s = 0; s < stepsPerFrame; ++s) {
                const bool snapThisStep = (s == stepsPerFrame - 1);
                engine.step(stepParams, snapThisStep);
            }
        } else {
            engine.requestSnapshot();
        }

        float4* d_heavyVbo = nullptr;
        float4* d_particleVbo = nullptr;
        renderer.mapForCuda(engine.stream(), &d_heavyVbo, &d_particleVbo);
        engine.writeRenderPositions(d_heavyVbo, d_particleVbo, originIdx);
        renderer.unmapFromCuda(engine.stream());
        engine.endTimer();

        const HeavySnapshot& snap = engine.snapshot();

        if (!isPaused && (++conservationSampleCounter % 6 == 0)) {
            telemetryUI.conservation.push(engine.newtonianEnergy(snap),
                                          engine.angularMomentumMag(snap));
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
        const float farPlane = std::max(5000.0f, camera.distanceAU * 25.0f);
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
        drawCtx.worldOffsetAU = glm::vec3(-originWorldAU_d);
        drawCtx.starCenterRelAU = (numHeavy > 0) ? heavyRelAU[0] : glm::vec3(-originWorldAU_d);
        drawCtx.viewportPx = glm::vec2(static_cast<float>(fbW), static_cast<float>(fbH));

        // Heliospheric Ram-Pressure Boundaries & Gravitational Deletion Floor R_g
        const double rGravSunAU = engine.gravitationalFloorAU();
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

        // 3. Start ImGui Frame & Draw Smart Deconflicted 2D Viewport HUD
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawSmartViewportHUD(ImGui::GetBackgroundDrawList(), io.DisplaySize, viewProj,
                             heavyRelAU, heavyDescs, parentIdx, engine, renderSettings,
                             showLabels, camera.distanceAU, camera.azimuthDeg, camera.elevationDeg,
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
            const size_t totalObjs = engine.numParticles() + static_cast<size_t>(engine.numHeavy());

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
                if (ImGui::Button("Zoom: Heliosphere (TS 89.4 AU / HP 120.7 AU)", ImVec2(-1.0f, 0.0f))) {
                    camera.lockTarget = false;
                    renderSettings.logScale = false;
                    renderSettings.showHeliosphereShells = true;
                    renderSettings.showHeliopause = true;
                    camera.panOffsetAU = glm::vec3(0.0f);
                    camera.distanceAU = 195.0f;
                }
            }

            if (ImGui::CollapsingHeader("3D Overlays & Viewport Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Show Deconflicted 2D Labels (HUD)", &showLabels);
                ImGui::Checkbox("Show 3D Orbit Trails", &renderSettings.showOrbitTrails);
                ImGui::Checkbox("Show 3R / 4R / 6.784R Magnetospheric Shells", &renderSettings.showVectorChannels);
                ImGui::Checkbox("Show Hill Sphere & Fluid Roche Limit", &renderSettings.showHillAndRoche);
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

                ImGui::Checkbox("1PN Relativistic Precession", &stepParams.relativistic);
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
                std::printf("[Smoke Test] Rendered %d frames (%zu particles + %d heavy bodies)\n",
                            frameIdx, engine.numParticles(), engine.numHeavy());
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
