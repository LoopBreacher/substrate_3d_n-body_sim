#pragma once

#include <glm/glm.hpp>
#include <imgui.h>

#include <functional>
#include <string>
#include <vector>

#include "astro_metrics.h"
#include "csv_export.h"
#include "payload.h"
#include "renderer.h"
#include "substrate_engine.h"

namespace substrate {

struct ConservationHistory {
    static constexpr int CAPACITY = 240;
    std::vector<float> relEnergyDrift;
    std::vector<float> relAngMomDrift;
    double baseEnergy = 0.0;
    double baseAngMom = 0.0;
    bool initialized = false;

    void reset(double E0, double L0);
    void push(double E, double L);
};

struct TelemetryUIState {
    bool showWindow = true;
    double coarseProbeAU = 120.74;

    // Solar wind stagnation calculator inputs (ACE/Wind 1 AU solar-cycle mean)
    float swDensity1AU_cm3 = 7.0f;
    float swSpeed_kms = 440.0f;

    // Interactive GC Sandbox slider state (log10 of nodes/m^3)
    float sandboxLog10Rho = 0.2f;  // ~1.58 nodes/m^3 when toggled on
    int lastSandboxTarget = -1;

    // Cached GPU belt & Saturnian ring stats (non-blocking async polling)
    BeltStats beltStats{};
    int beltStatsFrameCounter = 0;

    RingStats ringStats{};
    int ringStatsFrameCounter = 0;

    // Live close-encounter & periapsis event observatory
    AstroEventTracker eventTracker{};

    // Conservation & precession time-series log for live CSV export
    ConservationHistory conservation{};
    std::vector<ConservationSample> conservationLog;
    double initialEarthSpinPhi = 0.0;
    bool hasInitialEarthSpinPhi = false;
    double baseMomX = 0.0, baseMomY = 0.0, baseMomZ = 0.0;
    std::string lastExportStatus;
};

// Draws deconflicted 2D HUD labels (collapsing moon badges when zoomed out, radial spider
// leader lines when zoomed in, click-to-select in viewport), emergent L1-L5 Lagrange markers,
// and top-left HUD badge.
void drawSmartViewportHUD(ImDrawList* bgDraw,
                          const ImVec2& displaySize,
                          const glm::mat4& viewProj,
                          const std::vector<glm::vec3>& heavyRelAU,
                          const std::vector<HeavyBodyDesc>& heavyDescs,
                          const std::vector<int>& parentIdx,
                          const SubstrateEngine& engine,
                          const RenderSettings& renderSettings,
                          bool showLabels,
                          const glm::vec3 lpRelAU[5],
                          bool lpValid,
                          float camDistanceAU,
                          float camAzimuthDeg,
                          float camElevationDeg,
                          int& targetIdx,
                          bool& lockTarget,
                          glm::vec3& panOffsetAU);

// Draws the separate toggleable floating "Astrophysics & Substrate Telemetry" window.
void drawTelemetryWindow(TelemetryUIState& ui,
                         const std::vector<HeavyBodyDesc>& heavyDescs,
                         const std::vector<int>& parentIdx,
                         const HeavySnapshot& snap,
                         SubstrateEngine& engine,
                         const StepParams& stepParams,
                         RenderSettings& renderSettings,
                         int& targetIdx,
                         bool& lockTarget,
                         float& camDistanceAU,
                         glm::vec3& panOffsetAU,
                         const std::function<void()>& onResetSolarSystem);

} // namespace substrate
