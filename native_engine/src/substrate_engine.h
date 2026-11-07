#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "payload.h"

namespace substrate {

enum class Precision : int {
    FP64 = 0,   // IEEE double for everything (reference)
    Mixed = 1,  // FP64 state + position differences, FP32 force evaluation, separate
                // accumulators for the relativistic term (test particles only)
};

struct StepParams {
    double dtSeconds = 21600.0;  // macro step (6 h)
    int substeps = 20;           // heavy-body sub-steps per macro step (moon stability)
    bool relativistic = true;    // 1PN relativistic precession multiplier
    Precision precision = Precision::Mixed;
};

struct HeavySnapshot {
    std::vector<double> pos;  // 3*M, interleaved xyz, metres
    std::vector<double> vel;  // 3*M, interleaved xyz, m/s
    double simTimeSeconds = 0.0;
};

// GPU-binned statistical mechanics for up to 1.5M+ belt particles (zero particle host readback).
struct BeltStats {
    static constexpr int MAIN_BINS   = 64;  // a in [2.00, 3.50] AU
    static constexpr int KUIPER_BINS = 84;  // a in [30.0, 72.0] AU (0.50 AU bins capturing k=50 & C=36 cutoffs)
    static constexpr int ECC_BINS    = 32;  // e in [0.00, 0.50]

    static constexpr double KUIPER_MIN_AU = 30.0;
    static constexpr double KUIPER_MAX_AU = 72.0;

    float mainBeltHist[MAIN_BINS]     = {};
    float kuiperBeltHist[KUIPER_BINS] = {};
    float eccHist[ECC_BINS]           = {};

    uint32_t asteroidalCount = 0;  // T_J > 3.0
    uint32_t jfcCount        = 0;  // 2.0 < T_J <= 3.0 (Jupiter-Family Cometary)
    uint32_t halleyCount     = 0;  // T_J <= 2.0 (Damocloid / Halley-type)
    uint32_t unboundCount    = 0;  // Specific orbital energy >= 0
    uint32_t boundCount      = 0;
    float meanEcc            = 0.0f;
    float meanIncDeg         = 0.0f;
};

// GPU Substrate Logistics engine.
class SubstrateEngine {
public:
    SubstrateEngine();
    ~SubstrateEngine();
    SubstrateEngine(const SubstrateEngine&) = delete;
    SubstrateEngine& operator=(const SubstrateEngine&) = delete;

    void load(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles,
              bool correctBarycenter = true);

    void step(const StepParams& p, bool captureSnapshot = false);
    void requestSnapshot();
    const HeavySnapshot& snapshot();

    void writeRenderPositions(float4* heavyOut, float4* particleOut, int originIdx);

    // Interactive Garbage Collection Threshold Sandbox:
    // Overrides a heavy body's node density (nodes/m^3) live on the GPU so crossing
    // RHO_GC = 4.0 nodes/m^3 flips Theta(rho_N - 4) between 1 (Compiled) and 0 (I-Phase).
    void setDensityOverride(int bodyIdx, bool enabled, double overrideRhoNodes, bool relativistic = true);
    void resetDensityOverrides(bool relativistic = true);
    bool isDensityOverridden(int bodyIdx) const;
    double baseDensity(int bodyIdx) const;

    // Computes binned semi-major axis, eccentricity, and Tisserand stats directly on GPU.
    void computeBeltStats(BeltStats& out, double jupiterSemiMajorAU = 5.2044);

    void beginTimer();
    void endTimer();
    float lastTimedMs();

    void copyParticlePositions(std::vector<double>& xyzOut);
    void synchronize();

    int numHeavy() const { return m_numHeavy; }
    size_t numParticles() const { return m_numParticles; }
    double simTimeSeconds() const { return m_simTime; }
    size_t deviceBytes() const { return m_deviceBytes; }
    cudaStream_t stream() const { return m_stream; }

    const std::vector<double>& mass() const { return m_mass; }
    const std::vector<double>& radius() const { return m_radius; }
    const std::vector<double>& nodes() const { return m_nodes; }
    const std::vector<double>& density() const { return m_density; }
    const std::vector<double>& mu() const { return m_mu; }

    double gravitationalFloorAU() const;
    double heliopauseRadiusAU() const;
    static double heliopauseTopologicalDragAU(double swProtonCm3At1AU = 6.0,
                                              double swSpeedKms = 420.0,
                                              double heliosheathStagnationFactor = 1.35);

    double newtonianEnergy(const HeavySnapshot& s) const;      // heavy system, J
    double angularMomentumMag(const HeavySnapshot& s) const;   // heavy system |L|, kg*m^2/s
    double barycenterDriftSpeed(const HeavySnapshot& s) const; // |V_cm|, m/s

private:
    void release();
    void refreshCachedAccelerations(bool relativistic);

    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_snapEvent = nullptr;
    cudaEvent_t m_timerStart = nullptr;
    cudaEvent_t m_timerStop = nullptr;
    bool m_snapPending = false;
    bool m_timerPending = false;
    float m_lastTimedMs = 0.0f;

    int m_numHeavy = 0;
    size_t m_numParticles = 0;
    double m_simTime = 0.0;
    size_t m_deviceBytes = 0;

    double* d_heavy = nullptr;
    double* d_mu = nullptr;
    double* d_rc = nullptr;
    double* d_particles = nullptr;

    // Device & pinned buffers for zero-readback GPU belt histograms (~1 KB)
    uint32_t* d_beltBins = nullptr;
    float* d_beltSums = nullptr;

    double* h_snapPinned = nullptr;
    HeavySnapshot m_snapshot;
    double m_snapTime = 0.0;

    std::vector<double> m_mass, m_radius, m_nodes, m_baseDensity, m_density, m_mu, m_rc;
    std::vector<bool> m_overrideActive;
};

} // namespace substrate
