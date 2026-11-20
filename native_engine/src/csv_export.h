#pragma once

#include <string>
#include <vector>

#include "astro_metrics.h"
#include "payload.h"
#include "substrate_engine.h"

namespace substrate {

struct ConservationSample {
    double simYear = 0.0;
    double relEnergyDrift = 0.0;
    double relAngMomDrift = 0.0;
    double barycenterDriftMs = 0.0;
    double earthObliquityDeg = 0.0;
    double earthPrecessionArcsec = 0.0;
    double mercuryArgPeriDeg = 0.0;
};

struct CSVExportSummary {
    bool ok = false;
    std::string outDir;
    std::string errorMsg;
    size_t conservationRows = 0;
    size_t ringRows = 0;
    size_t eventRows = 0;
    size_t bodyRows = 0;
};

// Exports publication-ready CSV files into outDir:
//   - conservation_and_precession.csv
//   - saturn_ring_profile.csv
//   - close_encounters.csv
//   - body_ephemeris_summary.csv
CSVExportSummary exportTelemetryCSV(const std::string& outDir,
                                    const std::vector<ConservationSample>& conservationSamples,
                                    const RingStats& ringStats,
                                    const std::vector<AstroEvent>& events,
                                    const HeavySnapshot& finalSnap,
                                    const std::vector<HeavyBodyDesc>& heavyDescs,
                                    const std::vector<int>& parentIdx,
                                    const SubstrateEngine& engine,
                                    bool relativisticEnabled);

} // namespace substrate
