#include "csv_export.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace substrate {

namespace {

std::string csvEscape(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

const char* categoryToString(BodyCategory cat) {
    switch (cat) {
        case BodyCategory::Star:        return "Star";
        case BodyCategory::Planet:      return "Planet";
        case BodyCategory::DwarfPlanet: return "DwarfPlanet";
        case BodyCategory::Moon:        return "Moon";
    }
    return "Unknown";
}

} // namespace

CSVExportSummary exportTelemetryCSV(const std::string& outDir,
                                    const std::vector<ConservationSample>& conservationSamples,
                                    const RingStats& ringStats,
                                    const std::vector<AstroEvent>& events,
                                    const HeavySnapshot& finalSnap,
                                    const std::vector<HeavyBodyDesc>& heavyDescs,
                                    const std::vector<int>& parentIdx,
                                    const SubstrateEngine& engine,
                                    bool relativisticEnabled) {
    CSVExportSummary res{};
    res.outDir = outDir;

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        res.errorMsg = "Failed to create directory '" + outDir + "': " + ec.message();
        return res;
    }

    const fs::path base(outDir);

    // 1. conservation_and_precession.csv
    {
        std::ofstream f(base / "conservation_and_precession.csv");
        if (!f) {
            res.errorMsg = "Failed to write conservation_and_precession.csv";
            return res;
        }
        f << "sim_year,rel_energy_drift,rel_angmom_drift,barycenter_drift_ms,"
             "earth_obliquity_deg,earth_precession_arcsec,mercury_arg_peri_deg\n";
        for (const auto& s : conservationSamples) {
            f << std::fixed << std::setprecision(6) << s.simYear << ','
              << std::scientific << std::setprecision(6) << s.relEnergyDrift << ','
              << s.relAngMomDrift << ','
              << s.barycenterDriftMs << ','
              << std::fixed << std::setprecision(6) << s.earthObliquityDeg << ','
              << std::setprecision(4) << s.earthPrecessionArcsec << ','
              << std::setprecision(6) << s.mercuryArgPeriDeg << '\n';
            ++res.conservationRows;
        }
    }

    // 2. saturn_ring_profile.csv
    {
        const SaturnRingMarkers markers = computeSaturnRingMarkers(finalSnap, heavyDescs, engine);
        std::ofstream f(base / "saturn_ring_profile.csv");
        if (!f) {
            res.errorMsg = "Failed to write saturn_ring_profile.csv";
            return res;
        }
        f << "bin_center_km,particle_count,mimas_2to1_ilr_km,mimas_3to1_ilr_km,roche_ice_km\n";
        constexpr double binWidthKm =
            (RingStats::RING_MAX_KM - RingStats::RING_MIN_KM) / static_cast<double>(RingStats::RING_BINS);
        f << std::fixed;
        for (int b = 0; b < RingStats::RING_BINS; ++b) {
            const double centerKm = RingStats::RING_MIN_KM + (static_cast<double>(b) + 0.5) * binWidthKm;
            f << std::setprecision(1) << centerKm << ','
              << static_cast<uint64_t>(ringStats.radialHist[b] + 0.5f) << ','
              << std::setprecision(2) << markers.mimas2to1_km << ','
              << markers.mimas3to1_km << ','
              << markers.iceRoche_km << '\n';
            ++res.ringRows;
        }
    }

    // 3. close_encounters.csv
    {
        std::ofstream f(base / "close_encounters.csv");
        if (!f) {
            res.errorMsg = "Failed to write close_encounters.csv";
            return res;
        }
        f << "sim_year,event_type,body,reference_body,min_dist_km,min_dist_au,rel_speed_kms,tisserand_tj\n";
        f << std::fixed;
        for (const auto& ev : events) {
            const char* typeStr = (ev.type == AstroEventType::Periapsis) ? "Periapsis" : "CloseEncounter";
            f << std::setprecision(6) << ev.simTimeYears << ','
              << typeStr << ','
              << csvEscape(ev.bodyName) << ','
              << csvEscape(ev.refName) << ','
              << std::setprecision(2) << ev.minDistKm << ','
              << std::setprecision(6) << ev.minDistAU << ','
              << std::setprecision(4) << ev.relSpeedKms << ','
              << std::setprecision(4) << ev.tisserandTJ << '\n';
            ++res.eventRows;
        }
    }

    // 4. body_ephemeris_summary.csv
    {
        std::ofstream f(base / "body_ephemeris_summary.csv");
        if (!f) {
            res.errorMsg = "Failed to write body_ephemeris_summary.csv";
            return res;
        }
        f << "index,name,parent,category,mass_kg,nodes_N,radius_km,a_AU,e,inc_deg,period_days,"
             "spin_period_h,live_obliquity_deg,derived_j2,clock_lag_surface_ppb\n";
        const int M = std::min(engine.numHeavy(), static_cast<int>(heavyDescs.size()));
        for (int i = 0; i < M; ++i) {
            const int pIdx = (i < static_cast<int>(parentIdx.size())) ? parentIdx[i] : -1;
            const KeplerElements kep = computeKeplerElements(i, pIdx, finalSnap, engine);
            const SubstrateBodyAudit aud = computeSubstrateAudit(i, pIdx, finalSnap, kep, engine, relativisticEnabled);
            f << i << ','
              << csvEscape(heavyDescs[i].name) << ','
              << csvEscape(heavyDescs[i].parent) << ','
              << categoryToString(heavyDescs[i].category) << ','
              << std::scientific << std::setprecision(6) << engine.mass()[i] << ','
              << aud.nodesN << ','
              << std::fixed << std::setprecision(3) << (aud.radiusM / 1000.0) << ','
              << std::setprecision(6) << kep.a_AU << ','
              << std::setprecision(6) << kep.e << ','
              << std::setprecision(4) << kep.inc_deg << ','
              << std::setprecision(4) << kep.period_days << ','
              << std::setprecision(4) << aud.spinPeriodHours << ','
              << std::setprecision(5) << aud.liveObliquityDeg << ','
              << std::scientific << std::setprecision(6) << aud.derivedJ2 << ','
              << std::fixed << std::setprecision(5) << aud.clockLagSurface_ppb << '\n';
            ++res.bodyRows;
        }
    }

    res.ok = true;
    return res;
}

} // namespace substrate
