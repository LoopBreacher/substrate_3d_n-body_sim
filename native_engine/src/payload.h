#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace substrate {

enum class BodyCategory : int {
    Star = 0,
    Planet = 1,
    DwarfPlanet = 2,
    Moon = 3,
};

struct HeavyBodyDesc {
    std::string name;
    std::string parent;
    BodyCategory category = BodyCategory::Planet;
    double mass = 0.0;      // kg
    double radiusM = 1.0e6; // physical mean volumetric radius (m)
    double vol = 1.0;       // exact spherical volume (4/3)*pi*R^3 (m^3) -> drives node density
    double pos[3] = {};     // m
    double vel[3] = {};     // m/s
    float color[4] = {1, 1, 1, 1};
    float size = 4.0f;      // on-screen diameter in pixels

    // Axial rotation & internal structure parameters:
    //   tidalLockMode: 0 = free rotator (uses spinPeriodHours), 1 = 1:1 synchronous tidal lock to parent,
    //                  2 = 3:2 spin-orbit resonance (Mercury: omega_rot = 1.5 * n_orb)
    int tidalLockMode = 0;
    double spinPeriodHours = 24.0;   // Sidereal rotation period in hours (negative = retrograde; computed from orbit if tidalLockMode != 0)
    double obliquityDeg = 0.0;       // Axial tilt relative to ecliptic north (+Z) in degrees
    double poleRaanDeg = 0.0;        // Ascending node of equatorial plane on ecliptic in degrees
    double momentOfInertiaC = 0.33;  // Dimensionless polar moment of inertia C / (M * R^2)
    double spinAxis[3] = {0.0, 0.0, 1.0}; // Unit spin pole vector s_hat in ecliptic Cartesian coordinates
};

// Massless test particles (asteroids, KBOs), stored structure-of-arrays.
struct ParticleSet {
    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;
    std::vector<uint32_t> rgba;   // packed R | G<<8 | B<<16 | A<<24
    std::vector<float> size;      // on-screen diameter in pixels

    size_t count() const { return px.size(); }
    void reserve(size_t n);
};

uint32_t packRGBA(float r, float g, float b, float a);

// Sun, 8 planets, major moons, and 6 dwarf planets with physical IAU/JPL radii & Keplerian states.
std::vector<HeavyBodyDesc> buildHeavyBodies();

// Main asteroid belt (2.06-3.35 AU) + Classical Kuiper Belt (30.0-47.8 AU) & Scattered Disk (47.8-68.0 AU).
ParticleSet buildParticles(size_t numAsteroids, size_t numKBOs, uint64_t seed = 42);

// Splits a total particle budget with the prototype's 1:2 asteroid/KBO ratio.
ParticleSet buildParticles(size_t total, uint64_t seed = 42);

} // namespace substrate
