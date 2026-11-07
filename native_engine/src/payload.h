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

// Main asteroid belt (2.1-3.3 AU) + Kuiper belt (30-50 AU) on eccentric/inclined Keplerian orbits.
ParticleSet buildParticles(size_t numAsteroids, size_t numKBOs, uint64_t seed = 42);

// Splits a total particle budget with the prototype's 1:2 asteroid/KBO ratio.
ParticleSet buildParticles(size_t total, uint64_t seed = 42);

} // namespace substrate
