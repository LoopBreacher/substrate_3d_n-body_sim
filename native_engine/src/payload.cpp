#include "payload.h"
#include "substrate_constants.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace substrate {

namespace {

constexpr double M_SUN  = 1.9885e30;
constexpr double PI     = 3.14159265358979323846;
constexpr double TWO_PI = 6.28318530717958647692;
constexpr double DEG2RAD = PI / 180.0;

struct V3 {
    double x, y, z;
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
};

struct State6 {
    V3 pos;
    V3 vel;
};

double sphereVolume(double radiusM) {
    return (4.0 / 3.0) * PI * radiusM * radiusM * radiusM;
}

// Exact 3D Keplerian state from (a, e, inc, raan, argPeri, trueAnom) around gravitational parameter mu.
State6 keplerState(double mu, double aMeters, double ecc,
                   double incDeg, double raanDeg, double argPeriDeg, double nuDeg) {
    const double p = aMeters * (1.0 - ecc * ecc);
    const double nu = nuDeg * DEG2RAD;
    const double cosNu = std::cos(nu);
    const double sinNu = std::sin(nu);
    const double r = p / (1.0 + ecc * cosNu);
    const double vScale = std::sqrt(mu / p);

    // Perifocal frame
    const double rxP = r * cosNu;
    const double ryP = r * sinNu;
    const double vxP = -vScale * sinNu;
    const double vyP = vScale * (ecc + cosNu);

    // Rotate by argPeri (w), inclination (i), RAAN (O)
    const double w = argPeriDeg * DEG2RAD;
    const double inc = incDeg * DEG2RAD;
    const double O = raanDeg * DEG2RAD;

    const double cw = std::cos(w), sw = std::sin(w);
    const double ci = std::cos(inc), si = std::sin(inc);
    const double cO = std::cos(O), sO = std::sin(O);

    // Perifocal -> inertial rotation columns P_hat and Q_hat
    const double Px = cO * cw - sO * sw * ci;
    const double Py = sO * cw + cO * sw * ci;
    const double Pz = sw * si;

    const double Qx = -cO * sw - sO * cw * ci;
    const double Qy = -sO * sw + cO * cw * ci;
    const double Qz = cw * si;

    State6 st{};
    st.pos = {Px * rxP + Qx * ryP, Py * rxP + Qy * ryP, Pz * rxP + Qz * ryP};
    st.vel = {Px * vxP + Qx * vyP, Py * vxP + Qy * vyP, Pz * vxP + Qz * vyP};
    return st;
}

HeavyBodyDesc body(const char* name, const char* parent, BodyCategory cat,
                   double mass, double radiusM, const State6& st,
                   float r, float g, float b, float pxSize) {
    HeavyBodyDesc d;
    d.name = name;
    d.parent = parent;
    d.category = cat;
    d.mass = mass;
    d.radiusM = radiusM;
    d.vol = sphereVolume(radiusM);
    d.pos[0] = st.pos.x; d.pos[1] = st.pos.y; d.pos[2] = st.pos.z;
    d.vel[0] = st.vel.x; d.vel[1] = st.vel.y; d.vel[2] = st.vel.z;
    d.color[0] = r; d.color[1] = g; d.color[2] = b; d.color[3] = 1.0f;
    d.size = pxSize;
    return d;
}

State6 offsetState(const State6& parent, const State6& rel) {
    return {parent.pos + rel.pos, parent.vel + rel.vel};
}

} // namespace

void ParticleSet::reserve(size_t n) {
    px.reserve(n); py.reserve(n); pz.reserve(n);
    vx.reserve(n); vy.reserve(n); vz.reserve(n);
    rgba.reserve(n); size.reserve(n);
}

uint32_t packRGBA(float r, float g, float b, float a) {
    auto c = [](float f) { return static_cast<uint32_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f + 0.5f); };
    return c(r) | (c(g) << 8) | (c(b) << 16) | (c(a) << 24);
}

std::vector<HeavyBodyDesc> buildHeavyBodies() {
    std::vector<HeavyBodyDesc> h;
    h.reserve(32);

    const double muSun = (M_SUN / M_P) * K_OMEGA;

    // 0. Root Star: Sun (R = 695,700 km)
    h.push_back(body("Sun", "None", BodyCategory::Star,
                     M_SUN, 6.957e8, {{0, 0, 0}, {0, 0, 0}}, 1.0f, 0.88f, 0.22f, 22.0f));

    // 1. Mercury (a = 0.3871 AU, e = 0.2056, i = 7.00 deg, R = 2,439.7 km)
    const auto mercS = keplerState(muSun, 0.387098 * AU, 0.20563, 7.005, 48.33, 29.12, 15.0);
    h.push_back(body("Mercury", "Sun", BodyCategory::Planet,
                     3.3011e23, 2.4397e6, mercS, 0.76f, 0.74f, 0.72f, 6.0f));

    // 2. Venus (a = 0.7233 AU, e = 0.0068, i = 3.39 deg, R = 6,051.8 km)
    const auto venS = keplerState(muSun, 0.723332 * AU, 0.00677, 3.395, 76.68, 54.88, 110.0);
    h.push_back(body("Venus", "Sun", BodyCategory::Planet,
                     4.8675e24, 6.0518e6, venS, 0.92f, 0.68f, 0.28f, 8.0f));

    // 3-4. Earth + Moon (R_Earth = 6,371.0 km -> rho_N = 3.297e30 nodes/m^3, matching Paper 1 Sec 8.4)
    const double mEarth = 5.9722e24;
    const double muEarth = (mEarth / M_P) * K_OMEGA;
    const auto earthS = keplerState(muSun, 1.000000 * AU, 0.01671, 0.000, 0.0, 102.94, 0.0);
    h.push_back(body("Earth", "Sun", BodyCategory::Planet,
                     mEarth, 6.3710e6, earthS, 0.20f, 0.58f, 1.00f, 9.0f));

    const auto moonRel = keplerState(muEarth, 3.8440e8, 0.0549, 5.145, 125.0, 318.0, 40.0);
    h.push_back(body("Moon", "Earth", BodyCategory::Moon,
                     7.342e22, 1.7374e6, offsetState(earthS, moonRel), 0.82f, 0.82f, 0.85f, 4.0f));

    // 5-7. Mars + Phobos + Deimos (Phobos at 9,376 km hovers right near Mars's fluid Roche limit!)
    const double mMars = 6.4171e23;
    const double muMars = (mMars / M_P) * K_OMEGA;
    const auto marsS = keplerState(muSun, 1.523679 * AU, 0.0934, 1.850, 49.56, 286.50, 205.0);
    h.push_back(body("Mars", "Sun", BodyCategory::Planet,
                     mMars, 3.3895e6, marsS, 1.00f, 0.36f, 0.20f, 7.0f));

    const auto phobosRel = keplerState(muMars, 9.376e6, 0.0151, 1.09, 0.0, 0.0, 25.0);
    h.push_back(body("Phobos", "Mars", BodyCategory::Moon,
                     1.0659e16, 1.1267e4, offsetState(marsS, phobosRel), 0.72f, 0.54f, 0.42f, 3.0f));

    const auto deimosRel = keplerState(muMars, 2.3463e7, 0.00033, 1.79, 0.0, 0.0, 165.0);
    h.push_back(body("Deimos", "Mars", BodyCategory::Moon,
                     1.4762e15, 6.20e3, offsetState(marsS, deimosRel), 0.62f, 0.46f, 0.44f, 3.0f));

    // 8-12. Jupiter + Galilean Moons (Io, Europa, Ganymede, Callisto)
    const double mJup = 1.8982e27;
    const double muJup = (mJup / M_P) * K_OMEGA;
    const auto jupS = keplerState(muSun, 5.2044 * AU, 0.0489, 1.303, 100.46, 273.87, 65.0);
    h.push_back(body("Jupiter", "Sun", BodyCategory::Planet,
                     mJup, 6.9911e7, jupS, 0.88f, 0.58f, 0.36f, 14.0f));

    h.push_back(body("Io", "Jupiter", BodyCategory::Moon, 8.9319e22, 1.8216e6,
                     offsetState(jupS, keplerState(muJup, 4.2170e8, 0.0041, 0.05, 0.0, 0.0, 0.0)),
                     0.96f, 0.92f, 0.22f, 3.5f));
    h.push_back(body("Europa", "Jupiter", BodyCategory::Moon, 4.7998e22, 1.5608e6,
                     offsetState(jupS, keplerState(muJup, 6.7103e8, 0.0090, 0.47, 0.0, 0.0, 95.0)),
                     0.72f, 0.86f, 0.96f, 3.5f));
    h.push_back(body("Ganymede", "Jupiter", BodyCategory::Moon, 1.4819e23, 2.6341e6,
                     offsetState(jupS, keplerState(muJup, 1.0704e9, 0.0013, 0.20, 0.0, 0.0, 195.0)),
                     0.64f, 0.64f, 0.66f, 4.2f));
    h.push_back(body("Callisto", "Jupiter", BodyCategory::Moon, 1.0759e23, 2.4103e6,
                     offsetState(jupS, keplerState(muJup, 1.8827e9, 0.0074, 0.19, 0.0, 0.0, 290.0)),
                     0.46f, 0.46f, 0.54f, 4.0f));

    // 13-14. Saturn + Titan
    const double mSat = 5.6834e26;
    const double muSat = (mSat / M_P) * K_OMEGA;
    const auto satS = keplerState(muSun, 9.5826 * AU, 0.0565, 2.485, 113.66, 339.39, 145.0);
    h.push_back(body("Saturn", "Sun", BodyCategory::Planet,
                     mSat, 5.8232e7, satS, 0.95f, 0.85f, 0.55f, 12.0f));
    h.push_back(body("Titan", "Saturn", BodyCategory::Moon, 1.3452e23, 2.5747e6,
                     offsetState(satS, keplerState(muSat, 1.22187e9, 0.0288, 0.35, 0.0, 0.0, 60.0)),
                     0.85f, 0.62f, 0.22f, 4.2f));

    // 15. Uranus
    const auto uraS = keplerState(muSun, 19.2184 * AU, 0.0472, 0.773, 74.01, 96.99, 240.0);
    h.push_back(body("Uranus", "Sun", BodyCategory::Planet,
                     8.6810e25, 2.5362e7, uraS, 0.50f, 0.82f, 0.92f, 10.0f));

    // 16-17. Neptune + Retrograde Triton (inc = 156.885 deg)
    const double mNep = 1.02413e26;
    const double muNep = (mNep / M_P) * K_OMEGA;
    const auto nepS = keplerState(muSun, 30.110 * AU, 0.0086, 1.770, 131.78, 276.34, 315.0);
    h.push_back(body("Neptune", "Sun", BodyCategory::Planet,
                     mNep, 2.4622e7, nepS, 0.25f, 0.42f, 0.96f, 10.0f));
    h.push_back(body("Triton", "Neptune", BodyCategory::Moon, 2.1390e22, 1.3534e6,
                     offsetState(nepS, keplerState(muNep, 3.5476e8, 0.00002, 156.885, 0.0, 0.0, 120.0)),
                     0.72f, 0.74f, 0.84f, 3.5f));

    // 18. Ceres (Main Belt Dwarf Planet)
    const auto ceresS = keplerState(muSun, 2.7675 * AU, 0.0758, 10.593, 80.31, 73.60, 290.0);
    h.push_back(body("Ceres", "Sun", BodyCategory::DwarfPlanet,
                     9.3835e20, 4.697e5, ceresS, 0.65f, 0.65f, 0.65f, 4.5f));

    // 19-20. Pluto + Charon (3:2 Neptune Plutino resonance, e = 0.2488, i = 17.16 deg)
    const double mPlu = 1.303e22;
    const double mCha = 1.586e21;
    const double muPluCha = ((mPlu + mCha) / M_P) * K_OMEGA;
    const auto plutoS = keplerState(muSun, 39.482 * AU, 0.2488, 17.16, 110.30, 113.83, 45.0);
    h.push_back(body("Pluto", "Sun", BodyCategory::DwarfPlanet,
                     mPlu, 1.1883e6, plutoS, 0.78f, 0.58f, 0.46f, 5.0f));
    h.push_back(body("Charon", "Pluto", BodyCategory::Moon, mCha, 6.060e5,
                     offsetState(plutoS, keplerState(muPluCha, 1.9591e7, 0.0002, 0.08, 0.0, 0.0, 180.0)),
                     0.56f, 0.56f, 0.58f, 3.2f));

    // 21-24. Trans-Neptunian Dwarf Planets (Haumea, Makemake, Eris, Sedna)
    h.push_back(body("Haumea", "Sun", BodyCategory::DwarfPlanet, 4.006e21, 7.80e5,
                     keplerState(muSun, 43.116 * AU, 0.1912, 28.19, 121.8, 240.2, 130.0),
                     0.82f, 0.82f, 0.84f, 4.2f));
    h.push_back(body("Makemake", "Sun", BodyCategory::DwarfPlanet, 3.10e21, 7.15e5,
                     keplerState(muSun, 45.430 * AU, 0.1612, 28.98, 79.6, 294.8, 210.0),
                     0.82f, 0.52f, 0.32f, 4.2f));
    h.push_back(body("Eris", "Sun", BodyCategory::DwarfPlanet, 1.6466e22, 1.163e6,
                     keplerState(muSun, 67.864 * AU, 0.4361, 44.04, 35.9, 151.6, 135.0),
                     0.92f, 0.92f, 0.95f, 5.0f));
    // Sedna near perihelion (q = 76.19 AU, a = 506.0 AU, e = 0.8494)
    h.push_back(body("Sedna", "Sun", BodyCategory::DwarfPlanet, 7.60e21, 4.98e5,
                     keplerState(muSun, 506.0 * AU, 0.8494, 11.93, 144.5, 311.3, 12.0),
                     0.88f, 0.36f, 0.26f, 4.5f));

    // Balance each planet/dwarf + moon subsystem around its exact heliocentric barycenter
    // (critical for Pluto-Charon and Earth-Moon so subsystem CoM sits on the heliocentric Kepler orbit).
    for (size_t p = 1; p < h.size(); ++p) {
        if (h[p].category == BodyCategory::Moon) continue;
        double sysMass = h[p].mass;
        double dPos[3] = {0.0, 0.0, 0.0};
        double dVel[3] = {0.0, 0.0, 0.0};
        std::vector<size_t> moons;
        for (size_t m = 0; m < h.size(); ++m) {
            if (h[m].parent == h[p].name) {
                moons.push_back(m);
                sysMass += h[m].mass;
                for (int k = 0; k < 3; ++k) {
                    dPos[k] += h[m].mass * (h[m].pos[k] - h[p].pos[k]);
                    dVel[k] += h[m].mass * (h[m].vel[k] - h[p].vel[k]);
                }
            }
        }
        if (!moons.empty()) {
            for (int k = 0; k < 3; ++k) {
                const double shiftP = dPos[k] / sysMass;
                const double shiftV = dVel[k] / sysMass;
                h[p].pos[k] -= shiftP;
                h[p].vel[k] -= shiftV;
                for (size_t m : moons) {
                    h[m].pos[k] -= shiftP;
                    h[m].vel[k] -= shiftV;
                }
            }
        }
    }

    return h;
}

ParticleSet buildParticles(size_t numAsteroids, size_t numKBOs, uint64_t seed) {
    ParticleSet ps;
    ps.reserve(numAsteroids + numKBOs);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uAngle(0.0, 360.0);
    const double nSun = M_SUN / M_P;
    const double muSun = nSun * K_OMEGA;

    // Dodecahedral circumstellar boundaries derived directly from the Sun's integer node load:
    //   R_g(N_sun) / 50 = ~48.62 AU (Classical Kuiper Cliff truncation)
    //   R_g(N_sun) / 36 = ~67.52 AU (36-channel Scattered Disk envelope; cf. Eris SMA = 67.86 AU)
    const double rGravSunAU   = gravitationalFloorMetersFromNodes(nSun) / AU;
    const double rKuiperCliff = rGravSunAU / KAPPA;
    const double rScattered36 = rGravSunAU / C_CHANNELS;

    auto populateBelt = [&](size_t n, double aMinAU, double aMaxAU,
                            double eMin, double eMax,
                            double incSigmaDeg, uint32_t rgba, float pxSize) {
        if (n == 0) return;
        std::uniform_real_distribution<double> uA(aMinAU * AU, aMaxAU * AU);
        std::uniform_real_distribution<double> uE(eMin, eMax);
        std::normal_distribution<double> nInc(0.0, incSigmaDeg);

        for (size_t i = 0; i < n; ++i) {
            const double a = uA(rng);
            const double e = uE(rng);
            const double inc = std::abs(nInc(rng));
            const double raan = uAngle(rng);
            const double argP = uAngle(rng);
            const double nu = uAngle(rng);

            const State6 st = keplerState(muSun, a, e, inc, raan, argP, nu);
            ps.px.push_back(st.pos.x);
            ps.py.push_back(st.pos.y);
            ps.pz.push_back(st.pos.z);
            ps.vx.push_back(st.vel.x);
            ps.vy.push_back(st.vel.y);
            ps.vz.push_back(st.vel.z);
            ps.rgba.push_back(rgba);
            ps.size.push_back(pxSize);
        }
    };

    const size_t nClassicalKBO = (numKBOs * 9) / 10;
    const size_t nScatteredKBO = numKBOs - nClassicalKBO;

    // 1. Main Asteroid Belt (2.06 - 3.35 AU)
    populateBelt(numAsteroids, 2.06, 3.35, 0.002, 0.14, 5.5,
                 packRGBA(0.64f, 0.62f, 0.58f, 0.52f), 2.0f);
    // 2. Classical Kuiper Belt (30.0 AU -> derived k=50 Kuiper Cliff ~48.62 AU, 90% of trans-Neptunian belt)
    populateBelt(nClassicalKBO, 30.0, rKuiperCliff, 0.002, 0.15, 6.0,
                 packRGBA(0.35f, 0.68f, 0.90f, 0.42f), 2.0f);
    // 3. Sparse Scattered Disk (k=50 Cliff ~48.62 AU -> C=36 Routing Shell ~67.52 AU, 10% tail)
    populateBelt(nScatteredKBO, rKuiperCliff, rScattered36, 0.15, 0.38, 14.0,
                 packRGBA(0.48f, 0.60f, 0.96f, 0.36f), 2.0f);
    return ps;
}

ParticleSet buildParticles(size_t total, uint64_t seed) {
    const size_t nAst = total / 3;
    return buildParticles(nAst, total - nAst, seed);
}

} // namespace substrate
