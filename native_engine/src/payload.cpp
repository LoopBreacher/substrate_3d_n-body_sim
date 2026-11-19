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
                   float r, float g, float b, float pxSize,
                   int tidalLockMode = 0, double spinPeriodHours = 24.0,
                   double obliquityDeg = 0.0, double poleRaanDeg = 0.0,
                   double momentOfInertiaC = 0.34) {
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
    d.tidalLockMode = tidalLockMode;
    d.spinPeriodHours = spinPeriodHours;
    d.obliquityDeg = obliquityDeg;
    d.poleRaanDeg = poleRaanDeg;
    d.momentOfInertiaC = momentOfInertiaC;
    const double ob = obliquityDeg * DEG2RAD;
    const double rn = poleRaanDeg * DEG2RAD;
    d.spinAxis[0] = std::sin(ob) * std::sin(rn);
    d.spinAxis[1] = -std::sin(ob) * std::cos(rn);
    d.spinAxis[2] = std::cos(ob);
    return d;
}

double planetJ2(double massKg, double radiusM, double spinPeriodHours, double momentC) {
    const double muSelf = (massKg / M_P) * K_OMEGA;
    const double omegaRot = TWO_PI / (std::abs(spinPeriodHours) * 3600.0);
    return deriveHydrostaticJ2(omegaRot, radiusM, muSelf, momentC);
}

double j2CorrectedMu(double mParent, double rParent, double j2Parent, double mMoon, double aMoon) {
    const double muPair = ((mParent + mMoon) / M_P) * K_OMEGA;
    const double ratio = rParent / aMoon;
    return muPair * (1.0 + 1.5 * j2Parent * ratio * ratio);
}

State6 offsetState(const State6& parent, const State6& rel) {
    return {parent.pos + rel.pos, parent.vel + rel.vel};
}

State6 rotateFromParentEquator(const double spinAxis[3], const State6& eq) {
    const double sx = spinAxis[0], sy = spinAxis[1], sz = spinAxis[2];
    const double sLen = std::sqrt(sx * sx + sy * sy + sz * sz);
    if (sLen < 1e-12) return eq;
    const double zx = sx / sLen, zy = sy / sLen, zz = sz / sLen;
    const double hxy = std::hypot(zx, zy);
    // Ascending node of parent equator on ecliptic: x_eq = normalize((0,0,1) x z_eq) = (-zy, zx, 0)/hxy
    const double xx = (hxy > 1e-12) ? (-zy / hxy) : 1.0;
    const double xy = (hxy > 1e-12) ? ( zx / hxy) : 0.0;
    const double xz = 0.0;
    // y_eq = z_eq x x_eq
    const double yx = zy * xz - zz * xy;
    const double yy = zz * xx - zx * xz;
    const double yz = zx * xy - zy * xx;
    auto rot = [&](const V3& v) -> V3 {
        return {xx * v.x + yx * v.y + zx * v.z,
                xy * v.x + yy * v.y + zy * v.z,
                xz * v.x + yz * v.y + zz * v.z};
    };
    return {rot(eq.pos), rot(eq.vel)};
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

int findBodyIndex(const std::vector<HeavyBodyDesc>& descs, const char* name) {
    if (!name) return -1;
    for (size_t i = 0; i < descs.size(); ++i) {
        if (descs[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

std::vector<HeavyBodyDesc> buildHeavyBodies() {
    std::vector<HeavyBodyDesc> h;
    h.reserve(64);

    const double muSun = (M_SUN / M_P) * K_OMEGA;

    // 0. Root Star: Sun (R = 695,700 km, T_rot = 25.38 d = 609.12 h, obliq = 7.25 deg)
    h.push_back(body("Sun", "None", BodyCategory::Star,
                     M_SUN, 6.957e8, {{0, 0, 0}, {0, 0, 0}}, 1.0f, 0.88f, 0.22f, 22.0f,
                     0, 609.12, 7.25, 75.76, 0.070));

    // 1. Mercury (J2000.0: a = 0.387099 AU, e = 0.205636, i = 7.0050 deg, 3:2 spin-orbit resonance -> 58.646 d)
    const double mercPeriodDays = std::sqrt(4.0 * PI * PI * std::pow(0.387099 * AU, 3) / muSun) / 86400.0;
    const auto mercS = keplerState(muSun, 0.387099 * AU, 0.205636, 7.0050, 48.3308, 29.1270, 176.4929);
    h.push_back(body("Mercury", "Sun", BodyCategory::Planet,
                     3.3011e23, 2.4397e6, mercS, 0.76f, 0.74f, 0.72f, 6.0f,
                     2, (2.0 / 3.0) * mercPeriodDays * 24.0, 0.034, 48.3308, 0.346));

    // 2. Venus (J2000.0: a = 0.723336 AU, e = 0.006777, i = 3.3947 deg, retrograde T_rot = -5832.6 h)
    const auto venS = keplerState(muSun, 0.723336 * AU, 0.006777, 3.3947, 76.6798, 54.9226, 50.9780);
    h.push_back(body("Venus", "Sun", BodyCategory::Planet,
                     4.8675e24, 6.0518e6, venS, 0.92f, 0.68f, 0.28f, 8.0f,
                     0, -5832.6, 177.36, 76.6798, 0.337));

    // 3-4. Earth + Moon (J2000.0; R_Earth = 6,371.0 km -> rho_N = 3.297e30 nodes/m^3, matching Paper 1 Sec 8.4)
    const double mEarth = 5.9722e24;
    const double rEarth = 6.3710e6;
    const double mMoon  = 7.342e22;
    const double j2Earth = planetJ2(mEarth, rEarth, 23.93447, 0.3307);
    const double muEarthMoon = j2CorrectedMu(mEarth, rEarth, j2Earth, mMoon, 3.8440e8);
    const auto earthS = keplerState(muSun, 1.000003 * AU, 0.016711, 0.0000, 0.0000, 102.9377, 357.4425);
    h.push_back(body("Earth", "Sun", BodyCategory::Planet,
                     mEarth, rEarth, earthS, 0.20f, 0.58f, 1.00f, 9.0f,
                     0, 23.93447, 23.4393, 0.0, 0.3307));

    // Earth's Moon orbits outside Earth's equatorial Laplace plane -> keep in ecliptic frame (i = 5.145 deg)
    const auto moonRel = keplerState(muEarthMoon, 3.8440e8, 0.0549, 5.145, 125.0, 318.0, 40.0);
    h.push_back(body("Moon", "Earth", BodyCategory::Moon,
                     mMoon, 1.7374e6, offsetState(earthS, moonRel), 0.82f, 0.82f, 0.85f, 4.0f,
                     1, 655.72, 6.68, 125.0, 0.3929));

    // 5-7. Mars + Phobos + Deimos (J2000.0; regular moons rotated into Mars's 25.19 deg equatorial plane)
    const double mMars   = 6.4171e23;
    const double rMars   = 3.3895e6;
    const double mPhobos = 1.0659e16;
    const double mDeimos = 1.4762e15;
    const double j2Mars  = planetJ2(mMars, rMars, 24.62296, 0.3644);
    const double muMarsPhobos = j2CorrectedMu(mMars, rMars, j2Mars, mPhobos, 9.376e6);
    const double muMarsDeimos = j2CorrectedMu(mMars, rMars, j2Mars, mDeimos, 2.3463e7);
    const auto marsS = keplerState(muSun, 1.523710 * AU, 0.093394, 1.8497, 49.5595, 286.4968, 23.3853);
    h.push_back(body("Mars", "Sun", BodyCategory::Planet,
                     mMars, rMars, marsS, 1.00f, 0.36f, 0.20f, 7.0f,
                     0, 24.62296, 25.19, 35.5, 0.3644));

    const auto phobosRel = rotateFromParentEquator(
        h[5].spinAxis, keplerState(muMarsPhobos, 9.376e6, 0.0151, 1.09, 0.0, 0.0, 25.0));
    h.push_back(body("Phobos", "Mars", BodyCategory::Moon,
                     mPhobos, 1.1267e4, offsetState(marsS, phobosRel), 0.72f, 0.54f, 0.42f, 3.0f,
                     1, 7.6538, 25.19, 35.5, 0.34));

    const auto deimosRel = rotateFromParentEquator(
        h[5].spinAxis, keplerState(muMarsDeimos, 2.3463e7, 0.00033, 1.79, 0.0, 0.0, 165.0));
    h.push_back(body("Deimos", "Mars", BodyCategory::Moon,
                     mDeimos, 6.20e3, offsetState(marsS, deimosRel), 0.62f, 0.46f, 0.44f, 3.0f,
                     1, 30.312, 25.19, 35.5, 0.34));

    // 8-12. Jupiter + Galilean Moons (J2000.0; rotated into Jupiter's 3.13 deg equatorial plane)
    const double mJup = 1.8982e27;
    const double rJup = 6.9911e7;
    const double j2Jup = planetJ2(mJup, rJup, 9.9250, 0.264);
    const double mIo  = 8.9319e22;
    const double mEur = 4.7998e22;
    const double mGan = 1.4819e23;
    const double mCal = 1.0759e23;
    const auto jupS = keplerState(muSun, 5.202887 * AU, 0.048386, 1.3044, 100.4739, 274.2546, 21.6461);
    h.push_back(body("Jupiter", "Sun", BodyCategory::Planet,
                     mJup, rJup, jupS, 0.88f, 0.58f, 0.36f, 14.0f,
                     0, 9.9250, 3.13, 100.4739, 0.264));

    h.push_back(body("Io", "Jupiter", BodyCategory::Moon, mIo, 1.8216e6,
                     offsetState(jupS, rotateFromParentEquator(h[8].spinAxis,
                         keplerState(j2CorrectedMu(mJup, rJup, j2Jup, mIo, 4.2170e8), 4.2170e8, 0.0041, 0.05, 0.0, 0.0, 0.0))),
                     0.96f, 0.92f, 0.22f, 3.5f, 1, 42.459, 3.13, 100.4739, 0.378));
    h.push_back(body("Europa", "Jupiter", BodyCategory::Moon, mEur, 1.5608e6,
                     offsetState(jupS, rotateFromParentEquator(h[8].spinAxis,
                         keplerState(j2CorrectedMu(mJup, rJup, j2Jup, mEur, 6.7103e8), 6.7103e8, 0.0090, 0.47, 0.0, 0.0, 95.0))),
                     0.72f, 0.86f, 0.96f, 3.5f, 1, 85.228, 3.13, 100.4739, 0.346));
    h.push_back(body("Ganymede", "Jupiter", BodyCategory::Moon, mGan, 2.6341e6,
                     offsetState(jupS, rotateFromParentEquator(h[8].spinAxis,
                         keplerState(j2CorrectedMu(mJup, rJup, j2Jup, mGan, 1.0704e9), 1.0704e9, 0.0013, 0.20, 0.0, 0.0, 195.0))),
                     0.64f, 0.64f, 0.66f, 4.2f, 1, 171.709, 3.13, 100.4739, 0.3115));
    h.push_back(body("Callisto", "Jupiter", BodyCategory::Moon, mCal, 2.4103e6,
                     offsetState(jupS, rotateFromParentEquator(h[8].spinAxis,
                         keplerState(j2CorrectedMu(mJup, rJup, j2Jup, mCal, 1.8827e9), 1.8827e9, 0.0074, 0.19, 0.0, 0.0, 290.0))),
                     0.46f, 0.46f, 0.54f, 4.0f, 1, 400.536, 3.13, 100.4739, 0.3549));

    // 13-14. Saturn + Titan (J2000.0; Titan rotated into Saturn's 26.73 deg equatorial plane)
    const double mSat = 5.6834e26;
    const double rSat = 5.8232e7;
    const double j2Sat = planetJ2(mSat, rSat, 10.656, 0.220);
    const double mTit = 1.3452e23;
    const auto satS = keplerState(muSun, 9.536676 * AU, 0.053862, 2.4860, 113.6624, 338.9365, 312.9610);
    h.push_back(body("Saturn", "Sun", BodyCategory::Planet,
                     mSat, rSat, satS, 0.95f, 0.85f, 0.55f, 12.0f,
                     0, 10.656, 26.73, 169.5, 0.220));
    h.push_back(body("Titan", "Saturn", BodyCategory::Moon, mTit, 2.5747e6,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, mTit, 1.22187e9), 1.22187e9, 0.0288, 0.35, 0.0, 0.0, 60.0))),
                     0.85f, 0.62f, 0.22f, 4.2f, 1, 382.68, 26.73, 169.5, 0.341));

    // 15. Uranus (J2000.0)
    const double mUra = 8.6810e25;
    const double rUra = 2.5362e7;
    const double j2Ura = planetJ2(mUra, rUra, -17.24, 0.230);
    const auto uraS = keplerState(muSun, 19.189165 * AU, 0.047257, 0.7726, 74.0169, 96.9374, 145.4469);
    h.push_back(body("Uranus", "Sun", BodyCategory::Planet,
                     mUra, rUra, uraS, 0.50f, 0.82f, 0.92f, 10.0f,
                     0, -17.24, 97.77, 167.7, 0.230));

    // 16-17. Neptune + Retrograde Triton (J2000.0; incLocal = 156.885 deg relative to Neptune's 28.32 deg equator)
    const double mNep = 1.02413e26;
    const double rNep = 2.4622e7;
    const double j2Nep = planetJ2(mNep, rNep, 16.11, 0.241);
    const double mTri = 2.1390e22;
    const auto nepS = keplerState(muSun, 30.069923 * AU, 0.008590, 1.7700, 131.7842, 273.1805, 258.9479);
    h.push_back(body("Neptune", "Sun", BodyCategory::Planet,
                     mNep, rNep, nepS, 0.25f, 0.42f, 0.96f, 10.0f,
                     0, 16.11, 28.32, 131.7842, 0.241));
    h.push_back(body("Triton", "Neptune", BodyCategory::Moon, mTri, 1.3534e6,
                     offsetState(nepS, rotateFromParentEquator(h[16].spinAxis,
                         keplerState(j2CorrectedMu(mNep, rNep, j2Nep, mTri, 3.5476e8), 3.5476e8, 0.00002, 156.885, 0.0, 0.0, 120.0))),
                     0.72f, 0.74f, 0.84f, 3.5f, 1, -141.04, 156.885, 0.0, 0.34));

    // 18. Ceres (Main Belt Dwarf Planet)
    const auto ceresS = keplerState(muSun, 2.7675 * AU, 0.0758, 10.593, 80.31, 73.60, 290.0);
    h.push_back(body("Ceres", "Sun", BodyCategory::DwarfPlanet,
                     9.3835e20, 4.697e5, ceresS, 0.65f, 0.65f, 0.65f, 4.5f,
                     0, 9.074, 4.0, 80.3, 0.36));

    // 19-20. Pluto + Charon (J2000.0; 3:2 Neptune Plutino resonance, mutual 1:1 tidal lock = 153.29 h, ecliptic frame)
    const double mPlu = 1.303e22;
    const double mCha = 1.586e21;
    const double muPluCha = ((mPlu + mCha) / M_P) * K_OMEGA;
    const auto plutoS = keplerState(muSun, 39.482117 * AU, 0.248827, 17.1400, 110.3039, 113.7650, 25.1817);
    h.push_back(body("Pluto", "Sun", BodyCategory::DwarfPlanet,
                     mPlu, 1.1883e6, plutoS, 0.78f, 0.58f, 0.46f, 5.0f,
                     0, -153.2928, 119.59, 110.3039, 0.328));
    h.push_back(body("Charon", "Pluto", BodyCategory::Moon, mCha, 6.060e5,
                     offsetState(plutoS, keplerState(muPluCha, 1.9591e7, 0.0002, 119.59, 110.3039, 0.0, 180.0)),
                     0.56f, 0.56f, 0.58f, 3.2f, 1, -153.2928, 119.59, 110.3039, 0.34));

    // 21-24. Trans-Neptunian Dwarf Planets (Haumea, Makemake, Eris, Sedna)
    h.push_back(body("Haumea", "Sun", BodyCategory::DwarfPlanet, 4.006e21, 7.80e5,
                     keplerState(muSun, 43.116 * AU, 0.1912, 28.19, 121.8, 240.2, 130.0),
                     0.82f, 0.82f, 0.84f, 4.2f, 0, 3.9155, 28.2, 121.8, 0.35));
    h.push_back(body("Makemake", "Sun", BodyCategory::DwarfPlanet, 3.10e21, 7.15e5,
                     keplerState(muSun, 45.430 * AU, 0.1612, 28.98, 79.6, 294.8, 210.0),
                     0.82f, 0.52f, 0.32f, 4.2f, 0, 22.826, 29.0, 79.6, 0.35));
    h.push_back(body("Eris", "Sun", BodyCategory::DwarfPlanet, 1.6466e22, 1.163e6,
                     keplerState(muSun, 67.864 * AU, 0.4361, 44.04, 35.9, 151.6, 135.0),
                     0.92f, 0.92f, 0.95f, 5.0f, 0, 25.90, 44.0, 35.9, 0.35));
    // Sedna near perihelion (q = 76.19 AU, a = 506.0 AU, e = 0.8494)
    h.push_back(body("Sedna", "Sun", BodyCategory::DwarfPlanet, 7.60e21, 4.98e5,
                     keplerState(muSun, 506.0 * AU, 0.8494, 11.93, 144.5, 311.3, 12.0),
                     0.88f, 0.36f, 0.26f, 4.5f, 0, 10.273, 11.9, 144.5, 0.35));

    // 25-30. Saturn's Major Regular & Outer Moons (regular moons rotated into Saturn's 26.73 deg equator; Iapetus in ecliptic)
    // Mimas orbits at 185,539 km — right near Saturn's Fluid Roche limit!
    h.push_back(body("Mimas", "Saturn", BodyCategory::Moon, 3.7493e19, 1.982e5,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 3.7493e19, 1.85539e8), 1.85539e8, 0.0196, 1.57, 0.0, 0.0, 35.0))),
                     0.78f, 0.78f, 0.80f, 3.0f, 1, 22.618, 26.73, 169.5, 0.34));
    h.push_back(body("Enceladus", "Saturn", BodyCategory::Moon, 1.08022e20, 2.521e5,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 1.08022e20, 2.38042e8), 2.38042e8, 0.0047, 0.02, 0.0, 0.0, 110.0))),
                     0.92f, 0.96f, 1.00f, 3.2f, 1, 32.885, 26.73, 169.5, 0.335));
    h.push_back(body("Tethys", "Saturn", BodyCategory::Moon, 6.17449e20, 5.311e5,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 6.17449e20, 2.94619e8), 2.94619e8, 0.0001, 1.12, 0.0, 0.0, 185.0))),
                     0.85f, 0.86f, 0.88f, 3.4f, 1, 45.307, 26.73, 169.5, 0.34));
    h.push_back(body("Dione", "Saturn", BodyCategory::Moon, 1.095452e21, 5.614e5,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 1.095452e21, 3.77396e8), 3.77396e8, 0.0022, 0.02, 0.0, 0.0, 220.0))),
                     0.82f, 0.84f, 0.86f, 3.5f, 1, 65.686, 26.73, 169.5, 0.33));
    h.push_back(body("Rhea", "Saturn", BodyCategory::Moon, 2.306518e21, 7.638e5,
                     offsetState(satS, rotateFromParentEquator(h[13].spinAxis,
                         keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 2.306518e21, 5.27108e8), 5.27108e8, 0.0012, 0.35, 0.0, 0.0, 305.0))),
                     0.80f, 0.78f, 0.76f, 3.8f, 1, 108.42, 26.73, 169.5, 0.37));
    // Iapetus orbits outside Saturn's equatorial Laplace plane -> keep in ecliptic frame (i = 15.47 deg)
    h.push_back(body("Iapetus", "Saturn", BodyCategory::Moon, 1.805635e21, 7.345e5,
                     offsetState(satS, keplerState(j2CorrectedMu(mSat, rSat, j2Sat, 1.805635e21, 3.56082e9), 3.56082e9, 0.0286, 15.47, 142.0, 271.0, 15.0)),
                     0.70f, 0.65f, 0.58f, 3.7f, 1, 1903.7, 15.47, 142.0, 0.34));

    // 31-35. Uranus's Five Major Moons (rotated into Uranus's 97.77 deg sideways equatorial plane)
    h.push_back(body("Miranda", "Uranus", BodyCategory::Moon, 6.4e19, 2.358e5,
                     offsetState(uraS, rotateFromParentEquator(h[15].spinAxis,
                         keplerState(j2CorrectedMu(mUra, rUra, j2Ura, 6.4e19, 1.2990e8), 1.2990e8, 0.0013, 4.34, 0.0, 0.0, 25.0))),
                     0.76f, 0.84f, 0.88f, 3.0f, 1, -33.923, 97.77, 167.7, 0.34));
    h.push_back(body("Ariel", "Uranus", BodyCategory::Moon, 1.251e21, 5.789e5,
                     offsetState(uraS, rotateFromParentEquator(h[15].spinAxis,
                         keplerState(j2CorrectedMu(mUra, rUra, j2Ura, 1.251e21, 1.9090e8), 1.9090e8, 0.0012, 0.04, 0.0, 0.0, 95.0))),
                     0.82f, 0.88f, 0.94f, 3.4f, 1, -60.489, 97.77, 167.7, 0.34));
    h.push_back(body("Umbriel", "Uranus", BodyCategory::Moon, 1.275e21, 5.847e5,
                     offsetState(uraS, rotateFromParentEquator(h[15].spinAxis,
                         keplerState(j2CorrectedMu(mUra, rUra, j2Ura, 1.275e21, 2.6600e8), 2.6600e8, 0.0039, 0.13, 0.0, 0.0, 170.0))),
                     0.58f, 0.62f, 0.68f, 3.4f, 1, -99.460, 97.77, 167.7, 0.34));
    h.push_back(body("Titania", "Uranus", BodyCategory::Moon, 3.400e21, 7.884e5,
                     offsetState(uraS, rotateFromParentEquator(h[15].spinAxis,
                         keplerState(j2CorrectedMu(mUra, rUra, j2Ura, 3.400e21, 4.3630e8), 4.3630e8, 0.0011, 0.08, 0.0, 0.0, 245.0))),
                     0.78f, 0.74f, 0.72f, 3.8f, 1, -208.94, 97.77, 167.7, 0.34));
    h.push_back(body("Oberon", "Uranus", BodyCategory::Moon, 3.076e21, 7.614e5,
                     offsetState(uraS, rotateFromParentEquator(h[15].spinAxis,
                         keplerState(j2CorrectedMu(mUra, rUra, j2Ura, 3.076e21, 5.8350e8), 5.8350e8, 0.0014, 0.07, 0.0, 0.0, 320.0))),
                     0.72f, 0.66f, 0.64f, 3.7f, 1, -323.12, 97.77, 167.7, 0.34));

    // 36-38. Major Main-Belt Asteroids (Vesta, Pallas, Hygiea)
    h.push_back(body("Vesta", "Sun", BodyCategory::DwarfPlanet, 2.59076e20, 2.627e5,
                     keplerState(muSun, 2.3615 * AU, 0.0887, 7.140, 103.85, 151.20, 80.0),
                     0.78f, 0.74f, 0.68f, 4.0f, 0, 5.342, 29.0, 103.85, 0.35));
    h.push_back(body("Pallas", "Sun", BodyCategory::DwarfPlanet, 2.04e20, 2.56e5,
                     keplerState(muSun, 2.7730 * AU, 0.2310, 34.837, 173.09, 310.15, 155.0),
                     0.68f, 0.72f, 0.78f, 4.0f, 0, 7.813, 84.0, 173.09, 0.35));
    h.push_back(body("Hygiea", "Sun", BodyCategory::DwarfPlanet, 8.74e19, 2.17e5,
                     keplerState(muSun, 3.1415 * AU, 0.1125, 3.831, 283.20, 312.32, 215.0),
                     0.58f, 0.60f, 0.64f, 3.8f, 0, 13.825, 60.0, 283.20, 0.35));

    // 39-41. Emergent Lagrange Point Exemplars synchronized to J2000.0 Jupiter & Earth:
    // Seeded in their parent planet's orbital plane at +60 deg (L4) and -60 deg (L5) around the Sun-Planet barycenter:
    const double muSunJup = ((M_SUN + mJup) / M_P) * K_OMEGA;
    h.push_back(body("588 Achilles (Jup L4)", "Sun", BodyCategory::DwarfPlanet, 1.3e18, 6.50e4,
                     keplerState(muSunJup, 5.202887 * AU, 0.048386, 1.3044, 100.4739, 274.2546, 21.6461 + 60.0),
                     0.30f, 0.95f, 0.55f, 4.2f, 0, 7.306, 10.0, 100.4739, 0.35));
    h.push_back(body("617 Patroclus (Jup L5)", "Sun", BodyCategory::DwarfPlanet, 1.36e18, 7.00e4,
                     keplerState(muSunJup, 5.202887 * AU, 0.048386, 1.3044, 100.4739, 274.2546, 21.6461 - 60.0),
                     0.28f, 0.88f, 0.68f, 4.2f, 0, 102.8, 12.0, 100.4739, 0.35));
    const double muSunEarth = ((M_SUN + mEarth) / M_P) * K_OMEGA;
    h.push_back(body("2010 TK7 (Earth L4)", "Sun", BodyCategory::DwarfPlanet, 4.0e10, 1.50e2,
                     keplerState(muSunEarth, 1.000003 * AU, 0.016711, 0.0000, 0.0000, 102.9377, 357.4425 + 60.0),
                     0.35f, 1.00f, 0.65f, 3.8f, 0, 2.40, 0.0, 0.0, 0.35));

    // 42. Periodic Comet 1P/Halley (Retrograde i = 162.26 deg, e = 0.9671, a = 17.834 AU)
    h.push_back(body("1P/Halley", "Sun", BodyCategory::DwarfPlanet, 2.2e14, 5.50e3,
                     keplerState(muSun, 17.834 * AU, 0.96714, 162.26, 58.42, 111.33, 38.0),
                     0.55f, 0.92f, 1.00f, 4.2f, 0, 52.8, 162.26, 58.42, 0.35));

    // 43-44. Additional Neptunian Moons: Proteus (equatorial, 1:1 tidal lock) & Nereid (high-e ecliptic outer moon)
    h.push_back(body("Proteus", "Neptune", BodyCategory::Moon, 4.4e19, 2.10e5,
                     offsetState(nepS, rotateFromParentEquator(h[16].spinAxis,
                         keplerState(j2CorrectedMu(mNep, rNep, j2Nep, 4.4e19, 1.17647e8), 1.17647e8, 0.00053, 0.04, 0.0, 0.0, 40.0))),
                     0.55f, 0.56f, 0.60f, 3.0f, 1, 26.935, 28.32, 131.7842, 0.34));
    h.push_back(body("Nereid", "Neptune", BodyCategory::Moon, 3.1e19, 1.785e5,
                     offsetState(nepS, keplerState(j2CorrectedMu(mNep, rNep, j2Nep, 3.1e19, 5.5138e9), 5.5138e9, 0.7507, 7.09, 326.0, 290.0, 210.0)),
                     0.68f, 0.74f, 0.82f, 3.0f, 0, 11.594, 7.09, 326.0, 0.34));

    // 45-47. Outer-System Centaurs (2060 Chiron, 10199 Chariklo, 5145 Pholus)
    h.push_back(body("2060 Chiron", "Sun", BodyCategory::DwarfPlanet, 4.8e18, 1.05e5,
                     keplerState(muSun, 13.710 * AU, 0.379, 6.93, 209.3, 339.6, 45.0),
                     0.58f, 0.78f, 0.82f, 3.8f, 0, 5.918, 6.93, 209.3, 0.35));
    h.push_back(body("10199 Chariklo", "Sun", BodyCategory::DwarfPlanet, 7.0e18, 1.24e5,
                     keplerState(muSun, 15.822 * AU, 0.172, 23.37, 300.4, 242.1, 120.0),
                     0.68f, 0.66f, 0.72f, 3.8f, 0, 7.004, 23.37, 300.4, 0.35));
    h.push_back(body("5145 Pholus", "Sun", BodyCategory::DwarfPlanet, 6.6e17, 9.5e4,
                     keplerState(muSun, 20.318 * AU, 0.572, 24.69, 119.4, 354.8, 155.0),
                     0.86f, 0.38f, 0.32f, 3.7f, 0, 9.98, 24.69, 119.4, 0.35));

    // 48-50. Major Trans-Neptunian Dwarf Candidates (Quaoar, Gonggong, Orcus)
    h.push_back(body("50000 Quaoar", "Sun", BodyCategory::DwarfPlanet, 1.2e21, 5.45e5,
                     keplerState(muSun, 43.694 * AU, 0.039, 7.99, 188.9, 155.3, 280.0),
                     0.80f, 0.48f, 0.40f, 4.2f, 0, 17.679, 7.99, 188.9, 0.35));
    h.push_back(body("225088 Gonggong", "Sun", BodyCategory::DwarfPlanet, 1.75e21, 6.15e5,
                     keplerState(muSun, 67.485 * AU, 0.503, 30.74, 336.8, 207.7, 190.0),
                     0.88f, 0.34f, 0.28f, 4.3f, 0, 22.40, 30.74, 336.8, 0.35));
    h.push_back(body("90482 Orcus", "Sun", BodyCategory::DwarfPlanet, 6.35e20, 4.55e5,
                     keplerState(muSun, 39.174 * AU, 0.227, 20.59, 268.8, 73.1, 205.0),
                     0.78f, 0.82f, 0.88f, 4.1f, 0, 13.19, 20.59, 268.8, 0.35));

    // 51-53. Near-Earth Asteroids & Jupiter-Family Comet (Apophis, Bennu, 67P/Churyumov-Gerasimenko)
    h.push_back(body("99942 Apophis", "Sun", BodyCategory::DwarfPlanet, 6.1e10, 1.85e2,
                     keplerState(muSun, 0.9224 * AU, 0.1912, 3.331, 204.4, 126.4, 75.0),
                     0.86f, 0.72f, 0.52f, 3.6f, 0, 30.56, 3.331, 204.4, 0.35));
    h.push_back(body("101955 Bennu", "Sun", BodyCategory::DwarfPlanet, 7.329e10, 2.45e2,
                     keplerState(muSun, 1.1264 * AU, 0.2037, 6.035, 2.06, 66.22, 145.0),
                     0.56f, 0.58f, 0.62f, 3.6f, 0, 4.296, 6.035, 2.06, 0.35));
    h.push_back(body("67P/Churyumov-Gerasimenko", "Sun", BodyCategory::DwarfPlanet, 9.982e12, 2.0e3,
                     keplerState(muSun, 3.4628 * AU, 0.6409, 7.040, 50.14, 12.79, 35.0),
                     0.62f, 0.86f, 0.94f, 3.9f, 0, 12.404, 7.040, 50.14, 0.35));

    // 54. Interstellar Hyperbolic Visitor 1I/'Oumuamua (a = -1.2798 AU, e = 1.20113, inbound at 1.82 AU, v_inf = 26.32 km/s)
    h.push_back(body("1I/'Oumuamua", "Sun", BodyCategory::DwarfPlanet, 1.0e9, 1.15e2,
                     keplerState(muSun, -1.2798 * AU, 1.20113, 122.74, 24.60, 241.81, -125.0),
                     0.68f, 0.96f, 1.00f, 4.4f, 0, 7.34, 122.74, 24.60, 0.35));

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
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double nSun = M_SUN / M_P;
    const double muSun = nSun * K_OMEGA;
    const double muSunJup = ((M_SUN + 1.8982e27) / M_P) * K_OMEGA;

    auto pushParticle = [&](const State6& st, uint32_t rgba, float pxSize) {
        ps.px.push_back(st.pos.x);
        ps.py.push_back(st.pos.y);
        ps.pz.push_back(st.pos.z);
        ps.vx.push_back(st.vel.x);
        ps.vy.push_back(st.vel.y);
        ps.vz.push_back(st.vel.z);
        ps.rgba.push_back(rgba);
        ps.size.push_back(pxSize);
    };

    // Exact J2000.0 Jupiter mean-motion Kirkwood resonance locations: a_{p:q} = a_Jup * (q/p)^(2/3)
    constexpr double A_JUP = 5.202887;
    const double a41 = A_JUP * std::pow(1.0 / 4.0, 2.0 / 3.0); // 2.065 AU (4:1 inner boundary)
    const double a31 = A_JUP * std::pow(1.0 / 3.0, 2.0 / 3.0); // 2.501 AU (3:1 Alinda/Hestia gap)
    const double a52 = A_JUP * std::pow(2.0 / 5.0, 2.0 / 3.0); // 2.824 AU (5:2 gap)
    const double a73 = A_JUP * std::pow(3.0 / 7.0, 2.0 / 3.0); // 2.957 AU (7:3 gap)
    const double a21 = A_JUP * std::pow(1.0 / 2.0, 2.0 / 3.0); // 3.278 AU (2:1 Hecuba gap)

    // Inverted Gaussian survival probability across the 5 primary Kirkwood resonances
    auto kirkwoodSurvival = [&](double aAU) {
        auto notch = [](double a, double center, double width, double depth) {
            const double z = (a - center) / width;
            return 1.0 - depth * std::exp(-0.5 * z * z);
        };
        double p = 1.0;
        p *= notch(aAU, a41, 0.024, 0.92);
        p *= notch(aAU, a31, 0.026, 0.94);
        p *= notch(aAU, a52, 0.024, 0.92);
        p *= notch(aAU, a73, 0.018, 0.88);
        p *= notch(aAU, a21, 0.032, 0.95);
        return p;
    };

    auto populateKirkwoodBelt = [&](size_t n, double aMinAU, double aMaxAU,
                                    double eMin, double eMax,
                                    double incSigmaDeg, uint32_t rgba, float pxSize) {
        if (n == 0) return;
        std::uniform_real_distribution<double> uA(aMinAU, aMaxAU);
        std::uniform_real_distribution<double> uE(eMin, eMax);
        std::normal_distribution<double> nInc(0.0, incSigmaDeg);

        for (size_t i = 0; i < n; ++i) {
            double aAU = uA(rng);
            for (int attempt = 0; attempt < 16; ++attempt) {
                if (u01(rng) <= kirkwoodSurvival(aAU)) break;
                aAU = uA(rng);
            }
            const double e = uE(rng);
            const double inc = std::abs(nInc(rng));
            const double raan = uAngle(rng);
            const double argP = uAngle(rng);
            const double nu = uAngle(rng);
            pushParticle(keplerState(muSun, aAU * AU, e, inc, raan, argP, nu), rgba, pxSize);
        }
    };

    // Partition Inner System Asteroid Populations (numAsteroids):
    //   28% Inner Main Belt (2.10 - 2.50 AU, Flora & Vesta families)
    //   32% Middle Main Belt (2.50 - 2.825 AU, Eunomia & Maria families)
    //   20% Outer Main Belt (2.825 - 3.26 AU, Koronis, Themis & Hygiea families)
    //    8% Hilda 3:2 Resonant Triangle (a ~ 3.972 AU)
    //   12% Jupiter L4 (+60 deg Greek) & L5 (-60 deg Trojan) Swarms (a ~ 5.2044 AU)
    const size_t nInnerMB  = (numAsteroids * 28) / 100;
    const size_t nMiddleMB = (numAsteroids * 32) / 100;
    const size_t nOuterMB  = (numAsteroids * 20) / 100;
    const size_t nHilda    = (numAsteroids * 8) / 100;
    const size_t nTrojan   = numAsteroids - (nInnerMB + nMiddleMB + nOuterMB + nHilda);

    populateKirkwoodBelt(nInnerMB,  2.100, 2.500, 0.02, 0.15, 4.5, packRGBA(0.74f, 0.65f, 0.54f, 0.54f), 2.0f);
    populateKirkwoodBelt(nMiddleMB, 2.500, 2.825, 0.03, 0.16, 7.5, packRGBA(0.64f, 0.62f, 0.58f, 0.52f), 2.0f);
    populateKirkwoodBelt(nOuterMB,  2.825, 3.260, 0.02, 0.14, 3.8, packRGBA(0.52f, 0.56f, 0.62f, 0.50f), 2.0f);

    // 2. The Hilda 3:2 Resonant Triangle (a ~ 3.970 AU, 8% of numAsteroids)
    // Resonant angle 3*lambda_Jup - 2*lambda_Hilda - varpi_Hilda ~ 0 deg at aphelion forms a 3-fold triangle
    // with vertices at Jupiter's L4 (+60 deg), L5 (-60 deg), and L3 (+180 deg).
    if (nHilda > 0) {
        const double aHildaCenter = A_JUP * std::pow(2.0 / 3.0, 2.0 / 3.0); // 3.9705 AU
        std::normal_distribution<double> nAHilda(aHildaCenter, 0.035);
        std::uniform_real_distribution<double> uEHilda(0.14, 0.23);
        std::normal_distribution<double> nIncHilda(1.3044, 4.2);
        std::uniform_real_distribution<double> uMHilda(0.0, 360.0);
        std::normal_distribution<double> nLibPhi(0.0, 18.0);
        constexpr double LAMBDA_JUP_DEG = 34.39644; // Jupiter's exact J2000.0 mean longitude
        const uint32_t hildaRGBA = packRGBA(0.92f, 0.66f, 0.34f, 0.58f);

        for (size_t i = 0; i < nHilda; ++i) {
            const double aAU = std::clamp(nAHilda(rng), 3.86, 4.08);
            const double e   = uEHilda(rng);
            const double inc = std::abs(nIncHilda(rng));
            const double Mdeg = uMHilda(rng);
            const double dPhi = nLibPhi(rng);
            const double varpi = 3.0 * LAMBDA_JUP_DEG - 3.0 * Mdeg + dPhi;
            const double raan  = uAngle(rng);
            const double argP  = std::fmod(varpi - raan + 3600.0, 360.0);
            const double Mrad  = Mdeg * DEG2RAD;
            const double nuDeg = Mdeg + (180.0 / PI) * (2.0 * e * std::sin(Mrad) + 1.25 * e * e * std::sin(2.0 * Mrad));
            pushParticle(keplerState(muSun, aAU * AU, e, inc, raan, argP, nuDeg), hildaRGBA, 2.1f);
        }
    }

    // 3. Jupiter L4 (+60 deg Greek Camp) & L5 (-60 deg Trojan Camp) Swarms (a ~ 5.2029 AU, 12% of numAsteroids)
    if (nTrojan > 0) {
        std::normal_distribution<double> nATrojan(A_JUP, 0.032);
        std::uniform_real_distribution<double> uETrojan(0.02, 0.09);
        std::normal_distribution<double> nIncTrojan(1.3044, 6.5);
        std::normal_distribution<double> nRaanTrojan(100.4739, 4.0);
        std::normal_distribution<double> nArgPTrojan(274.2546, 6.0);
        std::normal_distribution<double> nNuLibration(0.0, 10.5);
        const uint32_t trojanRGBA = packRGBA(0.28f, 0.88f, 0.62f, 0.60f);

        for (size_t i = 0; i < nTrojan; ++i) {
            const double campSign = (i % 2 == 0) ? +60.0 : -60.0;
            const double aAU  = std::clamp(nATrojan(rng), 5.08, 5.33);
            const double e    = uETrojan(rng);
            const double inc  = std::abs(nIncTrojan(rng));
            const double raan = nRaanTrojan(rng);
            const double argP = nArgPTrojan(rng);
            const double nu   = 21.6461 + campSign + nNuLibration(rng);
            pushParticle(keplerState(muSunJup, aAU * AU, e, inc, raan, argP, nu), trojanRGBA, 2.1f);
        }
    }

    // 4-8. Outer Trans-Neptunian & Hills / Inner Oort Cloud Populations (numKBOs):
    //   17% 3:2 Resonant Plutinos (39.1 - 39.8 AU)
    //   42% Cold Classical Kuiper Belt (42.0 - 47.6 AU, sharp cutoff at Neptune 2:1 cliff at 47.8 AU)
    //   21% Hot Classical Kuiper Belt (40.2 - 47.6 AU)
    //   10% Scattered Disk (48.0 - 72.0 AU)
    //    5% Detached Sednoids (75.0 - 520.0 AU, q in [42, 80] AU across Termination Shock & Heliopause)
    //    5% Hills / Inner Oort Cloud (520 - 16,500 AU, log-uniform a, q in [60, 2,400] AU, isotropic 3D)
    const size_t nPlutinos  = (numKBOs * 17) / 100;
    const size_t nColdKBO   = (numKBOs * 42) / 100;
    const size_t nHotKBO    = (numKBOs * 21) / 100;
    const size_t nScattered = (numKBOs * 10) / 100;
    const size_t nSednoids  = (numKBOs * 5) / 100;
    const size_t nOort      = numKBOs - (nPlutinos + nColdKBO + nHotKBO + nScattered + nSednoids);

    if (nPlutinos > 0) {
        std::normal_distribution<double> nAPlut(39.45, 0.18);
        std::uniform_real_distribution<double> uEPlut(0.14, 0.27);
        std::normal_distribution<double> nIncPlut(0.0, 11.0);
        const uint32_t plutRGBA = packRGBA(0.76f, 0.56f, 0.92f, 0.48f);
        for (size_t i = 0; i < nPlutinos; ++i) {
            const double aAU = std::clamp(nAPlut(rng), 39.05, 39.85);
            const double e   = uEPlut(rng);
            const double inc = std::abs(nIncPlut(rng));
            pushParticle(keplerState(muSun, aAU * AU, e, inc, uAngle(rng), uAngle(rng), uAngle(rng)), plutRGBA, 2.0f);
        }
    }

    auto populateOuterBelt = [&](size_t n, double aMinAU, double aMaxAU,
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
            pushParticle(keplerState(muSun, a, e, inc, uAngle(rng), uAngle(rng), uAngle(rng)), rgba, pxSize);
        }
    };

    populateOuterBelt(nColdKBO,   42.0, 47.6, 0.005, 0.08,  2.2, packRGBA(0.32f, 0.74f, 0.95f, 0.46f), 2.0f);
    populateOuterBelt(nHotKBO,    40.2, 47.6, 0.060, 0.18, 13.5, packRGBA(0.42f, 0.64f, 0.88f, 0.42f), 2.0f);
    populateOuterBelt(nScattered, 48.0, 72.0, 0.180, 0.42, 15.0, packRGBA(0.48f, 0.58f, 0.96f, 0.38f), 2.0f);

    if (nSednoids > 0) {
        std::uniform_real_distribution<double> uASed(75.0, 520.0);
        std::uniform_real_distribution<double> uQSed(42.0, 80.0);
        std::normal_distribution<double> nIncSed(0.0, 18.0);
        const uint32_t sedRGBA = packRGBA(0.90f, 0.48f, 0.38f, 0.40f);
        for (size_t i = 0; i < nSednoids; ++i) {
            const double aAU = uASed(rng);
            const double qAU = std::min(uQSed(rng), 0.92 * aAU);
            const double e   = std::clamp(1.0 - qAU / aAU, 0.08, 0.88);
            const double inc = std::abs(nIncSed(rng));
            pushParticle(keplerState(muSun, aAU * AU, e, inc, uAngle(rng), uAngle(rng), uAngle(rng)), sedRGBA, 2.0f);
        }
    }

    // 8. Passive Hills / Inner Oort Cloud (520 - 16,500 AU, log-uniform a, q in [60, 2,400] AU, isotropic 3D)
    if (nOort > 0) {
        std::uniform_real_distribution<double> uLogA(std::log(520.0), std::log(16500.0));
        std::uniform_real_distribution<double> uQOort(60.0, 2400.0);
        std::uniform_real_distribution<double> uCosInc(-1.0, 1.0);
        std::uniform_real_distribution<double> uMeanAnom(0.0, TWO_PI);
        const uint32_t oortRGBA = packRGBA(0.72f, 0.78f, 0.92f, 0.34f);
        for (size_t i = 0; i < nOort; ++i) {
            const double aAU = std::exp(uLogA(rng));
            const double qAU = std::min(uQOort(rng), 0.95 * aAU);
            const double e   = std::clamp(1.0 - qAU / aAU, 0.05, 0.985);
            const double inc = std::acos(std::clamp(uCosInc(rng), -1.0, 1.0)) * (180.0 / PI);
            // Solve Kepler's equation M = E - e*sin(E) from uniform mean anomaly for steady-state radial distribution
            const double M = uMeanAnom(rng);
            double E = (M < PI) ? (M + 0.85 * e) : (M - 0.85 * e);
            for (int iter = 0; iter < 6; ++iter) {
                const double f  = E - e * std::sin(E) - M;
                const double df = 1.0 - e * std::cos(E);
                E -= f / df;
            }
            const double nuRad = 2.0 * std::atan2(std::sqrt(1.0 + e) * std::sin(0.5 * E),
                                                  std::sqrt(1.0 - e) * std::cos(0.5 * E));
            const double nuDeg = nuRad * (180.0 / PI);
            pushParticle(keplerState(muSun, aAU * AU, e, inc, uAngle(rng), uAngle(rng), nuDeg), oortRGBA, 2.0f);
        }
    }

    return ps;
}

ParticleSet buildParticles(size_t total, uint64_t seed) {
    const size_t nAst = total / 3;
    return buildParticles(nAst, total - nAst, seed);
}

} // namespace substrate
