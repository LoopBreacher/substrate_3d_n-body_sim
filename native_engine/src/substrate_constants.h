#pragma once

// Bare-metal Substrate Logistics hardware constants (shared by host and device code).
// Theoretical foundation — M. Lindenbeck (2026):
//   - "Substrate Logistics: The Master Regulatory Protocol" (https://doi.org/10.5281/zenodo.22127234)
//   - "Substrate Logistics: The Garbage Collection Threshold" (https://doi.org/10.5281/zenodo.22774810)
//   - "Substrate Logistics: The S8 Weak Lensing Tension" (https://doi.org/10.5281/zenodo.23476628)
//   - "Substrate Logistics: The Topological Drag Pressure Floor" (https://doi.org/10.5281/zenodo.24042263)
namespace substrate {

// Compile-time IEEE-754 square root valid in both C++20 constexpr and CUDA __host__/__device__ code.
constexpr double constSqrt(double x) {
    if (x <= 0.0) return 0.0;
    double scale = 1.0;
    while (x >= 4.0) { x *= 0.25; scale *= 2.0; }
    while (x < 0.25) { x *= 4.0;  scale *= 0.5; }
    double g = 1.0;
    for (int i = 0; i < 12; ++i) {
        g = 0.5 * (g + x / g);
    }
    return g * scale;
}

// Compile-time natural logarithm ln(x) valid in both C++20 constexpr and CUDA __host__/__device__ code.
constexpr double constLog(double x) {
    if (x <= 0.0) return 0.0;
    constexpr double LN2 = 0.69314718055994530942;
    int k = 0;
    while (x >= 1.5)  { x *= 0.5; ++k; }
    while (x < 0.75)  { x *= 2.0; --k; }
    const double u = (x - 1.0) / (x + 1.0);
    const double u2 = u * u;
    double term = u;
    double sum = u;
    for (int n = 1; n <= 18; ++n) {
        term *= u2;
        sum += term / static_cast<double>(2 * n + 1);
    }
    return 2.0 * sum + static_cast<double>(k) * LN2;
}

constexpr double PI_CONST = 3.14159265358979323846;

// --- Sub-Node Quantum & Topological Hardware Anchors ---
constexpr double M_P          = 1.6726219e-27;   // Single-proton mass translation anchor (kg, N_proton = 1)
constexpr double M_P_MEV      = 938.272088;      // Single-proton rest mass energy (MeV/c^2)
constexpr double LAMBDA_BAR_P = 2.103089e-16;    // Spatial pixel anchor: reduced proton Compton wavelength (m)
constexpr double V_NODE       = LAMBDA_BAR_P * LAMBDA_BAR_P * LAMBDA_BAR_P; // Sub-node voxel (~9.3019e-48 m^3)
constexpr double C_LIGHT      = 299792458.0;     // Master bus speed invariant (m/s)
constexpr double T_OMEGA      = 1.30147e19;      // Universal Attenuation Tensor (dimensionless cover step-down)
constexpr double T_OMEGA_SQ   = T_OMEGA * T_OMEGA; // ~1.693824e38 (Hierarchy ratio = T_OMEGA^2 / 2 = 8.4691e37)
constexpr double F_OMEGA      = 1.8549e43;       // Master System Clock refresh rate (Hz)

// --- Derived Gravitational & Topological Cover Ledger Constants ---
// Identity: K_OMEGA = (LAMBDA_BAR_P * c^2) / T_OMEGA^2 = 1.11587e-37 m^3 / (node * s^2)
constexpr double K_OMEGA        = 1.11587e-37;   // Substrate Anchor Constant (m^3 / (node * s^2))
constexpr double KAPPA          = 50.0;          // Dodecahedral cell packing constant (120-cell PDS cover)
constexpr double KAPPA_5        = KAPPA * KAPPA * KAPPA * KAPPA * KAPPA; // 5D state-space partition (50^5)

// Exact algebraic Universal Topological Drag constant: C_delta = (sqrt(5) - 2) / 500 = phi^-3 / 500
constexpr double C_DELTA        = (constSqrt(5.0) - 2.0) / 500.0; // 4.7213595499957939e-4
constexpr double SQRT_C_DELTA   = constSqrt(C_DELTA);             // 0.021728689675163984 (~2.1729%)

// Universal Dimensionless Stagnation Operators (Paper 4: Topological Drag Pressure Floor):
constexpr double C_DELTA_FOURTH_ROOT     = constSqrt(SQRT_C_DELTA);        // C_delta^(1/4)  = 0.1474065 (3D mass/energy compression)
constexpr double C_DELTA_INV_FOURTH_ROOT = 1.0 / C_DELTA_FOURTH_ROOT;      // C_delta^(-1/4) = 6.7839605 (3D central-body stagnation multiplier)
constexpr double DISK_TRUNCATION_RATIO   = 0.5 * constLog(1.0 / C_DELTA);  // 0.5 * ln(1/C_delta) = 3.829119 (2D exponential disk truncation R_trunc / R_d)

// --- Micro-Scale Hadronic Saturation Ledger (Sub-Node QCD Confinement & Pion Mass) ---
constexpr double MEV_PER_FM3_TO_PA   = 1.602176634e32;                     // 1 MeV/fm^3 in Pascals (J/m^3)
constexpr double EPSILON_MAX_PA      = (M_P * C_LIGHT * C_LIGHT) / V_NODE; // 1.61608e37 Pa (100.869 GeV/fm^3 single-voxel saturation)
constexpr double P_DELTA_MICRO_PA    = C_DELTA * EPSILON_MAX_PA;           // 7.6301e33 Pa (Micro-scale topological drag floor)
constexpr double B_QCD_MEV_FM3       = P_DELTA_MICRO_PA / MEV_PER_FM3_TO_PA; // 47.624 MeV/fm^3 (Derived MIT Bag Constant B_QCD; B^(1/4) = 145.0 MeV)
constexpr double R_YUKAWA_FM         = 0.2103089 * C_DELTA_INV_FOURTH_ROOT;  // 1.42673 fm (Derived Yukawa pion range 6.784 * lambda_bar_p)
constexpr double M_PION_MEV          = M_P_MEV * C_DELTA_FOURTH_ROOT;      // 138.307 MeV/c^2 (Derived pion mass m_p * C_delta^(1/4))

// --- Macro-Scale Global Resting Cover Ledger & Heliospheric Drag Floor ---
constexpr double DRAG_TAX_3D    = 0.017032;      // Volumetric 36-channel routing drag tax (1.7032%)
constexpr double SIGMA_0_RAW    = (2.0 * M_P * C_LIGHT * C_LIGHT) / (V_NODE * T_OMEGA_SQ * KAPPA_5);
constexpr double SIGMA_0        = SIGMA_0_RAW * (1.0 - DRAG_TAX_3D); // Global Resting Cover Ledger (~6.0022e-10 J/m^3)
constexpr double P_DRAG_FLOOR   = C_DELTA * SIGMA_0; // Macro topological drag pressure floor (~2.8339e-13 Pa = 0.2834 pPa)
constexpr double P_DELTA_MACRO_PA = P_DRAG_FLOOR;
constexpr double P_SW_1AU_PA    = M_P * (7.0e6) * (440000.0 * 440000.0); // 2.2667e-9 Pa (ACE/Wind 1 AU solar-cycle mean: n=7 cm^-3, v=440 km/s)
constexpr double HELIOSHEATH_STANDOFF = 1.35;    // Rankine-Hugoniot subsonic heliosheath standoff ratio R_HP / R_TS

// --- Garbage Collection Compilation Filter (He^4 4-Channel Vector Saturation) ---
constexpr double RHO_GC_EXACT = SIGMA_0 / (M_P * C_LIGHT * C_LIGHT); // ~3.9928 nodes/m^3
constexpr double RHO_GC       = 4.0;             // Integer 4-channel vector saturation ceiling (nodes/m^3 = 1 He^4/m^3)

// --- Zero-Sum Cosmological Cover Partitions (120-Cell Manifold) ---
constexpr double OMEGA_M_VOID        = (12.0 / KAPPA) - SQRT_C_DELTA;       // 0.218271
constexpr double OMEGA_LAMBDA_VOID   = 1.0 - OMEGA_M_VOID;                  // 0.781729
constexpr double OMEGA_M_GLOBAL      = (12.0 / KAPPA) + 1.2 * SQRT_C_DELTA; // 0.266074
constexpr double OMEGA_LAMBDA_GLOBAL = 1.0 - OMEGA_M_GLOBAL;                // 0.733926
constexpr double H_LAMBDA_KMS_MPC    = 59.61;    // Uncompiled pure-void baseline expansion (km/s/Mpc)
constexpr double H_GLOBAL_KMS_MPC    = 67.42;    // CMB global cover average (km/s/Mpc)
constexpr double H_LOCAL_KMS_MPC     = 73.06;    // Local KBC/Cepheid compiled pointer-recycling rate (km/s/Mpc)
constexpr double S8_VOID             = 0.7542;   // Weak lensing void-sheared amplitude
constexpr double S8_GLOBAL           = 0.8326;   // Planck CMB global cover amplitude
constexpr double S8_CORE             = 0.8603;   // Compiled virialized cluster core amplitude

// --- Unit Conversions ---
constexpr double AU                      = 1.495978707e11;         // Astronomical unit (m)
constexpr double MPC_METERS              = 3.0856775814913673e22;  // Megaparsec (m)

// Gravitational Deletion Floor R_g(N): radius where gravitational spatial-deletion energy density
// u_g(R) = m_p * g_Omega(R)^2 / (8 * pi * K_Omega) equals Sigma_{0,Ledger}:
//   R_g(N) = (m_p * N^2 * K_Omega / (8 * pi * SIGMA_0))^(1/4)
//          = lambda_bar_p * sqrt(N) * (kappa^5 / (16 * pi * (1 - Delta_tax)))^(1/4)
constexpr double gravitationalFloorMetersFromNodes(double compiledNodes) {
    if (compiledNodes <= 0.0) return 0.0;
    const double quarticArg = (M_P * compiledNodes * compiledNodes * K_OMEGA) / (8.0 * PI_CONST * SIGMA_0);
    return constSqrt(constSqrt(quarticArg));
}

// Derives rotational River-metric ratio q_rot = 2*(v_eq / v_inflow)^2 and Darwin-Radau hydrostatic J2:
//   sqrt(1 + eta_DR) = (5/2) * (1 - (3/2)*C/(M*R^2))  =>  eta_DR = 6.25 * (1 - 1.5*C)^2 - 1
//   k_2 = (3 - eta_DR) / (2 + eta_DR),  J_2 = (k_2 / 3) * q_rot
constexpr double deriveHydrostaticJ2(double omegaRotRadSec, double radiusM, double muSelf, double momentOfInertiaC) {
    if (radiusM <= 0.0 || muSelf <= 0.0) return 0.0;
    const double gSurf = muSelf / (radiusM * radiusM);
    const double vInflow2 = 2.0 * gSurf * radiusM; // v_inflow^2 = 2 * muSelf / R
    const double vEq = omegaRotRadSec * radiusM;
    const double qRot = (2.0 * vEq * vEq) / vInflow2;
    const double dC = 1.0 - 1.5 * momentOfInertiaC;
    const double etaRaw = 6.25 * dC * dC - 1.0;
    const double etaDR = (etaRaw > 0.0) ? etaRaw : 0.0;
    const double k2 = (3.0 - etaDR) / (2.0 + etaDR);
    return (k2 / 3.0) * qRot;
}

// Heavy bodies are integrated by a single fused CUDA block, one thread per body.
constexpr int MAX_HEAVY = 128;

} // namespace substrate
