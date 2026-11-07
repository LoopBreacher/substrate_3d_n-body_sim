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

constexpr double PI_CONST = 3.14159265358979323846;

// --- Sub-Node Quantum & Topological Hardware Anchors ---
constexpr double M_P          = 1.6726219e-27;   // Single-proton mass translation anchor (kg, N_proton = 1)
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

// Dodecahedral cell structural divisors (circumstellar astrosphere & debris-disk sub-shells):
constexpr double V_DODEC        = 20.0;          // 20 vertices of the dodecahedral cell (Astropause: R_g / 20)
constexpr double SHOCK_DIV      = 27.0;          // Stagnation shock divisor 1.35 * 20 = 27 (Termination Shock: R_g / 27)
constexpr double E_DODEC        = 30.0;          // 30 1D structural edge struts (Edge shell: R_g / 30)
constexpr double C_CHANNELS     = 36.0;          // 12 faces * 3 axes = 36 active routing channels (Scattered Disk: R_g / 36)

// Exact algebraic Universal Topological Drag constant: C_delta = (sqrt(5) - 2) / 500 = phi^-3 / 500
constexpr double C_DELTA        = (constSqrt(5.0) - 2.0) / 500.0; // 4.7213595499957939e-4
constexpr double SQRT_C_DELTA   = constSqrt(C_DELTA);             // 0.021728689675163984 (~2.1729%)
constexpr double DRAG_TAX_3D    = 0.017032;      // Volumetric 36-channel routing drag tax (1.7032%)
constexpr double SIGMA_0_RAW    = (2.0 * M_P * C_LIGHT * C_LIGHT) / (V_NODE * T_OMEGA_SQ * KAPPA_5);
constexpr double SIGMA_0        = SIGMA_0_RAW * (1.0 - DRAG_TAX_3D); // Global Resting Cover Ledger (~6.0022e-10 J/m^3)
constexpr double P_DRAG_FLOOR   = C_DELTA * SIGMA_0; // Topological drag pressure floor (~2.8339e-13 Pa = 0.2834 pPa)

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

// Heavy bodies are integrated by a single fused CUDA block, one thread per body.
constexpr int MAX_HEAVY = 128;

} // namespace substrate
