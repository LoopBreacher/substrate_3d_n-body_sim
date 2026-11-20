#pragma once

#include <glad/gl.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

#include "payload.h"

struct cudaGraphicsResource;

namespace substrate {

// OpenGL VBO registered with CUDA. CUDA kernels write straight into it; no host copies.
class InteropBuffer {
public:
    void create(size_t bytes);
    void destroy();
    GLuint vbo() const { return m_vbo; }
    cudaGraphicsResource* resource() const { return m_res; }

private:
    GLuint m_vbo = 0;
    cudaGraphicsResource* m_res = nullptr;
};

struct OverlayShell {
    glm::vec3 centerRelAU{0.0f};
    float radiusAU = 1.0f;
    glm::vec4 color{1.0f};
    bool sphereWireframe = true;
    bool equatorialRing = true;
    glm::mat3 basis{1.0f};      // Orthonormal basis with column 2 aligned to body's 3D spin axis s_hat
    bool drawSpinAxis = false;  // Draws North-South spin pole vector + rotating prime-meridian spoke
    float spinPhaseRad = 0.0f;  // Instantaneous rotation angle phi(t) = omega_rot * t (or sub-parent longitude)
};

struct TrailVertex {
    glm::vec3 posRelAU{0.0f};
    uint32_t rgba = 0xFFFFFFFFu;
};

struct RenderSettings {
    bool showGrid = true;
    bool showHeliopause = false;            // Ram-pressure Heliopause wireframe sphere (120.74 AU)
    bool showHeliosphereShells = true;      // Termination Shock ring (89.44 AU), Heliopause ring (120.74 AU) & Grav. Floor R_g (2,430.82 AU)
    bool customHeliopauseOverride = false;  // True when user applies custom P_SW wind stagnation radius
    bool showOrbitTrails = true;
    bool showVectorChannels = true;         // 1R / 3R / 4R / 6.784R magnetospheric shells
    bool showHillAndRoche = true;           // Hill sphere & fluid Roche limit overlays
    bool showLagrangePoints = true;         // Live emergent 3D Lagrange points (L1-L5) for active target
    bool showSaturnRings = true;            // Dedicated passive 40k GPU Saturnian ring system
    bool logScale = false;
    bool additiveParticles = true;
    float particleSizeScale = 1.0f;
    float heavySizeScale = 1.0f;
    float particleAlpha = 0.6f;
    float terminationShockRadiusAU = 89.44f;
    float heliopauseRadiusAU = 120.74f;
};

struct DrawContext {
    glm::mat4 viewProj{1.0f};
    glm::mat4 view{1.0f};
    float camDistanceAU = 35.0f;
    glm::vec3 worldOffsetAU{0.0f};    // -origin position: ecliptic grid is world-anchored
    glm::vec3 starCenterRelAU{0.0f};  // Root star position: heliosphere is star-anchored
    glm::vec2 viewportPx{1.0f};
    std::vector<int> heavyDrawOrder;  // far-to-near (painter's algorithm, no depth buffer)
    std::vector<OverlayShell> shells; // Heliospheric rings, 3R/4R/6.784R magnetospheric shells, Hill sphere, Roche limit
};

class Renderer {
public:
    void init();
    void shutdown();

    void setPayload(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles,
                    const ParticleSet* saturnRings = nullptr);
    void updateTrailVertices(const std::vector<TrailVertex>& lineVerts);

    // Maps active interop VBOs into CUDA's address space for the current frame.
    void mapForCuda(cudaStream_t stream, float4** heavyOut, float4** particleOut, float4** ringOut = nullptr);
    void unmapFromCuda(cudaStream_t stream);

    void draw(const DrawContext& ctx, const RenderSettings& s);

private:
    void destroyPayload();

    GLuint m_spriteProg = 0, m_lineProg = 0, m_trailProg = 0;
    GLuint m_heavyVao = 0, m_particleVao = 0, m_satRingVao = 0;
    GLuint m_heavyAttr = 0, m_particleAttr = 0, m_satRingAttr = 0;
    InteropBuffer m_heavyPos, m_particlePos, m_ringPos;
    int m_numHeavy = 0;
    size_t m_numParticles = 0;
    size_t m_numRings = 0;
    bool m_mapped = false;

    GLuint m_gridVao = 0, m_gridVbo = 0;
    GLsizei m_gridVerts = 0;
    GLuint m_helioVao = 0, m_helioVbo = 0;
    GLsizei m_helioVerts = 0;
    GLuint m_ringVao = 0, m_ringVbo = 0;
    GLsizei m_ringVerts = 0;
    GLuint m_spinAxisVao = 0, m_spinAxisVbo = 0;
    GLsizei m_spinAxisVerts = 0;

    GLuint m_trailVao = 0, m_trailVbo = 0;
    GLsizei m_trailVerts = 0;
    size_t m_trailCapacity = 0;
};

} // namespace substrate
