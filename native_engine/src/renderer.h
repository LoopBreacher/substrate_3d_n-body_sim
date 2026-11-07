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
};

struct TrailVertex {
    glm::vec3 posRelAU{0.0f};
    uint32_t rgba = 0xFFFFFFFFu;
};

struct RenderSettings {
    bool showGrid = true;
    bool showHeliopause = false;            // 20-vertex Heliopause wireframe sphere (R_g / 20)
    bool showDodecahedralShells = true;     // Circumstellar dodecahedral shells: R_g / {50, 36, 30, 27, 20} + R_g
    bool customHeliopauseOverride = false;  // True when user applies custom P_SW wind stagnation radius
    bool showOrbitTrails = true;
    bool showVectorChannels = true;  // 3R / 4R magnetospheric vector-channel shells
    bool showHillAndRoche = true;    // Hill sphere & fluid Roche limit overlays
    bool logScale = false;
    bool additiveParticles = true;
    float particleSizeScale = 1.0f;
    float heavySizeScale = 1.0f;
    float particleAlpha = 0.6f;
    float heliopauseRadiusAU = 121.54f;
};

struct DrawContext {
    glm::mat4 viewProj{1.0f};
    glm::vec3 worldOffsetAU{0.0f};    // -origin position: ecliptic grid is world-anchored
    glm::vec3 starCenterRelAU{0.0f};  // Root star position: astrosphere / heliopause is star-anchored
    glm::vec2 viewportPx{1.0f};
    std::vector<int> heavyDrawOrder;  // far-to-near (painter's algorithm, no depth buffer)
    std::vector<OverlayShell> shells; // Dodecahedral shells, 3R/4R vector shells, Hill sphere, Roche limit
};

class Renderer {
public:
    void init();
    void shutdown();

    void setPayload(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles);
    void updateTrailVertices(const std::vector<TrailVertex>& lineVerts);

    // Maps both interop VBOs into CUDA's address space for the current frame.
    void mapForCuda(cudaStream_t stream, float4** heavyOut, float4** particleOut);
    void unmapFromCuda(cudaStream_t stream);

    void draw(const DrawContext& ctx, const RenderSettings& s);

private:
    void destroyPayload();

    GLuint m_spriteProg = 0, m_lineProg = 0, m_trailProg = 0;
    GLuint m_heavyVao = 0, m_particleVao = 0;
    GLuint m_heavyAttr = 0, m_particleAttr = 0;
    InteropBuffer m_heavyPos, m_particlePos;
    int m_numHeavy = 0;
    size_t m_numParticles = 0;
    bool m_mapped = false;

    GLuint m_gridVao = 0, m_gridVbo = 0;
    GLsizei m_gridVerts = 0;
    GLuint m_helioVao = 0, m_helioVbo = 0;
    GLsizei m_helioVerts = 0;
    GLuint m_ringVao = 0, m_ringVbo = 0;
    GLsizei m_ringVerts = 0;

    GLuint m_trailVao = 0, m_trailVbo = 0;
    GLsizei m_trailVerts = 0;
    size_t m_trailCapacity = 0;
};

} // namespace substrate
