#include "renderer.h"

// glad must precede any header that pulls in GL/gl.h (cuda_gl_interop.h does).
#include <cuda_gl_interop.h>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace substrate {

namespace {

void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Shared GLSL helper: logarithmic radial compression around the world origin (Sun).
const char* kLogFn = R"(
vec3 logCompress(vec3 p) {
    float r = length(p);
    if (r < 1e-6) return p;
    return p / r * (log(1.0 + r * 10.0) * 0.43429448190325182);
}
)";

// Instanced point sprites: one camera-facing quad per body, expanded in clip space.
const char* kSpriteVS = R"(
layout(location = 0) in vec4 aPos;    // per instance: AU, origin-relative (CUDA-written)
layout(location = 1) in vec4 aColor;  // per instance: normalized RGBA8
layout(location = 2) in vec2 aSize;   // per instance: x = diameter px, y = emissive flag

uniform mat4 uViewProj;
uniform vec2 uViewport;
uniform bool uLog;
uniform float uSizeScale;
uniform float uAlphaScale;

out vec2 vUV;
out vec4 vColor;
flat out float vEmissive;
flat out float vLit;

void main() {
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    vec3 p = aPos.xyz;
    if (uLog) p = logCompress(p);
    vec4 clip = uViewProj * vec4(p, 1.0);
    float size = max(aSize.x * uSizeScale, 1.0);
    clip.xy += corner * size / uViewport * clip.w;
    gl_Position = clip;
    vUV = corner;
    vColor = vec4(aColor.rgb, aColor.a * uAlphaScale);
    vEmissive = aSize.y;
    vLit = size > 3.5 ? 1.0 : 0.0;
}
)";

const char* kSpriteFS = R"(
in vec2 vUV;
in vec4 vColor;
flat in float vEmissive;
flat in float vLit;
out vec4 fragColor;

void main() {
    float r2 = dot(vUV, vUV);
    if (r2 > 1.0) discard;

    if (vLit < 0.5) {
        float a = 1.0 - smoothstep(0.35, 1.0, r2);
        fragColor = vec4(vColor.rgb, vColor.a * a);
        return;
    }

    float edge = clamp((1.0 - r2) / max(fwidth(r2), 1e-4), 0.0, 1.0);
    vec3 n = vec3(vUV, sqrt(1.0 - r2));
    if (vEmissive > 0.5) {
        vec3 c = vColor.rgb * (0.7 + 0.3 * n.z) + vec3(0.25) * pow(n.z, 6.0);
        fragColor = vec4(c, edge);
        return;
    }
    vec3 L = normalize(vec3(-0.4, 0.5, 0.75));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0) * 0.3;
    fragColor = vec4(vColor.rgb * (0.25 + 0.85 * diff) + vec3(spec), edge);
}
)";

const char* kLineVS = R"(
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform vec3 uOffset;
uniform float uScale;
uniform bool uLog;
void main() {
    vec3 p = aPos * uScale + uOffset;
    if (uLog) p = logCompress(p);
    gl_Position = uViewProj * vec4(p, 1.0);
}
)";

const char* kLineFS = R"(
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }
)";

const char* kTrailVS = R"(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uViewProj;
uniform bool uLog;
out vec4 vColor;
void main() {
    vec3 p = aPos;
    if (uLog) p = logCompress(p);
    gl_Position = uViewProj * vec4(p, 1.0);
    vColor = aColor;
}
)";

const char* kTrailFS = R"(
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)";

GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("Shader compile failed:\n") + log);
    }
    return s;
}

GLuint link(const char* vs, const char* fs, bool needsLog) {
    const std::string header = "#version 460 core\n";
    const std::string vsSrc = header + (needsLog ? kLogFn : "") + vs;
    const std::string fsSrc = header + fs;
    GLuint v = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint f = compile(GL_FRAGMENT_SHADER, fsSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("Program link failed:\n") + log);
    }
    return p;
}

void makeLineVao(const std::vector<float>& verts, GLuint& vao, GLuint& vbo, GLsizei& count) {
    glCreateBuffers(1, &vbo);
    glNamedBufferStorage(vbo, verts.size() * sizeof(float), verts.data(), 0);
    glCreateVertexArrays(1, &vao);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0);
    count = static_cast<GLsizei>(verts.size() / 3);
}

struct SpriteAttr {
    uint32_t rgba;
    float size;
    float emissive;
};

GLuint makeSpriteVao(GLuint posVbo, GLuint attrVbo) {
    GLuint vao;
    glCreateVertexArrays(1, &vao);
    glVertexArrayVertexBuffer(vao, 0, posVbo, 0, sizeof(float) * 4);
    glVertexArrayBindingDivisor(vao, 0, 1);
    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0);

    glVertexArrayVertexBuffer(vao, 1, attrVbo, 0, sizeof(SpriteAttr));
    glVertexArrayBindingDivisor(vao, 1, 1);
    glEnableVertexArrayAttrib(vao, 1);
    glVertexArrayAttribFormat(vao, 1, 4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(SpriteAttr, rgba));
    glVertexArrayAttribBinding(vao, 1, 1);
    glEnableVertexArrayAttrib(vao, 2);
    glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(SpriteAttr, size));
    glVertexArrayAttribBinding(vao, 2, 1);
    return vao;
}

} // namespace

// ---------------------------------------------------------------------------
void InteropBuffer::create(size_t bytes) {
    destroy();
    glCreateBuffers(1, &m_vbo);
    glNamedBufferData(m_vbo, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_DRAW);
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_res, m_vbo, cudaGraphicsRegisterFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");
}

void InteropBuffer::destroy() {
    if (m_res) { cudaGraphicsUnregisterResource(m_res); m_res = nullptr; }
    if (m_vbo) { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
}

// ---------------------------------------------------------------------------
void Renderer::init() {
    m_spriteProg = link(kSpriteVS, kSpriteFS, true);
    m_lineProg = link(kLineVS, kLineFS, true);
    m_trailProg = link(kTrailVS, kTrailFS, true);

    // Reference grid in the ecliptic (XY) plane, +-150 AU, 10 AU spacing.
    std::vector<float> grid;
    const float ext = 150.0f, step = 10.0f, seg = 2.0f;
    for (float c = -ext; c <= ext + 0.01f; c += step) {
        for (float t = -ext; t < ext - 0.01f; t += seg) {
            grid.insert(grid.end(), {c, t, 0.0f, c, t + seg, 0.0f});
            grid.insert(grid.end(), {t, c, 0.0f, t + seg, c, 0.0f});
        }
    }
    makeLineVao(grid, m_gridVao, m_gridVbo, m_gridVerts);

    // Reusable unit-radius wireframe sphere (for Heliopause, 3R/4R Vector Channels, Hill Sphere).
    std::vector<float> sph;
    const int n = 96;
    const float PI = 3.14159265358979f;
    for (int lat = -60; lat <= 60; lat += 30) {
        const float phi = lat * PI / 180.0f, rz = std::sin(phi), rr = std::cos(phi);
        for (int i = 0; i < n; ++i) {
            const float a0 = 2 * PI * i / n, a1 = 2 * PI * (i + 1) / n;
            sph.insert(sph.end(), {rr * std::cos(a0), rr * std::sin(a0), rz,
                                   rr * std::cos(a1), rr * std::sin(a1), rz});
        }
    }
    for (int lon = 0; lon < 180; lon += 45) {
        const float th = lon * PI / 180.0f;
        for (int i = 0; i < n; ++i) {
            const float a0 = 2 * PI * i / n, a1 = 2 * PI * (i + 1) / n;
            sph.insert(sph.end(), {std::sin(a0) * std::cos(th), std::sin(a0) * std::sin(th), std::cos(a0),
                                   std::sin(a1) * std::cos(th), std::sin(a1) * std::sin(th), std::cos(a1)});
        }
    }
    makeLineVao(sph, m_helioVao, m_helioVbo, m_helioVerts);

    // Reusable unit-radius equatorial ring (accentuates equatorial plane & Roche limit).
    std::vector<float> ring;
    const int nRing = 128;
    for (int i = 0; i < nRing; ++i) {
        const float a0 = 2 * PI * i / nRing, a1 = 2 * PI * (i + 1) / nRing;
        ring.insert(ring.end(), {std::cos(a0), std::sin(a0), 0.0f,
                                 std::cos(a1), std::sin(a1), 0.0f});
    }
    makeLineVao(ring, m_ringVao, m_ringVbo, m_ringVerts);

    // Dynamic orbit trail VAO/VBO
    glCreateBuffers(1, &m_trailVbo);
    glCreateVertexArrays(1, &m_trailVao);
    glVertexArrayVertexBuffer(m_trailVao, 0, m_trailVbo, 0, sizeof(TrailVertex));
    glEnableVertexArrayAttrib(m_trailVao, 0);
    glVertexArrayAttribFormat(m_trailVao, 0, 3, GL_FLOAT, GL_FALSE, offsetof(TrailVertex, posRelAU));
    glVertexArrayAttribBinding(m_trailVao, 0, 0);
    glEnableVertexArrayAttrib(m_trailVao, 1);
    glVertexArrayAttribFormat(m_trailVao, 1, 4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(TrailVertex, rgba));
    glVertexArrayAttribBinding(m_trailVao, 1, 0);
}

void Renderer::updateTrailVertices(const std::vector<TrailVertex>& lineVerts) {
    m_trailVerts = static_cast<GLsizei>(lineVerts.size());
    if (lineVerts.empty()) return;
    if (lineVerts.size() > m_trailCapacity) {
        m_trailCapacity = lineVerts.size() * 2;
        glNamedBufferData(m_trailVbo, static_cast<GLsizeiptr>(m_trailCapacity * sizeof(TrailVertex)),
                          nullptr, GL_STREAM_DRAW);
    }
    glNamedBufferSubData(m_trailVbo, 0, static_cast<GLsizeiptr>(lineVerts.size() * sizeof(TrailVertex)),
                         lineVerts.data());
}

void Renderer::destroyPayload() {
    if (m_mapped) unmapFromCuda(nullptr);
    m_heavyPos.destroy();
    m_particlePos.destroy();
    if (m_heavyVao) glDeleteVertexArrays(1, &m_heavyVao);
    if (m_particleVao) glDeleteVertexArrays(1, &m_particleVao);
    if (m_heavyAttr) glDeleteBuffers(1, &m_heavyAttr);
    if (m_particleAttr) glDeleteBuffers(1, &m_particleAttr);
    m_heavyVao = m_particleVao = m_heavyAttr = m_particleAttr = 0;
    m_numHeavy = 0;
    m_numParticles = 0;
}

void Renderer::shutdown() {
    destroyPayload();
    glDeleteVertexArrays(1, &m_gridVao);
    glDeleteBuffers(1, &m_gridVbo);
    glDeleteVertexArrays(1, &m_helioVao);
    glDeleteBuffers(1, &m_helioVbo);
    glDeleteVertexArrays(1, &m_ringVao);
    glDeleteBuffers(1, &m_ringVbo);
    glDeleteVertexArrays(1, &m_trailVao);
    glDeleteBuffers(1, &m_trailVbo);
    glDeleteProgram(m_spriteProg);
    glDeleteProgram(m_lineProg);
    glDeleteProgram(m_trailProg);
}

void Renderer::setPayload(const std::vector<HeavyBodyDesc>& heavy, const ParticleSet& particles) {
    destroyPayload();

    m_numHeavy = static_cast<int>(heavy.size());
    std::vector<SpriteAttr> ha(m_numHeavy);
    for (int i = 0; i < m_numHeavy; ++i) {
        const auto& b = heavy[i];
        ha[i] = {packRGBA(b.color[0], b.color[1], b.color[2], b.color[3]), b.size,
                 b.parent == "None" ? 1.0f : 0.0f};
    }
    m_heavyPos.create(m_numHeavy * sizeof(float4));
    glCreateBuffers(1, &m_heavyAttr);
    glNamedBufferStorage(m_heavyAttr, ha.size() * sizeof(SpriteAttr), ha.data(), 0);
    m_heavyVao = makeSpriteVao(m_heavyPos.vbo(), m_heavyAttr);

    m_numParticles = particles.count();
    if (m_numParticles > 0) {
        std::vector<SpriteAttr> pa(m_numParticles);
        for (size_t i = 0; i < m_numParticles; ++i) pa[i] = {particles.rgba[i], particles.size[i], 0.0f};
        m_particlePos.create(m_numParticles * sizeof(float4));
        glCreateBuffers(1, &m_particleAttr);
        glNamedBufferStorage(m_particleAttr, pa.size() * sizeof(SpriteAttr), pa.data(), 0);
        m_particleVao = makeSpriteVao(m_particlePos.vbo(), m_particleAttr);
    }
}

void Renderer::mapForCuda(cudaStream_t stream, float4** heavyOut, float4** particleOut) {
    cudaGraphicsResource* res[2] = {m_heavyPos.resource(), m_particlePos.resource()};
    const int count = m_numParticles > 0 ? 2 : 1;
    cudaCheck(cudaGraphicsMapResources(count, res, stream), "cudaGraphicsMapResources");
    m_mapped = true;
    size_t bytes = 0;
    cudaCheck(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(heavyOut), &bytes, res[0]),
              "cudaGraphicsResourceGetMappedPointer(heavy)");
    *particleOut = nullptr;
    if (count == 2)
        cudaCheck(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(particleOut), &bytes, res[1]),
                  "cudaGraphicsResourceGetMappedPointer(particles)");
}

void Renderer::unmapFromCuda(cudaStream_t stream) {
    if (!m_mapped) return;
    cudaGraphicsResource* res[2] = {m_heavyPos.resource(), m_particlePos.resource()};
    cudaCheck(cudaGraphicsUnmapResources(m_numParticles > 0 ? 2 : 1, res, stream),
              "cudaGraphicsUnmapResources");
    m_mapped = false;
}

void Renderer::draw(const DrawContext& ctx, const RenderSettings& s) {
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    const float* vp = glm::value_ptr(ctx.viewProj);

    // 1. Grid + Heliopause + 3D Boundary Shells (3R/4R Vector Channels, Hill Sphere, Roche Limit)
    glUseProgram(m_lineProg);
    glUniformMatrix4fv(glGetUniformLocation(m_lineProg, "uViewProj"), 1, GL_FALSE, vp);
    glUniform1i(glGetUniformLocation(m_lineProg, "uLog"), s.logScale);
    const GLint uOffset = glGetUniformLocation(m_lineProg, "uOffset");
    const GLint uColor = glGetUniformLocation(m_lineProg, "uColor");
    const GLint uScale = glGetUniformLocation(m_lineProg, "uScale");

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform3fv(uOffset, 1, glm::value_ptr(ctx.worldOffsetAU));
    if (s.showGrid) {
        glUniform1f(uScale, 1.0f);
        glUniform4f(uColor, 0.30f, 0.38f, 0.55f, 0.22f);
        glBindVertexArray(m_gridVao);
        glDrawArrays(GL_LINES, 0, m_gridVerts);
    }
    if (s.showHeliopause && s.heliopauseRadiusAU > 0.0f) {
        glUniform3fv(uOffset, 1, glm::value_ptr(ctx.starCenterRelAU));
        glUniform1f(uScale, s.heliopauseRadiusAU);
        glUniform4f(uColor, 0.15f, 0.55f, 0.95f, 0.22f);
        glBindVertexArray(m_helioVao);
        glDrawArrays(GL_LINES, 0, m_helioVerts);
    }

    for (const auto& sh : ctx.shells) {
        if (sh.radiusAU <= 0.0f) continue;
        glUniform3fv(uOffset, 1, glm::value_ptr(sh.centerRelAU));
        glUniform1f(uScale, sh.radiusAU);
        if (sh.sphereWireframe) {
            glUniform4fv(uColor, 1, glm::value_ptr(sh.color));
            glBindVertexArray(m_helioVao);
            glDrawArrays(GL_LINES, 0, m_helioVerts);
        }
        if (sh.equatorialRing) {
            glm::vec4 rc = sh.color;
            rc.a = std::min(1.0f, sh.color.a * 1.85f);
            glUniform4fv(uColor, 1, glm::value_ptr(rc));
            glBindVertexArray(m_ringVao);
            glDrawArrays(GL_LINES, 0, m_ringVerts);
        }
    }

    // 2. Dynamic 3D Orbit Trails (alpha-faded per-body orbital arcs)
    if (s.showOrbitTrails && m_trailVerts > 0) {
        glUseProgram(m_trailProg);
        glUniformMatrix4fv(glGetUniformLocation(m_trailProg, "uViewProj"), 1, GL_FALSE, vp);
        glUniform1i(glGetUniformLocation(m_trailProg, "uLog"), s.logScale);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(m_trailVao);
        glDrawArrays(GL_LINES, 0, m_trailVerts);
    }

    // 3. Test Particles & Heavy Bodies
    glUseProgram(m_spriteProg);
    glUniformMatrix4fv(glGetUniformLocation(m_spriteProg, "uViewProj"), 1, GL_FALSE, vp);
    glUniform2fv(glGetUniformLocation(m_spriteProg, "uViewport"), 1, glm::value_ptr(ctx.viewportPx));
    glUniform1i(glGetUniformLocation(m_spriteProg, "uLog"), s.logScale);
    const GLint uSize = glGetUniformLocation(m_spriteProg, "uSizeScale");
    const GLint uAlpha = glGetUniformLocation(m_spriteProg, "uAlphaScale");

    if (m_numParticles > 0) {
        glBlendFunc(GL_SRC_ALPHA, s.additiveParticles ? GL_ONE : GL_ONE_MINUS_SRC_ALPHA);
        glUniform1f(uSize, s.particleSizeScale);
        glUniform1f(uAlpha, s.particleAlpha);
        glBindVertexArray(m_particleVao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(m_numParticles));
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform1f(uSize, s.heavySizeScale);
    glUniform1f(uAlpha, 1.0f);
    glBindVertexArray(m_heavyVao);
    if (ctx.heavyDrawOrder.size() == static_cast<size_t>(m_numHeavy)) {
        for (int idx : ctx.heavyDrawOrder)
            glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, static_cast<GLuint>(idx));
    } else {
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, m_numHeavy);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace substrate
