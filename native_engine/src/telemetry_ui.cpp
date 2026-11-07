#include "telemetry_ui.h"
#include "substrate_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace substrate {

namespace {

glm::vec3 logCompressHUD(glm::vec3 p) {
    const float r = glm::length(p);
    if (r < 1e-6f) return p;
    return (p / r) * (std::log(1.0f + r * 10.0f) * 0.43429448190325182f);
}

struct Rect2D {
    float x0, y0, x1, y1;
    bool overlaps(const Rect2D& o, float pad = 3.0f) const {
        return !(x1 + pad < o.x0 || x0 - pad > o.x1 || y1 + pad < o.y0 || y0 - pad > o.y1);
    }
    bool contains(float x, float y) const {
        return x >= x0 && x <= x1 && y >= y0 && y <= y1;
    }
};

float overlapArea(const Rect2D& a, const Rect2D& b, float pad = 3.0f) {
    const float ix = std::max(0.0f, std::min(a.x1 + pad, b.x1 + pad) - std::max(a.x0 - pad, b.x0 - pad));
    const float iy = std::max(0.0f, std::min(a.y1 + pad, b.y1 + pad) - std::max(a.y0 - pad, b.y0 - pad));
    return ix * iy;
}

int priorityTier(int idx, int targetIdx, BodyCategory cat) {
    if (idx == targetIdx) return 0;
    switch (cat) {
        case BodyCategory::Star:        return 1;
        case BodyCategory::Planet:      return 2;
        case BodyCategory::DwarfPlanet: return 3;
        case BodyCategory::Moon:        return 4;
    }
    return 4;
}

void drawVerticalResonanceMarker(const ImVec2& plotMin, const ImVec2& plotMax,
                                 float valAU, float minAU, float maxAU,
                                 const char* label, ImU32 color, float yOffsetPx = 3.0f) {
    const float t = (valAU - minAU) / (maxAU - minAU);
    if (t < 0.0f || t > 1.0f) return;
    const float x = plotMin.x + t * (plotMax.x - plotMin.x);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddLine(ImVec2(x, plotMin.y), ImVec2(x, plotMax.y), color, 1.5f);
    dl->AddText(ImVec2(x + 3.0f, plotMin.y + yOffsetPx), color, label);
}

} // namespace

// ---------------------------------------------------------------------------
// ConservationHistory
// ---------------------------------------------------------------------------
void ConservationHistory::reset(double E0, double L0) {
    relEnergyDrift.clear();
    relAngMomDrift.clear();
    baseEnergy = E0;
    baseAngMom = L0;
    initialized = true;
}

void ConservationHistory::push(double E, double L) {
    if (!initialized) {
        reset(E, L);
    }
    const float eDrift = ( std::abs(baseEnergy) > 1e-30 )
                             ? static_cast<float>(std::abs((E - baseEnergy) / baseEnergy))
                             : 0.0f;
    const float lDrift = ( std::abs(baseAngMom) > 1e-30 )
                             ? static_cast<float>(std::abs((L - baseAngMom) / baseAngMom))
                             : 0.0f;

    if (static_cast<int>(relEnergyDrift.size()) >= CAPACITY) {
        relEnergyDrift.erase(relEnergyDrift.begin());
        relAngMomDrift.erase(relAngMomDrift.begin());
    }
    relEnergyDrift.push_back(eDrift);
    relAngMomDrift.push_back(lDrift);
}

// ---------------------------------------------------------------------------
// Smart 2D Viewport HUD & Label Deconfliction
// ---------------------------------------------------------------------------
void drawSmartViewportHUD(ImDrawList* bgDraw,
                          const ImVec2& displaySize,
                          const glm::mat4& viewProj,
                          const std::vector<glm::vec3>& heavyRelAU,
                          const std::vector<HeavyBodyDesc>& heavyDescs,
                          const std::vector<int>& parentIdx,
                          const SubstrateEngine& engine,
                          const RenderSettings& renderSettings,
                          bool showLabels,
                          float camDistanceAU,
                          float camAzimuthDeg,
                          float camElevationDeg,
                          int& targetIdx,
                          bool& /*lockTarget*/,
                          glm::vec3& panOffsetAU) {
    const int numHeavy = static_cast<int>(heavyDescs.size());

    // 1. Top-left HUD Telemetry Badge
    const Rect2D hudBadgeRect{16.0f, 16.0f, 385.0f, 88.0f};
    {
        bgDraw->AddRectFilled(ImVec2(hudBadgeRect.x0, hudBadgeRect.y0),
                              ImVec2(hudBadgeRect.x1, hudBadgeRect.y1),
                              IM_COL32(10, 15, 24, 210), 5.0f);
        bgDraw->AddRect(ImVec2(hudBadgeRect.x0, hudBadgeRect.y0),
                        ImVec2(hudBadgeRect.x1, hudBadgeRect.y1),
                        IM_COL32(0, 190, 255, 135), 5.0f);
        bgDraw->AddText(ImVec2(26.0f, 24.0f), IM_COL32(0, 225, 255, 255),
                        "PIPELINE: NATIVE C++20 / CUDA -> GL 4.6 VBO");
        char line2[128];
        std::snprintf(line2, sizeof(line2), "PAYLOAD: %zu PARTICLES + %d HEAVY BODIES",
                      engine.numParticles(), engine.numHeavy());
        bgDraw->AddText(ImVec2(26.0f, 44.0f), IM_COL32(190, 210, 230, 255), line2);
        char line3[128];
        if (camDistanceAU < 0.01f) {
            std::snprintf(line3, sizeof(line3), "CAM: %.0f km | AZ: %.0f deg | EL: %.0f deg",
                          camDistanceAU * (AU / 1000.0), camAzimuthDeg, camElevationDeg);
        } else {
            std::snprintf(line3, sizeof(line3), "CAM: %.3f AU | AZ: %.0f deg | EL: %.0f deg",
                          camDistanceAU, camAzimuthDeg, camElevationDeg);
        }
        bgDraw->AddText(ImVec2(26.0f, 62.0f), IM_COL32(150, 175, 200, 255), line3);
    }

    // 2. Project all heavy bodies to 2D screen coordinates
    struct ScreenBody {
        bool onScreen = false;
        float sx = 0.0f;
        float sy = 0.0f;
        float depthW = 0.0f;
    };
    std::vector<ScreenBody> proj(numHeavy);
    for (int i = 0; i < numHeavy; ++i) {
        glm::vec3 p = heavyRelAU[i];
        if (renderSettings.logScale) p = logCompressHUD(p);
        const glm::vec4 clip = viewProj * glm::vec4(p, 1.0f);
        if (clip.w <= 0.0f) continue;
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (ndc.x < -1.08f || ndc.x > 1.08f || ndc.y < -1.08f || ndc.y > 1.08f) continue;
        proj[i].onScreen = true;
        proj[i].sx = (ndc.x * 0.5f + 0.5f) * displaySize.x;
        proj[i].sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * displaySize.y;
        proj[i].depthW = clip.w;
    }

    // Allow clicking directly on a sprite disc when mouse is in the 3D viewport
    const ImGuiIO& io = ImGui::GetIO();
    const bool leftClickViewport = !io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    int clickedBody = -1;
    float bestSpriteClickDist = 14.0f;
    if (leftClickViewport) {
        for (int i = 0; i < numHeavy; ++i) {
            if (!proj[i].onScreen) continue;
            const float d = std::hypot(io.MousePos.x - proj[i].sx, io.MousePos.y - proj[i].sy);
            const float hitR = std::max(10.0f, heavyDescs[i].size * renderSettings.heavySizeScale * 0.65f + 4.0f);
            if (d <= hitR && d < bestSpriteClickDist) {
                bestSpriteClickDist = d;
                clickedBody = i;
            }
        }
    }

    // Always draw target corner reticle if target is on screen
    if (targetIdx >= 0 && targetIdx < numHeavy && proj[targetIdx].onScreen) {
        const float sx = proj[targetIdx].sx;
        const float sy = proj[targetIdx].sy;
        const float r = 17.0f;
        const float c = 6.0f;
        const ImU32 colCyan = (engine.density()[targetIdx] >= RHO_GC)
                                  ? IM_COL32(0, 235, 255, 240)
                                  : IM_COL32(255, 145, 35, 245);
        bgDraw->AddLine(ImVec2(sx - r, sy - r), ImVec2(sx - r + c, sy - r), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx - r, sy - r), ImVec2(sx - r, sy - r + c), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx + r, sy - r), ImVec2(sx + r - c, sy - r), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx + r, sy - r), ImVec2(sx + r, sy - r + c), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx - r, sy + r), ImVec2(sx - r + c, sy + r), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx - r, sy + r), ImVec2(sx - r, sy + r - c), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx + r, sy + r), ImVec2(sx + r - c, sy + r), colCyan, 2.0f);
        bgDraw->AddLine(ImVec2(sx + r, sy + r), ImVec2(sx + r, sy + r - c), colCyan, 2.0f);
    }

    // 3. Determine which moons fold into their parent planet's badge vs. expand individually
    const int activeSystemRoot = (targetIdx >= 0 && heavyDescs[targetIdx].category == BodyCategory::Moon)
                                     ? parentIdx[targetIdx]
                                     : targetIdx;
    constexpr float MOON_COLLAPSE_PX = 42.0f;

    std::vector<int> collapsedMoons(numHeavy, 0);
    std::vector<bool> suppressMoonLabel(numHeavy, false);

    for (int i = 0; i < numHeavy; ++i) {
        if (!proj[i].onScreen) continue;
        if (heavyDescs[i].category != BodyCategory::Moon) continue;
        const int p = parentIdx[i];
        if (p < 0 || !proj[p].onScreen) continue;

        const bool inspectingThisSystem = (p == activeSystemRoot) || (i == targetIdx);
        const float screenDist = std::hypot(proj[i].sx - proj[p].sx, proj[i].sy - proj[p].sy);

        if (!inspectingThisSystem && screenDist < MOON_COLLAPSE_PX) {
            suppressMoonLabel[i] = true;
            collapsedMoons[p] += 1;
        }
    }

    // 4. Sort visible labels by priority tier so planets NEVER lose to moons
    std::vector<int> labelOrder;
    labelOrder.reserve(numHeavy);
    for (int i = 0; i < numHeavy; ++i) {
        if (!proj[i].onScreen) continue;
        if (i != targetIdx && !showLabels) continue;
        if (suppressMoonLabel[i]) continue;
        labelOrder.push_back(i);
    }

    std::sort(labelOrder.begin(), labelOrder.end(), [&](int a, int b) {
        const int ta = priorityTier(a, targetIdx, heavyDescs[a].category);
        const int tb = priorityTier(b, targetIdx, heavyDescs[b].category);
        if (ta != tb) return ta < tb;
        return proj[a].depthW < proj[b].depthW;
    });

    // 5. Place labels with 2D AABB deconfliction & radial spider leader lines
    std::vector<Rect2D> placedRects;
    placedRects.push_back(hudBadgeRect);

    // Also keep labels off the glowing sprite centres of all visible planets/stars
    std::vector<Rect2D> spriteDiscs;
    for (int i = 0; i < numHeavy; ++i) {
        if (!proj[i].onScreen) continue;
        if (heavyDescs[i].category == BodyCategory::Moon && suppressMoonLabel[i]) continue;
        const float sr = std::max(5.0f, heavyDescs[i].size * renderSettings.heavySizeScale * 0.5f + 2.0f);
        spriteDiscs.push_back({proj[i].sx - sr, proj[i].sy - sr, proj[i].sx + sr, proj[i].sy + sr});
    }

    constexpr float TWO_PI_F = 6.2831853f;

    for (int idx : labelOrder) {
        const float sx = proj[idx].sx;
        const float sy = proj[idx].sy;
        const bool isTarget = (idx == targetIdx);
        const bool decompiled = (engine.density()[idx] < RHO_GC);

        char textBuf[160];
        if (isTarget) {
            if (decompiled) {
                std::snprintf(textBuf, sizeof(textBuf), "TARGET: %s [I-PHASE / Theta=0]",
                              heavyDescs[idx].name.c_str());
            } else if (collapsedMoons[idx] > 0) {
                std::snprintf(textBuf, sizeof(textBuf), "TARGET: %s (+%d)",
                              heavyDescs[idx].name.c_str(), collapsedMoons[idx]);
            } else {
                std::snprintf(textBuf, sizeof(textBuf), "TARGET: %s", heavyDescs[idx].name.c_str());
            }
        } else {
            if (decompiled && collapsedMoons[idx] > 0) {
                std::snprintf(textBuf, sizeof(textBuf), "%s (+%d) [Theta=0]",
                              heavyDescs[idx].name.c_str(), collapsedMoons[idx]);
            } else if (decompiled) {
                std::snprintf(textBuf, sizeof(textBuf), "%s [Theta=0]", heavyDescs[idx].name.c_str());
            } else if (collapsedMoons[idx] > 0) {
                std::snprintf(textBuf, sizeof(textBuf), "%s (+%d)",
                              heavyDescs[idx].name.c_str(), collapsedMoons[idx]);
            } else {
                std::snprintf(textBuf, sizeof(textBuf), "%s", heavyDescs[idx].name.c_str());
            }
        }

        const ImVec2 tsz = ImGui::CalcTextSize(textBuf);
        const float padX = isTarget ? 6.0f : 5.0f;
        const float padY = isTarget ? 3.5f : 2.5f;
        const float pillW = tsz.x + 2.0f * padX;
        const float pillH = tsz.y + 2.0f * padY;

        // Candidate slots: default right-side pill, then radial spider spokes
        const float baseOff = isTarget ? 23.0f : std::max(9.0f, heavyDescs[idx].size * 0.55f + 6.0f);

        auto scoreCandidate = [&](const Rect2D& r) -> float {
            float penalty = 0.0f;
            if (r.x0 < 8.0f || r.y0 < 8.0f || r.x1 > displaySize.x - 8.0f || r.y1 > displaySize.y - 8.0f)
                penalty += 50000.0f;
            for (const auto& pr : placedRects) {
                if (r.overlaps(pr, 3.0f)) penalty += 1000.0f + overlapArea(r, pr, 3.0f) * 10.0f;
            }
            for (size_t s = 0; s < spriteDiscs.size(); ++s) {
                if (r.overlaps(spriteDiscs[s], 1.0f)) penalty += 400.0f + overlapArea(r, spriteDiscs[s], 1.0f) * 5.0f;
            }
            return penalty;
        };

        Rect2D bestRect{sx + baseOff, sy - pillH * 0.5f, sx + baseOff + pillW, sy + pillH * 0.5f};
        float bestScore = scoreCandidate(bestRect);
        bool usedSpiderCallout = false;

        if (bestScore > 0.0f) {
            // Bias radial spoke search away from parent planet if this is a moon
            float startAngle = -0.35f;
            const int p = parentIdx[idx];
            if (p >= 0 && proj[p].onScreen) {
                const float dxP = sx - proj[p].sx;
                const float dyP = sy - proj[p].sy;
                if (dxP * dxP + dyP * dyP > 4.0f) {
                    startAngle = std::atan2(dyP, dxP);
                } else {
                    startAngle = (idx * 1.6180339f) * TWO_PI_F;  // golden-angle spread when coincident!
                }
            }

            const float radii[] = {24.0f, 40.0f, 58.0f, 78.0f, 102.0f};
            constexpr int NUM_SPOKES = 16;
            for (float rad : radii) {
                bool foundClean = false;
                for (int k = 0; k < NUM_SPOKES; ++k) {
                    // Alternate +k / -k around the outward radial direction
                    const int step = (k + 1) / 2;
                    const float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                    const float ang = startAngle + sign * step * (TWO_PI_F / NUM_SPOKES);
                    const float cx = sx + std::cos(ang) * rad;
                    const float cy = sy + std::sin(ang) * rad;

                    // Anchor pill on left edge if spoke goes right, right edge if spoke goes left
                    const float x0 = (std::cos(ang) >= -0.15f) ? cx : (cx - pillW);
                    const float y0 = cy - pillH * 0.5f;
                    const Rect2D cand{x0, y0, x0 + pillW, y0 + pillH};
                    const float sc = scoreCandidate(cand);
                    if (sc < bestScore) {
                        bestScore = sc;
                        bestRect = cand;
                        usedSpiderCallout = true;
                        if (sc == 0.0f) {
                            foundClean = true;
                            break;
                        }
                    }
                }
                if (foundClean) break;
            }
        }

        placedRects.push_back(bestRect);

        // Draw leader line when spider-displaced
        if (usedSpiderCallout) {
            const float anchorX = std::clamp(sx, bestRect.x0, bestRect.x1);
            const float anchorY = std::clamp(sy, bestRect.y0, bestRect.y1);
            const ImU32 lineCol = isTarget ? IM_COL32(0, 225, 255, 190)
                                           : (decompiled ? IM_COL32(255, 145, 40, 175)
                                                         : IM_COL32(150, 185, 220, 145));
            bgDraw->AddLine(ImVec2(sx, sy), ImVec2(anchorX, anchorY), lineCol, 1.25f);
            bgDraw->AddCircleFilled(ImVec2(sx, sy), 2.2f, lineCol);
        }

        // Pill styling by category & GC compilation state
        ImU32 bgCol   = IM_COL32(12, 18, 28, 188);
        ImU32 brdCol  = IM_COL32(70, 95, 125, 110);
        ImU32 txtCol  = IM_COL32(222, 234, 248, 230);

        if (decompiled) {
            bgCol  = IM_COL32(58, 28, 8, 215);
            brdCol = IM_COL32(255, 140, 35, 210);
            txtCol = IM_COL32(255, 190, 110, 250);
        } else if (isTarget) {
            bgCol  = IM_COL32(0, 42, 64, 225);
            brdCol = IM_COL32(0, 215, 255, 200);
            txtCol = IM_COL32(0, 238, 255, 255);
        } else if (heavyDescs[idx].category == BodyCategory::Star) {
            bgCol  = IM_COL32(42, 34, 10, 205);
            brdCol = IM_COL32(255, 210, 70, 165);
            txtCol = IM_COL32(255, 232, 135, 245);
        } else if (heavyDescs[idx].category == BodyCategory::Moon) {
            bgCol  = IM_COL32(14, 20, 30, 172);
            txtCol = IM_COL32(180, 202, 224, 215);
        }

        bgDraw->AddRectFilled(ImVec2(bestRect.x0, bestRect.y0),
                              ImVec2(bestRect.x1, bestRect.y1), bgCol, 3.5f);
        bgDraw->AddRect(ImVec2(bestRect.x0, bestRect.y0),
                        ImVec2(bestRect.x1, bestRect.y1), brdCol, 3.5f);
        bgDraw->AddText(ImVec2(bestRect.x0 + padX, bestRect.y0 + padY), txtCol, textBuf);

        if (leftClickViewport && bestRect.contains(io.MousePos.x, io.MousePos.y)) {
            clickedBody = idx;
        }
    }

    if (clickedBody >= 0 && clickedBody != targetIdx) {
        targetIdx = clickedBody;
        panOffsetAU = glm::vec3(0.0f);
    }
}

// ---------------------------------------------------------------------------
// Separate Toggleable Floating Window: Astrophysics & Substrate Telemetry
// ---------------------------------------------------------------------------
void drawTelemetryWindow(TelemetryUIState& ui,
                         const std::vector<HeavyBodyDesc>& heavyDescs,
                         const std::vector<int>& parentIdx,
                         const HeavySnapshot& snap,
                         SubstrateEngine& engine,
                         const StepParams& stepParams,
                         RenderSettings& renderSettings,
                         int& targetIdx,
                         bool& lockTarget,
                         float& camDistanceAU,
                         glm::vec3& panOffsetAU,
                         const std::function<void()>& onResetSolarSystem) {
    if (!ui.showWindow) return;

    ImGui::SetNextWindowPos(ImVec2(16.0f, 100.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(530.0f, 690.0f), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Astrophysics & Substrate Telemetry", &ui.showWindow)) {
        ImGui::End();
        return;
    }

    const int numHeavy = static_cast<int>(heavyDescs.size());
    const int pIdx = (targetIdx >= 0 && targetIdx < numHeavy) ? parentIdx[targetIdx] : -1;
    const KeplerElements kep = computeKeplerElements(targetIdx, pIdx, snap, engine);
    const DomainRadii dom = computeDomainRadii(targetIdx, pIdx, heavyDescs, kep, engine);
    const SubstrateBodyAudit audit = computeSubstrateAudit(
        targetIdx, pIdx, snap, kep, engine, stepParams.relativistic);

    // Compact persistent target selector bar at top of telemetry window
    ImGui::PushItemWidth(210.0f);
    if (ImGui::BeginCombo("Active Body", heavyDescs[targetIdx].name.c_str())) {
        for (int i = 0; i < numHeavy; ++i) {
            const bool sel = (i == targetIdx);
            if (ImGui::Selectable(heavyDescs[i].name.c_str(), sel)) {
                targetIdx = i;
                if (lockTarget) panOffsetAU = glm::vec3(0.0f);
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (audit.thetaGC > 0.5) {
        ImGui::TextColored(ImVec4(0.15f, 1.0f, 0.55f, 1.0f), "[COMPILED Gamma-Knot | Theta = 1]");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.15f, 1.0f), "[UNCOMPILED I-Phase | Theta = 0]");
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("SubstrateTelemetryTabs")) {
        // ===================================================================
        // TAB 1: SUBSTRATE LOGISTICS HARDWARE AUDIT
        // ===================================================================
        if (ImGui::BeginTabItem("1. Substrate Audit")) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "A. Discretized River Metric & Spatial Pixel Purge");
            if (ImGui::BeginTable("RiverTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("At Body Surface (R)");
                ImGui::TableSetupColumn(pIdx >= 0 ? "At Orbit (Parent Field)" : "At Barycenter");
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Deletion Accel g_Omega");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.4e m/s^2", audit.gSurface_ms2);
                ImGui::TableSetColumnIndex(2);
                if (pIdx >= 0) ImGui::Text("%.4e m/s^2", audit.gParentAtOrbit_ms2);
                else ImGui::TextDisabled("Root Star");

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Coordinate Inflow v_inflow");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f km/s", audit.vInflowSurface_ms / 1000.0);
                ImGui::TableSetColumnIndex(2);
                if (pIdx >= 0) ImGui::Text("%.3f km/s", audit.vInflowOrbit_ms / 1000.0);
                else ImGui::TextDisabled("N/A");

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Vacuum Deletion dV/dt");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.3e m^3/s", audit.volDeletedSurface_m3s);
                ImGui::TableSetColumnIndex(2);
                if (pIdx >= 0) ImGui::Text("%.3e m^3/s", audit.pixelPurgeParentAtOrbit_s * V_NODE);
                else ImGui::TextDisabled("N/A");

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Pixel Purge Rate dN_px/dt");
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(ImVec4(0.3f, 0.95f, 1.0f, 1.0f), "%.3e px/s", audit.pixelPurgeSurface_s);
                ImGui::TableSetColumnIndex(2);
                if (pIdx >= 0) ImGui::Text("%.3e px/s", audit.pixelPurgeParentAtOrbit_s);
                else ImGui::TextDisabled("N/A");

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "B. Bandwidth Governor & Atomic Clock Throttling");
            ImGui::Text("Integer Node Load (N = M/m_p) : %.4e nodes", audit.nodesN);
            ImGui::Text("Kernel Lock Horizon (R_EH)    : %.4e m (Safety R/R_req = %.2e)",
                        audit.schwarzschildRadius_m, audit.safetyFactor);
            ImGui::Text("Surface Bandwidth Omega(R)    : %.4e  (Limit <= 1.0)", audit.omegaSurface);
            ImGui::Text("Surface Clock Lag (1 - u_O)   : %.4f ppb  (-%.3f us/day vs. void)",
                        audit.clockLagSurface_ppb, audit.clockLagSurface_usDay);
            if (pIdx >= 0) {
                ImGui::Text("Orbital Clock Lag (g_O + v_O) : %.4f ppb  (-%.3f us/day vs. void)",
                            audit.clockLagOrbit_ppb, audit.clockLagOrbit_usDay);
                ImGui::TextColored(ImVec4(0.95f, 0.82f, 0.35f, 1.0f),
                                   "1PN Periapsis Precession      : %.3f arcsec / Julian century",
                                   audit.precessionArcsecCentury);
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "C. Macro-Scale Quantization & Vector-Channel Geometry");
            ImGui::Text("Collective Compton Floor      : %.4e m", audit.lambdaComptonBody_m);
            ImGui::Text("Top-Down Quantum Radius       : %.2f km  (Physical: %.2f km)",
                        audit.quantumTopDownRadius_m / 1000.0, audit.radiusM / 1000.0);
            ImGui::TextColored(ImVec4(0.0f, 0.90f, 1.0f, 1.0f),
                               "3R Directional Mass/Axial     : %.1f km  (Plasmapause delta)",
                               dom.vector3R_m / 1000.0);
            ImGui::TextColored(ImVec4(0.80f, 0.45f, 1.0f, 1.0f),
                               "4R Unspooled 4-Channel Bound  : %.1f km  (Van Allen / Charge analog)",
                               dom.vector4R_m / 1000.0);
            if (ImGui::Button("Lock & Zoom to 3R / 4R Vector Channels")) {
                lockTarget = true;
                renderSettings.logScale = false;
                renderSettings.showVectorChannels = true;
                panOffsetAU = glm::vec3(0.0f);
                camDistanceAU = std::max(1.0e-5f, static_cast<float>((14.0 * dom.core1R_m) / AU));
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "D. Zero-Sum 120-Cell Cover Ledger & Coarse-Graining");
            float probeAUf = static_cast<float>(ui.coarseProbeAU);
            if (ImGui::SliderFloat("Probe Sphere R (AU)", &probeAUf, 1.0f, 4000.0f, "%.1f AU",
                                   ImGuiSliderFlags_Logarithmic)) {
                ui.coarseProbeAU = probeAUf;
            }
            const CoarseGrainAudit cg = computeCoarseGrainAudit(ui.coarseProbeAU, snap, engine);
            ImGui::Text("Enclosed Mean <rho_N>_V       : %.3e nodes/m^3 (%s)",
                        cg.meanRhoNodes, cg.exceedsGCCeiling ? ">= 4.0 [ACTIVE]" : "< 4.0 [VOID]");
            ImGui::Text("Grav. Field Density u_g(R)    : %.3e J/m^3  (Sigma_0 = %.4e J/m^3)",
                        cg.gravFieldEnergyDensity_Jm3, SIGMA_0);
            ImGui::Text("Topological Drag Constant C_d : (sqrt(5)-2)/500 = %.6e  (P_d = %.4f pPa)",
                        C_DELTA, P_DRAG_FLOOR * 1.0e12);
            ImGui::Text("120-Cell Ledger Partitions    : Void Om_m=%.6f / Om_L=%.6f | Global Om_m=%.6f",
                        OMEGA_M_VOID, OMEGA_LAMBDA_VOID, OMEGA_M_GLOBAL);
            ImGui::Text("H0 & S8 Scale Split           : H0 = %.2f / %.2f / %.2f | S8 = %.4f -> %.4f",
                        H_LAMBDA_KMS_MPC, H_GLOBAL_KMS_MPC, H_LOCAL_KMS_MPC, S8_VOID, S8_GLOBAL);

            ImGui::Spacing();
            if (ImGui::TreeNodeEx("Dodecahedral Astrosphere & Exoplanet Disk Comparator",
                                  ImGuiTreeNodeFlags_DefaultOpen)) {
                struct StarPreset {
                    const char* label;
                    double massSolar;
                    double radiusSolar;
                    const char* noteCliff50;
                    const char* noteChan36;
                    const char* noteEdge30;
                    const char* noteShock27;
                    const char* noteAstro20;
                    const char* summary;
                };
                static constexpr StarPreset kPresets[] = {
                    {"Solar System (G2V, 1.00 M_sun)", 1.000, 1.000,
                     "Kuiper Cliff (48-50 AU)", "Scattered Disk / Eris a=67.9 AU",
                     "Sedna q=76.2 / V2 TS 83.7 AU", "Mean V1/V2 Shock (88.9 AU)",
                     "Heliopause (V2 119.0 - V1 121.6 AU)",
                     "Exact cancellation: lambda_bar_p * sqrt(N_sun) = 48.47 AU -> Kuiper Cliff R_g/50 = 48.62 AU."},
                    {"Epsilon Eridani (K2V, 0.82 M_sun)", 0.820, 0.735,
                     "Inner Dust Clearing", "ALMA Belt Inner Edge (~64 AU)",
                     "ALMA Narrow Belt Peak (70 AU)", "ALMA Belt Outer Edge (~76 AU)",
                     "Outer Astropause Cutoff",
                     "ALMA resolves a narrow ring at 64-76 AU (peak 70 AU), bracketed by R_g/36 -> R_g/27 and centered near R_g/30."},
                    {"HR 8799 (F0V, 1.52 M_sun)", 1.520, 1.502,
                     "Outermost Giant Planet b (68 AU)", "Planet b Chaotic Clearing Zone",
                     "Inner Cavity Transition", "Pre-Belt Stagnation Step",
                     "ALMA Cold Belt Inner Edge (145+-12 AU)",
                     "Resolves HR 8799's 90-145 AU cavity puzzle without an unseen 5th giant planet: R_g/20 = 149.8 AU."},
                    {"Fomalhaut (A3V, 1.92 M_sun)", 1.920, 1.842,
                     "Inner Warm Cavity", "JWST Mid-Belt (83-104 AU, mid 93.5 AU)",
                     "JWST Outer Gap (104-133 AU)", "Main KBA Inner Step (133-136 AU)",
                     "Outer KBA & Halo Cutoff (~165 AU)",
                     "JWST/MIRI & ALMA nested belts lock onto R_TS,1 (91.9 AU), R_g/36 (93.6 AU), and R_g/27 -> R_g/20."},
                    {"Vega (A0V, 2.135 M_sun)", 2.135, 2.726,
                     "JWST KBA Inner Edge (74-80 AU)", "Inner Planetesimal Core",
                     "Mid Planetesimal Disk", "Outer Planetesimal Core",
                     "JWST/ALMA Outer Disk Edge (~170 AU)",
                     "Oblate rapid rotator (R_eq = 2.726 R_sun): R_TS,1/1.35 = 59.0 AU matches JWST's ~60 AU gap."},
                    {"Custom Stellar System", 1.000, 1.000,
                     "k=50 Planetesimal Cliff", "36-Channel Routing Shell",
                     "30-Edge Structural Strut", "Supersonic Stagnation Shock",
                     "20-Vertex Astropause",
                     "Universal hardware scaling: R_g(N_*) proportional to sqrt(N_*), independent of empirical ISM tuning."}
                };

                ImGui::Combo("Benchmark Star", &ui.astrospherePresetIdx,
                             "Solar System (1.00 M_sun)\0"
                             "Epsilon Eridani (0.82 M_sun)\0"
                             "HR 8799 (1.52 M_sun)\0"
                             "Fomalhaut (1.92 M_sun)\0"
                             "Vega (2.135 M_sun)\0"
                             "Custom Star (M_*, R_*)\0");

                const int pSel = std::clamp(ui.astrospherePresetIdx, 0, 5);
                double mSolar = kPresets[pSel].massSolar;
                double rSolar = kPresets[pSel].radiusSolar;
                if (pSel == 5) {
                    ImGui::SliderFloat("Star Mass (M / M_sun)", &ui.customStarMassSolar, 0.08f, 20.0f, "%.3f",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat("Star Radius (R / R_sun)", &ui.customStarRadiusSolar, 0.10f, 20.0f, "%.3f",
                                       ImGuiSliderFlags_Logarithmic);
                    mSolar = ui.customStarMassSolar;
                    rSolar = ui.customStarRadiusSolar;
                }

                constexpr double M_SUN_REF = 1.9885e30;
                const double nSunRef = M_SUN_REF / M_P;
                const double nStar = (pSel == 0 && engine.numHeavy() > 0)
                                         ? ((engine.density()[0] >= RHO_GC) ? engine.nodes()[0] : 0.0)
                                         : ((mSolar * M_SUN_REF) / M_P);

                const double rComptonSqrtN = (LAMBDA_BAR_P * std::sqrt(std::max(nStar, 0.0))) / AU;
                const double rGravAU       = gravitationalFloorMetersFromNodes(nStar) / AU;
                const double rGravSunRefAU = gravitationalFloorMetersFromNodes(nSunRef) / AU;

                const double rCliff50 = rGravAU / KAPPA;
                const double rChan36  = rGravAU / C_CHANNELS;
                const double rEdge30  = rGravAU / E_DODEC;
                const double rShock27 = rGravAU / SHOCK_DIV;
                const double rAstro20 = rGravAU / V_DODEC;

                const double inflowFactor = std::sqrt(mSolar / std::max(rSolar, 1.0e-4));
                const double rShockSurf1  = (rGravSunRefAU / SHOCK_DIV) * inflowFactor;
                const double pWindPred1AU_nPa = P_DRAG_FLOOR * (rShock27 * rShock27) * 1.0e9;

                ImGui::Text("Compton-Node Scale (lambda_p * sqrt(N_*)) : %.2f AU  |  Surface Shock R_TS,1: %.1f AU",
                            rComptonSqrtN, rShockSurf1);
                ImGui::Text("Predicted 1 AU Wind Pressure (P_d * R_TS^2): %.3f nPa  (ACE/Wind Solar Mean: ~2.27 nPa)",
                            pWindPred1AU_nPa);

                if (ImGui::BeginTable("DodecShellTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Dodecahedral Shell", ImGuiTableColumnFlags_WidthFixed, 145.0f);
                    ImGui::TableSetupColumn("Derived Radius", ImGuiTableColumnFlags_WidthFixed, 95.0f);
                    ImGui::TableSetupColumn("ALMA / JWST / Voyager Match");
                    ImGui::TableHeadersRow();

                    auto row = [](const char* tier, double valAU, const char* note, const ImVec4& col) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::TextColored(col, "%s", tier);
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f AU", valAU);
                        ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(note);
                    };

                    row("k=50 Cliff (R_g/50)",     rCliff50, kPresets[pSel].noteCliff50, ImVec4(0.20f, 0.96f, 0.82f, 1.0f));
                    row("C=36 Channels (R_g/36)",  rChan36,  kPresets[pSel].noteChan36,  ImVec4(0.40f, 0.80f, 1.00f, 1.0f));
                    row("E=30 Struts (R_g/30)",    rEdge30,  kPresets[pSel].noteEdge30,  ImVec4(0.72f, 0.62f, 1.00f, 1.0f));
                    row("Shock Div 27 (R_g/27)",   rShock27, kPresets[pSel].noteShock27, ImVec4(1.00f, 0.72f, 0.28f, 1.0f));
                    row("V=20 Astropause (R_g/20)",rAstro20, kPresets[pSel].noteAstro20, ImVec4(0.35f, 0.72f, 1.00f, 1.0f));
                    row("Grav. Floor (R_g)",       rGravAU,  "u_g(R_g) == Sigma_{0,Ledger}", ImVec4(1.00f, 0.88f, 0.42f, 1.0f));

                    ImGui::EndTable();
                }

                ImGui::TextWrapped("%s", kPresets[pSel].summary);

                if (ImGui::Button("Frame Solar Dodecahedral Shells in 3D")) {
                    lockTarget = false;
                    renderSettings.logScale = false;
                    renderSettings.showDodecahedralShells = true;
                    renderSettings.showHeliopause = true;
                    panOffsetAU = glm::vec3(0.0f);
                    camDistanceAU = 195.0f;
                }

                if (ImGui::TreeNode("Dynamic Solar Wind Ram-Pressure Test (P_SW vs. P_delta)")) {
                    ImGui::SliderFloat("Solar Wind n(1 AU) [cm^-3]", &ui.swDensity1AU_cm3, 2.0f, 12.0f, "%.1f");
                    ImGui::SliderFloat("Solar Wind v_SW [km/s]", &ui.swSpeed_kms, 300.0f, 650.0f, "%.0f");
                    const double rTS_sw = SubstrateEngine::heliopauseTopologicalDragAU(
                        ui.swDensity1AU_cm3, ui.swSpeed_kms, 1.0);
                    const double rHP_sw = 1.35 * rTS_sw;
                    ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.70f, 1.0f),
                                       "Ram Stagnation: R_TS = %.1f AU  |  R_HP = %.1f AU  (R_g/20 = %.2f AU)",
                                       rTS_sw, rHP_sw, engine.heliopauseRadiusAU());
                    if (ImGui::SmallButton("Apply Wind R_HP to 3D Sphere")) {
                        renderSettings.customHeliopauseOverride = true;
                        renderSettings.heliopauseRadiusAU = static_cast<float>(rHP_sw);
                        renderSettings.showHeliopause = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Restore Derived R_g/20 Shell")) {
                        renderSettings.customHeliopauseOverride = false;
                        renderSettings.heliopauseRadiusAU = static_cast<float>(engine.heliopauseRadiusAU());
                    }
                    ImGui::TreePop();
                }

                ImGui::TreePop();
            }

            ImGui::EndTabItem();
        }

        // ===================================================================
        // TAB 2: CLASSICAL ASTRODYNAMICS & CONSERVATION
        // ===================================================================
        if (ImGui::BeginTabItem("2. Astrodynamics & Conservation")) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Osculating 3D Keplerian Elements (Parent: %s)",
                               pIdx >= 0 ? heavyDescs[pIdx].name.c_str() : "None / Root Barycenter");

            if (pIdx < 0) {
                ImGui::TextDisabled("Root star selected — select any planet, dwarf planet, or moon for orbital elements.");
            } else if (!kep.valid) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.25f, 1.0f),
                                   "Parent spatial deletion is inactive (Theta = 0) — Two-body orbit uncompiled!");
            } else if (!kep.bound) {
                ImGui::TextColored(ImVec4(1.0f, 0.40f, 0.25f, 1.0f),
                                   "UNBOUND / HYPERBOLIC TRAJECTORY (e = %.4f, eps = +%.3e J/kg)",
                                   kep.e, kep.specificEnergy);
                ImGui::Text("Instantaneous Distance r : %.4f AU (%.0f km)", kep.r_m / AU, kep.r_m / 1000.0);
                ImGui::Text("Relative Speed v         : %.3f km/s", kep.v_ms / 1000.0);
            } else {
                const bool isMoon = (heavyDescs[targetIdx].category == BodyCategory::Moon);
                if (isMoon) {
                    ImGui::Text("Semi-Major Axis (a)      : %.1f km  (%.6f AU)", kep.a_m / 1000.0, kep.a_AU);
                    ImGui::Text("Periapsis / Apoapsis     : %.1f km  /  %.1f km",
                                kep.periapsis_m / 1000.0, kep.apoapsis_m / 1000.0);
                } else {
                    ImGui::Text("Semi-Major Axis (a)      : %.5f AU  (%.3e km)", kep.a_AU, kep.a_m / 1000.0);
                    ImGui::Text("Perihelion / Aphelion    : %.4f AU  /  %.4f AU",
                                kep.periapsis_m / AU, kep.apoapsis_m / AU);
                }
                ImGui::Text("Eccentricity (e)         : %.5f", kep.e);
                ImGui::Text("Inclination (i)          : %.3f deg%s",
                            kep.inc_deg, kep.inc_deg > 90.0 ? "  [RETROGRADE]" : "");
                ImGui::Text("Ascending Node (Omega)   : %.2f deg   |  Arg. Periapsis (omega): %.2f deg",
                            kep.raan_deg, kep.argPeri_deg);
                ImGui::Text("True Anomaly (nu)        : %.2f deg", kep.trueAnom_deg);
                ImGui::Text("Orbital Period (T)       : %.2f days  (%.4f Julian years)",
                            kep.period_days, kep.period_days / 365.25);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f), "Gravitational & Tidal Stability Domains");
            if (dom.hillRadius_m > 0.0) {
                ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.25f, 1.0f),
                                   "Hill Sphere (r_H)        : %.0f km  (%.5f AU)",
                                   dom.hillRadius_m / 1000.0, dom.hillRadius_m / AU);
                ImGui::Text("Laplace SOI (r_SOI)      : %.0f km  (%.5f AU)",
                            dom.soiRadius_m / 1000.0, dom.soiRadius_m / AU);
            }
            if (heavyDescs[targetIdx].category == BodyCategory::Moon && pIdx >= 0) {
                ImGui::Text("Host (%s) Fluid Roche : %.0f km  (Rigid: %.0f km)",
                            heavyDescs[pIdx].name.c_str(),
                            dom.rocheFluid_m / 1000.0, dom.rocheRigid_m / 1000.0);
                if (kep.r_m > 0.0 && dom.rocheFluid_m > 0.0) {
                    const double ratio = kep.r_m / dom.rocheFluid_m;
                    if (ratio < 1.0) {
                        ImGui::TextColored(ImVec4(1.0f, 0.32f, 0.35f, 1.0f),
                                           "Tidal Status             : INSIDE FLUID ROCHE LIMIT (r/d_R = %.3f)!", ratio);
                    } else {
                        ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.55f, 1.0f),
                                           "Tidal Status             : Tidally Stable (r/d_R = %.2f)", ratio);
                    }
                }
            } else {
                ImGui::Text("Fluid Roche Limit (d_R)  : %.0f km  (Rigid: %.0f km)",
                            dom.rocheFluid_m / 1000.0, dom.rocheRigid_m / 1000.0);
            }
            if (dom.hillRadius_m > 0.0 && ImGui::Button("Lock & Zoom to Hill / Moon System")) {
                lockTarget = true;
                renderSettings.logScale = false;
                renderSettings.showHillAndRoche = true;
                panOffsetAU = glm::vec3(0.0f);
                camDistanceAU = std::max(5.0e-5f, static_cast<float>((2.4 * dom.hillRadius_m) / AU));
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Symplectic Velocity-Verlet Conservation Audit");
            const double curE = engine.newtonianEnergy(snap);
            const double curL = engine.angularMomentumMag(snap);
            const double vBary = engine.barycenterDriftSpeed(snap);
            const float latestE = ui.conservation.relEnergyDrift.empty()
                                      ? 0.0f
                                      : ui.conservation.relEnergyDrift.back();
            const float latestL = ui.conservation.relAngMomDrift.empty()
                                      ? 0.0f
                                      : ui.conservation.relAngMomDrift.back();

            ImGui::Text("Rel. Energy Drift |(E-E0)/E0| : %.3e", latestE);
            if (!ui.conservation.relEnergyDrift.empty()) {
                ImGui::PlotLines("##EnergySpark", ui.conservation.relEnergyDrift.data(),
                                 static_cast<int>(ui.conservation.relEnergyDrift.size()),
                                 0, "Energy Drift History", 0.0f,
                                 std::max(latestE * 2.5f, 1e-11f), ImVec2(-1.0f, 55.0f));
            }
            ImGui::Text("Rel. AngMom Drift |(L-L0)/L0| : %.3e   |  Barycenter |V_cm|: %.2e m/s",
                        latestL, vBary);
            if (!ui.conservation.relAngMomDrift.empty()) {
                ImGui::PlotLines("##AngMomSpark", ui.conservation.relAngMomDrift.data(),
                                 static_cast<int>(ui.conservation.relAngMomDrift.size()),
                                 0, "Angular Momentum Drift History", 0.0f,
                                 std::max(latestL * 2.5f, 1e-13f), ImVec2(-1.0f, 50.0f));
            }
            if (ImGui::SmallButton("Re-Zero Conservation Baseline")) {
                ui.conservation.reset(curE, curL);
            }

            ImGui::EndTabItem();
        }

        // ===================================================================
        // TAB 3: BELT STATISTICAL MECHANICS & RESONANCES (Zero-Readback GPU)
        // ===================================================================
        if (ImGui::BeginTabItem("3. Belt Resonances (GPU)")) {
            // Compute live semi-major axes of Jupiter (idx 8) & Neptune (idx 16) for dynamic resonance markers
            const KeplerElements jupK = computeKeplerElements(8, 0, snap, engine);
            const KeplerElements nepK = computeKeplerElements(16, 0, snap, engine);
            const double aJup = (jupK.bound && jupK.a_AU > 0.1) ? jupK.a_AU : 5.2044;
            const double aNep = (nepK.bound && nepK.a_AU > 1.0) ? nepK.a_AU : 30.110;

            if (ui.beltStatsFrameCounter++ % 15 == 0) {
                engine.computeBeltStats(ui.beltStats, aJup);
            }

            const BeltStats& bs = ui.beltStats;
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Main Asteroid Belt Semi-Major Axis (2.00 - 3.50 AU) & Kirkwood Resonances");
            ImGui::PlotHistogram("##MainBeltHist", bs.mainBeltHist, BeltStats::MAIN_BINS,
                                 0, nullptr, 0.0f, FLT_MAX, ImVec2(-1.0f, 115.0f));
            {
                const ImVec2 pMin = ImGui::GetItemRectMin();
                const ImVec2 pMax = ImGui::GetItemRectMax();
                const ImU32 colRes = IM_COL32(255, 95, 95, 220);
                // Live Kepler 3rd-law resonance locations a_{p:q} = a_Jup * (q / p)^(2/3)
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aJup * std::pow(1.0 / 3.0, 2.0 / 3.0)), 2.00f, 3.50f, "3:1", colRes);
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aJup * std::pow(2.0 / 5.0, 2.0 / 3.0)), 2.00f, 3.50f, "5:2", colRes);
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aJup * std::pow(3.0 / 7.0, 2.0 / 3.0)), 2.00f, 3.50f, "7:3", colRes);
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aJup * std::pow(1.0 / 2.0, 2.0 / 3.0)), 2.00f, 3.50f, "2:1", colRes);
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Kuiper Belt & Scattered Disk (30.0 - 72.0 AU) — Dodecahedral Cutoffs");
            ImGui::PlotHistogram("##KuiperHist", bs.kuiperBeltHist, BeltStats::KUIPER_BINS,
                                 0, nullptr, 0.0f, FLT_MAX, ImVec2(-1.0f, 115.0f));
            {
                const ImVec2 pMin = ImGui::GetItemRectMin();
                const ImVec2 pMax = ImGui::GetItemRectMax();
                const float kMin = static_cast<float>(BeltStats::KUIPER_MIN_AU);
                const float kMax = static_cast<float>(BeltStats::KUIPER_MAX_AU);
                const ImU32 colNep = IM_COL32(90, 220, 255, 225);
                // Live exterior Neptune resonances a_{p:q} = a_Nep * (p / q)^(2/3)
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aNep * std::pow(3.0 / 2.0, 2.0 / 3.0)),
                                            kMin, kMax, "3:2", colNep, 3.0f);
                drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(aNep * std::pow(2.0 / 1.0, 2.0 / 3.0)),
                                            kMin, kMax, "2:1", colNep, 3.0f);
                const double rGravAU = engine.gravitationalFloorAU();
                if (rGravAU > 0.0) {
                    drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(rGravAU / KAPPA),
                                                kMin, kMax, "k=50 Cliff", IM_COL32(50, 255, 195, 240), 20.0f);
                    drawVerticalResonanceMarker(pMin, pMax, static_cast<float>(rGravAU / C_CHANNELS),
                                                kMin, kMax, "C=36 Shell", IM_COL32(255, 205, 85, 240), 3.0f);
                }
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Orbital Eccentricity Distribution (0.00 - 0.50)");
            ImGui::PlotHistogram("##EccHist", bs.eccHist, BeltStats::ECC_BINS,
                                 0, nullptr, 0.0f, FLT_MAX, ImVec2(-1.0f, 75.0f));

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "Jupiter Tisserand Invariant (T_J) Dynamical Regimes");
            const double tot = std::max<double>(1.0, static_cast<double>(bs.boundCount + bs.unboundCount));
            ImGui::Text("Bound Asteroidal (T_J > 3.0)       : %u  (%.1f%%)",
                        bs.asteroidalCount, 100.0 * bs.asteroidalCount / tot);
            ImGui::Text("Jupiter-Family Cometary (2 < T_J<=3): %u  (%.1f%%)",
                        bs.jfcCount, 100.0 * bs.jfcCount / tot);
            ImGui::Text("Halley / Damocloid (T_J <= 2.0)    : %u  (%.1f%%)",
                        bs.halleyCount, 100.0 * bs.halleyCount / tot);
            if (bs.unboundCount > 0) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.25f, 1.0f),
                                   "Unbound / Escaping (eps >= 0)      : %u  (%.1f%%)",
                                   bs.unboundCount, 100.0 * bs.unboundCount / tot);
            } else {
                ImGui::Text("Unbound / Escaping (eps >= 0)      : 0  (0.0%%)");
            }
            ImGui::Text("Ensemble Mean Eccentricity <e>     : %.4f   |  Mean Inclination <i>: %.2f deg",
                        bs.meanEcc, bs.meanIncDeg);

            ImGui::EndTabItem();
        }

        // ===================================================================
        // TAB 4: INTERACTIVE GARBAGE COLLECTION SANDBOX (Paper 3: Theta Filter)
        // ===================================================================
        if (ImGui::BeginTabItem("4. GC Sandbox (Theta Filter)")) {
            if (ui.lastSandboxTarget != targetIdx) {
                ui.lastSandboxTarget = targetIdx;
                ui.sandboxLog10Rho = static_cast<float>(
                    std::log10(std::max(engine.density()[targetIdx], 1e-2)));
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.0f, 0.88f, 1.0f, 1.0f),
                               "He^4 4-Channel Compilation Filter: Theta(rho_N - 4.0 nodes/m^3)");
            ImGui::TextWrapped(
                "In Substrate Logistics, spatial address deletion (g_Omega) activates strictly when "
                "localized node density reaches the 4-channel tetrahedral saturation ceiling "
                "rho_{N,GC} = Sigma_{0,Ledger} / (m_p c^2) = 4.0 nodes/m^3 (1 He^4 / m^3). "
                "Below 4.0 nodes/m^3, matter decompiles into passive I-phase gas (Theta = 0, g_Omega = 0).");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Selected Target       : %s", heavyDescs[targetIdx].name.c_str());
            ImGui::Text("Physical Base Density : %.3e nodes/m^3", engine.baseDensity(targetIdx));
            ImGui::Text("Active Engine Density : %.3e nodes/m^3", engine.density()[targetIdx]);

            bool overridden = engine.isDensityOverridden(targetIdx);
            if (ImGui::Checkbox("Enable Live Node-Density Override on Target", &overridden)) {
                const double rho = std::pow(10.0, static_cast<double>(ui.sandboxLog10Rho));
                engine.setDensityOverride(targetIdx, overridden, rho, stepParams.relativistic);
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }

            if (ImGui::SliderFloat("log10(rho_N) [nodes/m^3]", &ui.sandboxLog10Rho, -2.0f, 32.0f, "%.2f")) {
                const double rho = std::pow(10.0, static_cast<double>(ui.sandboxLog10Rho));
                engine.setDensityOverride(targetIdx, true, rho, stepParams.relativistic);
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }

            // Visual threshold gauge
            {
                const float frac = std::clamp((ui.sandboxLog10Rho - (-2.0f)) / 34.0f, 0.0f, 1.0f);
                const float threshFrac = (std::log10(4.0f) - (-2.0f)) / 34.0f;
                const ImVec2 p0 = ImGui::GetCursorScreenPos();
                const float barW = ImGui::GetContentRegionAvail().x;
                const float barH = 18.0f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddRectFilled(p0, ImVec2(p0.x + barW, p0.y + barH), IM_COL32(18, 24, 34, 255), 3.0f);
                const ImU32 fillCol = (engine.density()[targetIdx] >= RHO_GC)
                                          ? IM_COL32(30, 210, 120, 210)
                                          : IM_COL32(255, 130, 30, 225);
                dl->AddRectFilled(p0, ImVec2(p0.x + barW * frac, p0.y + barH), fillCol, 3.0f);
                const float tx = p0.x + barW * threshFrac;
                dl->AddLine(ImVec2(tx, p0.y - 2.0f), ImVec2(tx, p0.y + barH + 2.0f),
                            IM_COL32(255, 70, 70, 255), 2.5f);
                dl->AddText(ImVec2(tx + 5.0f, p0.y + 1.0f), IM_COL32(255, 180, 180, 255),
                            "GC Ceiling = 4.0 nodes/m^3");
                ImGui::Dummy(ImVec2(barW, barH + 6.0f));
            }

            // One-click live physics presets
            if (ImGui::Button("Decompile -> I-Phase (1.2 nodes/m^3, Theta = 0)")) {
                ui.sandboxLog10Rho = std::log10(1.2f);
                engine.setDensityOverride(targetIdx, true, 1.2, stepParams.relativistic);
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }
            ImGui::SameLine();
            if (ImGui::Button("Compile Edge (4.5 nodes/m^3, Theta = 1)")) {
                ui.sandboxLog10Rho = std::log10(4.5f);
                engine.setDensityOverride(targetIdx, true, 4.5, stepParams.relativistic);
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }

            if (ImGui::Button("Restore Physical Gamma-Knot Density")) {
                engine.setDensityOverride(targetIdx, false, engine.baseDensity(targetIdx), stepParams.relativistic);
                ui.sandboxLog10Rho = static_cast<float>(std::log10(engine.baseDensity(targetIdx)));
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset All Overrides")) {
                engine.resetDensityOverrides(stepParams.relativistic);
                ui.sandboxLog10Rho = static_cast<float>(std::log10(engine.baseDensity(targetIdx)));
                ui.conservation.reset(engine.newtonianEnergy(snap), engine.angularMomentumMag(snap));
            }

            ImGui::Spacing();
            if (ImGui::Button("Reload Pristine Solar System Orbits", ImVec2(-1.0f, 30.0f))) {
                if (onResetSolarSystem) onResetSolarSystem();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.75f, 0.85f, 0.95f, 1.0f),
                               "Tip: Select Jupiter, Lock Camera to it, then click 'Decompile -> I-Phase' "
                               "to watch Io, Europa, Ganymede, and Callisto unbind tangentially in real time!");

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

} // namespace substrate

