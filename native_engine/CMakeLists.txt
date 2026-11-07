cmake_minimum_required(VERSION 3.24)

# Default to the GPU installed in this machine (RTX 5070 Ti -> sm_120).
# Override with -DCMAKE_CUDA_ARCHITECTURES="86;89;120" for redistributable builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(SubstrateNative LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenGL REQUIRED)

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(glfw glm imgui)

# Dear ImGui has no CMake project of its own.
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

# GLAD 2 loader (OpenGL 4.6 core), generated once and vendored in third_party/glad.
add_library(glad STATIC third_party/glad/src/gl.c)
target_include_directories(glad PUBLIC third_party/glad/include)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(substrate_native
    src/main.cpp
    src/payload.cpp
    src/astro_metrics.cpp
    src/telemetry_ui.cpp
    src/renderer.cpp
    src/substrate_engine.cu)

target_include_directories(substrate_native PRIVATE src)
target_link_libraries(substrate_native PRIVATE
    glad glfw imgui glm::glm CUDA::cudart_static OpenGL::GL)

target_compile_definitions(substrate_native PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN GLFW_INCLUDE_NONE GLM_FORCE_RADIANS)

# Note: no --use_fast_math. The FP64 path must stay IEEE-exact; the mixed path
# uses explicit intrinsics (rsqrtf) where approximation is intended.
target_compile_options(substrate_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8 /W3>)

set_target_properties(substrate_native PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
