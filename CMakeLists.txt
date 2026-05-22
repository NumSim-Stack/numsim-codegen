cmake_minimum_required(VERSION 3.25)
# 3.25 minimum so that CPM's `SYSTEM YES` parameter actually propagates
# SYSTEM-include semantics to numsim-cas's headers. CMake 3.25 added the
# `SYSTEM` argument to `add_subdirectory()` and `FetchContent_Declare()`;
# CPM forwards it through (see cmake/CPM.cmake around lines 974 and 1163).
# On CMake 3.20-3.24 it would simply not be appended — no warning, no
# effect. CI uses ubuntu-24.04 (CMake 3.28+), so no cost to bumping.

project(NumSim_CodeGen
    VERSION 0.1.0
    DESCRIPTION "Code generation from numsim-cas symbolic expressions for MOOSE constitutive models"
    HOMEPAGE_URL "https://github.com/NumSim-Stack/numsim-codegen"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── Options ────────────────────────────────────────────────────────────────

option(NUMSIM_CODEGEN_BUILD_TESTS "Build numsim-codegen tests" ON)
option(NUMSIM_CODEGEN_BUILD_EXAMPLES "Build numsim-codegen examples" ON)
option(NUMSIM_CODEGEN_USE_LOCAL_CAS "Use a local numsim-cas checkout (set NUMSIM_CAS_SOURCE_DIR)" OFF)

# ─── Dependency manager: CPM ───────────────────────────────────────────────
#
# CPM.cmake is downloaded once at configure time and reused thereafter.
# This is the pattern CPM's own README recommends — avoids vendoring a
# 1300-line third-party file in our tree, keeps bumping to a one-line
# change, and stays reproducible via the SHA256 pin below.
#
# To bump CPM:
#   1. Pick a new release at https://github.com/cpm-cmake/CPM.cmake/releases
#   2. Update CPM_DOWNLOAD_VERSION and CPM_HASH_SHA256 below.
#      (curl -sL <url> | sha256sum  for the hash.)
#
# CPM_SOURCE_CACHE (env var or -D) — share the downloaded CPM.cmake AND
# all CPM-fetched dependencies across projects on the same machine.
# Recommended for local dev:  export CPM_SOURCE_CACHE=$HOME/.cache/CPM

set(CPM_DOWNLOAD_VERSION 0.42.3)
set(CPM_HASH_SHA256
    a609e875fd532b067174250f6abbc3dac22fe2d64869783fb1e80bda1625c844)

if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION
        "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION
        "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION
        "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Verify the cached file's hash on every configure (not just on initial
# download). A truncated download (interrupted ctrl-C), disk corruption,
# or tampering with a shared CPM_SOURCE_CACHE would otherwise be trusted
# silently. file(SHA256) on a ~60 KB file is sub-millisecond, so the cost
# is negligible.
set(_numsim_cpm_cache_ok FALSE)
if(EXISTS "${CPM_DOWNLOAD_LOCATION}")
    file(SHA256 "${CPM_DOWNLOAD_LOCATION}" _numsim_cpm_cached_hash)
    if(_numsim_cpm_cached_hash STREQUAL "${CPM_HASH_SHA256}")
        set(_numsim_cpm_cache_ok TRUE)
    else()
        message(STATUS
            "Cached CPM.cmake at ${CPM_DOWNLOAD_LOCATION} has wrong hash; re-downloading")
        file(REMOVE "${CPM_DOWNLOAD_LOCATION}")
    endif()
endif()

if(NOT _numsim_cpm_cache_ok)
    message(STATUS
        "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION} to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}"
        EXPECTED_HASH SHA256=${CPM_HASH_SHA256})
endif()

include("${CPM_DOWNLOAD_LOCATION}")

# ─── numsim-cas dependency ──────────────────────────────────────────────────

if(NUMSIM_CODEGEN_USE_LOCAL_CAS)
    # Suppress numsim-cas's own tests/examples in the local-checkout path.
    # CPM scopes these via OPTIONS in the else-branch below.
    set(NUMSIM_CAS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NUMSIM_CAS_INSTALL_LIBRARY OFF CACHE BOOL "" FORCE)

    if(NOT NUMSIM_CAS_SOURCE_DIR)
        message(FATAL_ERROR "NUMSIM_CODEGEN_USE_LOCAL_CAS=ON requires NUMSIM_CAS_SOURCE_DIR to be set")
    endif()
    message(STATUS "Using local numsim-cas at ${NUMSIM_CAS_SOURCE_DIR}")
    add_subdirectory(${NUMSIM_CAS_SOURCE_DIR} numsim_cas_build EXCLUDE_FROM_ALL)
else()
    # Pin: bump this SHA deliberately. The previous `GIT_TAG main` made CI
    # non-reproducible — a breaking upstream commit would have flipped CI
    # red with no commit here. See issues #13, #21.
    #
    # GIT_SHALLOW is intentionally absent. CPM auto-disables shallow clones
    # when GIT_TAG is a commit hash (a SHA pin cannot be fetched shallowly
    # without server-side uploadpack.allowReachableSHA1InWant, which GitHub
    # does not guarantee). See `cpm_is_git_tag_commit_hash` in cmake/CPM.cmake.
    #
    # SYSTEM YES marks numsim-cas's include directories as system-includes
    # so its warnings don't surface in our builds (requires CMake 3.25+).
    #
    # EXCLUDE_FROM_ALL keeps numsim-cas's *own internal* targets (its tests,
    # examples, benchmark) off the default build. The NumSim_CAS library
    # itself is still pulled in transitively via the link dependency below
    # (target_link_libraries(numsim_codegen INTERFACE NumSim_CAS)).
    #
    # OPTIONS forwards the four NUMSIM_CAS_* flags as plain (non-cache)
    # variables into the subproject's directory scope; CPM forces
    # CMP0077 NEW, so these override numsim-cas's `option()` defaults
    # without touching the parent project's cache.
    CPMAddPackage(
        NAME numsim_cas
        GITHUB_REPOSITORY NumSim-Stack/numsim-cas
        GIT_TAG d9a9e8db574d4e3e1f56537b7702bd4fbb1fad44
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "NUMSIM_CAS_BUILD_TESTS OFF"
            "NUMSIM_CAS_BUILD_EXAMPLES OFF"
            "NUMSIM_CAS_BUILD_BENCHMARK OFF"
            "NUMSIM_CAS_INSTALL_LIBRARY OFF"
    )
endif()

# ─── Library (header-only) ──────────────────────────────────────────────────

add_library(numsim_codegen INTERFACE)
add_library(numsim::codegen ALIAS numsim_codegen)

target_include_directories(numsim_codegen
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(numsim_codegen INTERFACE NumSim_CAS)
target_compile_features(numsim_codegen INTERFACE cxx_std_23)

# ─── Tests ──────────────────────────────────────────────────────────────────

if(NUMSIM_CODEGEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Examples ───────────────────────────────────────────────────────────────

if(NUMSIM_CODEGEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ─── Install ────────────────────────────────────────────────────────────────
# Header-only install. CMake package config / export targets are deferred
# until numsim-cas itself exports installable targets (currently it does not
# expose NumSim_CAS in an export set, which blocks install(EXPORT) here).

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
