# Reuse GoogleTest from numsim-cas's FetchContent (it pulls it in via tests/).
# If unavailable, fall back to fetching directly.

include(FetchContent)

if(NOT TARGET GTest::gtest)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(numsim_codegen_tests
    main.cpp
    CodegenContextTest.cpp
    ScalarCodeEmitTest.cpp
    TensorCodeEmitTest.cpp
    SpectralCodeEmitTest.cpp
    SpectralMirrorTest.cpp
    RecipeTest.cpp
    MooseTargetTest.cpp
    RoleFlexibilityTest.cpp
    RecipeRegistrySmokeTest.cpp
    PassFrameworkTest.cpp
    StateVariableTest.cpp
    TimeIntegrationTest.cpp
    LocalJacobianTest.cpp
    IntegrationTest.cpp
    LocalNewtonTest.cpp
    AlgorithmicTangentTest.cpp
    LinearAlgebraEmitterTest.cpp
    NumSimMaterialTargetTest.cpp
    TargetFactoryTest.cpp
)

target_link_libraries(numsim_codegen_tests
    PRIVATE
        numsim::codegen
        numsim_codegen_warnings
        GTest::gtest
)
# The registry smoke test consumes examples/recipe_registry.h directly.
target_include_directories(numsim_codegen_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/examples)

include(GoogleTest)
gtest_discover_tests(numsim_codegen_tests)

# ─── Compile-check test ────────────────────────────────────────────────────
#
# Round-trip test: build a small recipe, run it to generate a header,
# compile a driver that #include's the generated header, run the driver.
# Catches codegen bugs that produce syntactically invalid C++ or
# semantically wrong math — the regular unit tests only check strings.

add_executable(generate_compile_check_recipe
    generated/generate_compile_check_recipe.cpp)
target_link_libraries(generate_compile_check_recipe
    PRIVATE numsim::codegen numsim_codegen_warnings)

set(GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/CompileCheck.h)
set(GENERATED_HARDENING_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/HardeningCheck.h)
set(GENERATED_NEWTON_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/NewtonCheck.h)
set(GENERATED_AUTOCAT_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/AutocatalyticCheck.h)
# Phase 3b-2b (issue #35): coupled-Newton compile-check header. Its generated
# code uses Eigen for the dense solve (see below).
set(GENERATED_COUPLED_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/CoupledCheck.h)
# CAS f3e799e catch-up (#275/#285): piecewise tensor-output compile-check —
# exercises the tensor_if_then_else materialized-ternary emission vs tmech.
set(GENERATED_PIECEWISE_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/PiecewiseCheck.h)
# Review #87 t2s coverage: the tensor_to_scalar_if_then_else sibling, reachable
# only as a subterm (lifted into a tensor output via with_tensor_mul).
set(GENERATED_PIECEWISE_T2S_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/PiecewiseT2sCheck.h)
# Verification spine (#90): nonlinear-stress consistent-tangent FD compile-check.
set(GENERATED_TANGENT_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/TangentCheck.h)

add_custom_command(
    OUTPUT ${GENERATED_HEADER} ${GENERATED_HARDENING_HEADER}
           ${GENERATED_NEWTON_HEADER} ${GENERATED_AUTOCAT_HEADER}
           ${GENERATED_COUPLED_HEADER} ${GENERATED_PIECEWISE_HEADER}
           ${GENERATED_PIECEWISE_T2S_HEADER} ${GENERATED_TANGENT_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_compile_check_recipe>
            ${GENERATED_HEADER} ${GENERATED_HARDENING_HEADER}
            ${GENERATED_NEWTON_HEADER} ${GENERATED_AUTOCAT_HEADER}
            ${GENERATED_COUPLED_HEADER} ${GENERATED_PIECEWISE_HEADER}
            ${GENERATED_PIECEWISE_T2S_HEADER} ${GENERATED_TANGENT_HEADER}
    DEPENDS generate_compile_check_recipe
    COMMENT "Generating compile-check headers via numsim-codegen"
    VERBATIM)

# Phase 3b-2b: the coupled-Newton generated code includes <Eigen/Dense>. The CI
# runners don't ship libeigen3-dev, so fetch Eigen (header-only — DOWNLOAD_ONLY,
# no Eigen CMake configure) and expose just its include dir to the driver.
CPMAddPackage(
    NAME eigen3
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY YES)

add_executable(compile_check_driver
    generated/compile_check_driver.cpp
    ${GENERATED_HEADER}
    ${GENERATED_HARDENING_HEADER}
    ${GENERATED_NEWTON_HEADER}
    ${GENERATED_AUTOCAT_HEADER}
    ${GENERATED_COUPLED_HEADER}
    ${GENERATED_PIECEWISE_HEADER}
    ${GENERATED_PIECEWISE_T2S_HEADER}
    ${GENERATED_TANGENT_HEADER})
target_include_directories(compile_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${eigen3_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/support)
target_link_libraries(compile_check_driver
    PRIVATE numsim::codegen numsim_codegen_warnings_light
            GTest::gtest GTest::gtest_main)

gtest_discover_tests(compile_check_driver)

# ─── Spectral lowering end-to-end gate (#105, slice 5) ───────────────────────
# Generate log(A) and its tangent d log(A)/dA as tmech-only C++, then compile
# and run it, checking both against the CAS evaluator — including at eigenvalue
# coalescence (b=I), where the divided-difference guard must stay finite. This
# is the compile-and-run gate behind the symbolic spectral tangent.
add_executable(generate_spectral_check
    generated/generate_spectral_check.cpp)
target_link_libraries(generate_spectral_check
    PRIVATE numsim::codegen numsim_codegen_warnings)

set(GENERATED_SPECTRAL_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/SpectralCheck.h)
add_custom_command(
    OUTPUT ${GENERATED_SPECTRAL_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_spectral_check> ${GENERATED_SPECTRAL_HEADER}
    DEPENDS generate_spectral_check
    COMMENT "Generating spectral e2e header via numsim-codegen"
    VERBATIM)

add_executable(spectral_check_driver
    generated/spectral_check_driver.cpp
    ${GENERATED_SPECTRAL_HEADER})
target_include_directories(spectral_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(spectral_check_driver
    PRIVATE numsim::codegen numsim_codegen_warnings_light
            GTest::gtest GTest::gtest_main)

gtest_discover_tests(spectral_check_driver)

# ─── MOOSE Hencky material end-to-end gate (#105 follow-up) ──────────────────
# Emit the Hencky hyperelastic material (spectral stress + exact consistent
# tangent) via the StandaloneCxx target, compile it, and FD-verify the tangent
# (distinct + near-identity) plus finiteness at coalescence. Proves the spectral
# lowering composes into a real generated constitutive material.
add_executable(generate_hencky_check
    generated/generate_hencky_check.cpp)
target_include_directories(generate_hencky_check
    PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_link_libraries(generate_hencky_check
    PRIVATE numsim::codegen numsim_codegen_warnings)

set(GENERATED_HENCKY_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/HenckyCheck.h)
add_custom_command(
    OUTPUT ${GENERATED_HENCKY_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_hencky_check> ${GENERATED_HENCKY_HEADER}
    DEPENDS generate_hencky_check
    COMMENT "Generating Hencky material header via numsim-codegen"
    VERBATIM)

add_executable(hencky_check_driver
    generated/hencky_check_driver.cpp
    ${GENERATED_HENCKY_HEADER})
target_include_directories(hencky_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(hencky_check_driver
    PRIVATE numsim::codegen numsim_codegen_warnings_light
            GTest::gtest GTest::gtest_main)

gtest_discover_tests(hencky_check_driver)

# ─── First-PK SVK (P, F) end-to-end gate ─────────────────────────────────────
# Emit the first-Piola SVK example (P = F·S + consistent tangent dP/dF) via the
# StandaloneCxx target, compile it, and FD-verify dP/dF in general (non-symmetric)
# directions. The strain enters through C = FᵀF, so this is the end-to-end
# regression for the tmech-v1.1.1 lazy-operand fix.
add_executable(generate_fpsvk_check
    generated/generate_fpsvk_check.cpp)
target_include_directories(generate_fpsvk_check
    PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_link_libraries(generate_fpsvk_check
    PRIVATE numsim::codegen numsim_codegen_warnings)

set(GENERATED_FPSVK_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/generated/FpSvkCheck.h)
add_custom_command(
    OUTPUT ${GENERATED_FPSVK_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_fpsvk_check> ${GENERATED_FPSVK_HEADER}
    DEPENDS generate_fpsvk_check
    COMMENT "Generating first-PK SVK material header via numsim-codegen"
    VERBATIM)

add_executable(fpsvk_check_driver
    generated/fpsvk_check_driver.cpp
    ${GENERATED_FPSVK_HEADER})
target_include_directories(fpsvk_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(fpsvk_check_driver
    PRIVATE numsim::codegen numsim_codegen_warnings_light
            GTest::gtest GTest::gtest_main)

gtest_discover_tests(fpsvk_check_driver)

# ─── St. Venant–Kirchhoff (S, E) end-to-end gate ─────────────────────────────
# Emit the shipped St. Venant–Kirchhoff example (2nd-PK stress S + consistent
# tangent dS/dE) via the StandaloneCxx target, compile it, and FD-verify the
# tangent in every symmetric direction. Demonstrates selecting the (2nd-PK,
# Green–Lagrange) conjugate pair by making E the differentiation input — the
# tangent MOOSE's total-Lagrangian kernels consume. Uses only diff(tensor,
# tensor) (no spectral machinery), so it lives on the primary path.
add_executable(generate_svk_check
    generated/generate_svk_check.cpp)
target_include_directories(generate_svk_check
    PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_link_libraries(generate_svk_check
    PRIVATE numsim::codegen numsim_codegen_warnings)

set(GENERATED_SVK_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/SvkCheck.h)
add_custom_command(
    OUTPUT ${GENERATED_SVK_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_svk_check> ${GENERATED_SVK_HEADER}
    DEPENDS generate_svk_check
    COMMENT "Generating St. Venant–Kirchhoff material header via numsim-codegen"
    VERBATIM)

add_executable(svk_check_driver
    generated/svk_check_driver.cpp
    ${GENERATED_SVK_HEADER})
target_include_directories(svk_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(svk_check_driver
    PRIVATE numsim::codegen numsim_codegen_warnings_light
            GTest::gtest GTest::gtest_main)

gtest_discover_tests(svk_check_driver)

# ─── numsim-materials end-to-end gate (Phase B) ──────────────────────────────
#
# The string-asserting NumSimMaterialTargetTest proves the emitted TEXT matches
# the rk_integrator contract; it CANNOT prove the emitted code compiles and runs
# against the real solver. This gate does: it generates NumSimMaterialTarget rate
# materials (linear dα/dt=K·α and nonlinear dα/dt=K·α²), compiles them against
# the REAL numsim-materials + numsim-core headers, runs them through the real
# rk_integrator, and verifies both convergence (closed-form) and the emitted
# rate / rate_derivative VALUES — closing the dominant gap behind #89.
#
# Source resolution (in order):
#   1. a local checkout at NUMSIM_MATERIALS_DIR / NUMSIM_CORE_DIR if it has the
#      headers (fast path for dev — no download);
#   2. else, if NUMSIM_CODEGEN_FETCH_MATERIALS=ON, CPM-fetch them (DOWNLOAD_ONLY:
#      headers only, so numsim-materials' own CMake / FetchContents never run and
#      cannot collide with our tmech/Eigen). CI sets this ON.
#   3. else SKIP.
#
# IMPORTANT (pins): the codegen↔solver contract this gate verifies is NOT on the
# siblings' `main` yet — it lives on the feature branches below. The SHAs are
# immutable so CI stays reproducible, but they MUST be bumped (and ideally
# re-pointed at a `main` tag) once the contract lands upstream. Tracked in #89.
set(NUMSIM_MATERIALS_DIR "${CMAKE_SOURCE_DIR}/../numsim-materials"
    CACHE PATH "numsim-materials checkout (header-only) for the Phase B e2e gate")
set(NUMSIM_CORE_DIR "${CMAKE_SOURCE_DIR}/../numsim-core"
    CACHE PATH "numsim-core checkout (header-only) for the Phase B e2e gate")
option(NUMSIM_CODEGEN_FETCH_MATERIALS
    "CPM-fetch numsim-materials/numsim-core for the Phase B e2e gate when no \
local checkout is present (headers only)" OFF)
# Feature-branch pins — see the IMPORTANT note above. Single source of truth.
set(_ncg_materials_pin "aaa6614c1aa9eebaa62f8d07a6306bb9defe47ab") # feature/drucker-prager
set(_ncg_core_pin      "c462d9caf09cba8722076d68da7c7346e0c0623e") # feature/missing-param-names

set(_ncg_materials_inc "")
set(_ncg_core_inc "")
if(EXISTS "${NUMSIM_MATERIALS_DIR}/include/numsim-materials/solvers/rk_integrator.h"
   AND EXISTS "${NUMSIM_CORE_DIR}/include/numsim-core/input_parameter_controller.h")
    set(_ncg_materials_inc "${NUMSIM_MATERIALS_DIR}/include")
    set(_ncg_core_inc "${NUMSIM_CORE_DIR}/include")
    message(STATUS "Phase B e2e gate: using local checkouts "
                   "(${NUMSIM_MATERIALS_DIR})")
elseif(NUMSIM_CODEGEN_FETCH_MATERIALS)
    # DOWNLOAD_ONLY: fetch sources without running their CMake (header-only use).
    CPMAddPackage(NAME numsim_materials_e2e
        GITHUB_REPOSITORY NumSim-Stack/numsim-materials
        GIT_TAG ${_ncg_materials_pin} DOWNLOAD_ONLY YES)
    CPMAddPackage(NAME numsim_core_e2e
        GITHUB_REPOSITORY NumSim-Stack/numsim-core
        GIT_TAG ${_ncg_core_pin} DOWNLOAD_ONLY YES)
    set(_ncg_materials_inc "${numsim_materials_e2e_SOURCE_DIR}/include")
    set(_ncg_core_inc "${numsim_core_e2e_SOURCE_DIR}/include")
    message(STATUS "Phase B e2e gate: CPM-fetched siblings (pinned)")
endif()

if(_ncg_materials_inc AND _ncg_core_inc)
    add_executable(generate_numsim_material_check
        generated/generate_numsim_material_check.cpp)
    target_link_libraries(generate_numsim_material_check
        PRIVATE numsim::codegen numsim_codegen_warnings)

    set(GENERATED_LINEAR_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/LinearHardening.h)
    set(GENERATED_NONLINEAR_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/NonlinearDecay.h)
    set(GENERATED_VISCO_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/Viscoelastic.h)
    set(GENERATED_RETURNMAP_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/ReturnMap.h)
    set(GENERATED_RETURNMAP_CUBIC_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/ReturnMapCubic.h)
    set(GENERATED_J2_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/J2Return.h)
    set(GENERATED_J2_VOCE_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/J2Voce.h)
    set(GENERATED_J2_SWIFT_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/J2Swift.h)
    add_custom_command(
        OUTPUT ${GENERATED_LINEAR_HEADER} ${GENERATED_NONLINEAR_HEADER}
               ${GENERATED_VISCO_HEADER} ${GENERATED_RETURNMAP_HEADER}
               ${GENERATED_RETURNMAP_CUBIC_HEADER} ${GENERATED_J2_HEADER}
               ${GENERATED_J2_VOCE_HEADER} ${GENERATED_J2_SWIFT_HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND $<TARGET_FILE:generate_numsim_material_check>
                ${GENERATED_LINEAR_HEADER} ${GENERATED_NONLINEAR_HEADER}
                ${GENERATED_VISCO_HEADER} ${GENERATED_RETURNMAP_HEADER}
                ${GENERATED_RETURNMAP_CUBIC_HEADER} ${GENERATED_J2_HEADER}
                ${GENERATED_J2_VOCE_HEADER} ${GENERATED_J2_SWIFT_HEADER}
        DEPENDS generate_numsim_material_check
        COMMENT "Generating numsim-materials rate materials via NumSimMaterialTarget"
        VERBATIM)

    add_executable(numsim_material_check_driver
        generated/numsim_material_check_driver.cpp
        ${GENERATED_LINEAR_HEADER} ${GENERATED_NONLINEAR_HEADER}
        ${GENERATED_VISCO_HEADER} ${GENERATED_RETURNMAP_HEADER}
        ${GENERATED_RETURNMAP_CUBIC_HEADER} ${GENERATED_J2_HEADER}
        ${GENERATED_J2_VOCE_HEADER} ${GENERATED_J2_SWIFT_HEADER})
    # numsim-materials, numsim-core, Eigen and tmech are header-only and included
    # as SYSTEM so their warnings stay silent under the first-party -Werror gate.
    # tmech (via cas's CPM) is needed for the tensor-stress material/test.
    target_include_directories(numsim_material_check_driver SYSTEM PRIVATE
        ${_ncg_materials_inc}
        ${_ncg_core_inc}
        ${eigen3_SOURCE_DIR}
        ${tmech_SOURCE_DIR}/include)
    target_include_directories(numsim_material_check_driver PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
    # The generated material is first-party emitted code — gate it -Wall -Wextra.
    target_link_libraries(numsim_material_check_driver
        PRIVATE numsim_codegen_warnings_light GTest::gtest GTest::gtest_main)
    gtest_discover_tests(numsim_material_check_driver)
    message(STATUS "Phase B e2e gate ENABLED")
else()
    message(STATUS "Phase B e2e gate SKIPPED: no local checkout at "
                   "${NUMSIM_MATERIALS_DIR} / ${NUMSIM_CORE_DIR} and "
                   "NUMSIM_CODEGEN_FETCH_MATERIALS=OFF")
endif()

# ─── Header-purity guard ───────────────────────────────────────────────────
#
# Public headers under include/ must never #include heavy template libraries
# that should stay PRIVATE to the STATIC backend implementations. This is
# the structural invariant that lets the inja + nlohmann/json adoption in
# #20 happen without exploding consuming-TU compile times. Adding new
# disallowed names here is cheaper than a separate CI workflow and runs in
# every matrix configuration via `ctest`.
add_test(
    NAME HeaderPurity
    COMMAND ${CMAKE_COMMAND}
        "-DSEARCH_ROOT=${CMAKE_SOURCE_DIR}/include"
        "-DDISALLOWED_PATTERN=inja|nlohmann/json"
        -P ${CMAKE_SOURCE_DIR}/cmake/check_header_purity.cmake
)
# The script exits non-zero on violation via message(FATAL_ERROR) — that's
# what fails the test. The FAIL_REGULAR_EXPRESSION below is belt-and-braces
# so even a future non-fatal `message(WARNING ...)` still trips ctest.
set_tests_properties(HeaderPurity PROPERTIES
    FAIL_REGULAR_EXPRESSION "HeaderPurity violation")
