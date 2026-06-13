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
