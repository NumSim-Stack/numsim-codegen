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
    RecipeTest.cpp
    MooseTargetTest.cpp
    RoleFlexibilityTest.cpp
)

target_link_libraries(numsim_codegen_tests
    PRIVATE
        numsim::codegen
        GTest::gtest
)

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
target_link_libraries(generate_compile_check_recipe PRIVATE numsim::codegen)

set(GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/CompileCheck.h)

add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND $<TARGET_FILE:generate_compile_check_recipe> ${GENERATED_HEADER}
    DEPENDS generate_compile_check_recipe
    COMMENT "Generating CompileCheck.h via numsim-codegen"
    VERBATIM)

add_executable(compile_check_driver
    generated/compile_check_driver.cpp
    ${GENERATED_HEADER})
target_include_directories(compile_check_driver
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(compile_check_driver
    PRIVATE numsim::codegen GTest::gtest GTest::gtest_main)

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
        -DSEARCH_ROOT=${CMAKE_SOURCE_DIR}/include
        -DDISALLOWED_PATTERN=inja|nlohmann/json
        -P ${CMAKE_SOURCE_DIR}/cmake/check_header_purity.cmake
)
set_tests_properties(HeaderPurity PROPERTIES
    FAIL_REGULAR_EXPRESSION "HeaderPurity violation")
