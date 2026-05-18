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
)

target_link_libraries(numsim_codegen_tests
    PRIVATE
        numsim::codegen
        GTest::gtest
)

include(GoogleTest)
gtest_discover_tests(numsim_codegen_tests)
