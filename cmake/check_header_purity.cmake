# Header-purity guard for numsim-codegen.
#
# Fails if any header under SEARCH_ROOT mentions any name matching the
# DISALLOWED_PATTERN regex. Used to enforce that heavy template libraries
# (inja, nlohmann/json — see issue #20) stay PRIVATE to the STATIC core
# and never leak into the public headers that downstream TUs include.
#
# Invoked from tests/CMakeLists.txt as a CTest test. Run manually with:
#   cmake -DSEARCH_ROOT=include -DDISALLOWED_PATTERN='inja|nlohmann/json' \
#         -P cmake/check_header_purity.cmake

if(NOT DEFINED SEARCH_ROOT)
    message(FATAL_ERROR "SEARCH_ROOT not set")
endif()
if(NOT DEFINED DISALLOWED_PATTERN)
    message(FATAL_ERROR "DISALLOWED_PATTERN not set")
endif()

# Glob the common C/C++ header extensions. file(GLOB_RECURSE) re-evaluates
# every invocation when this script runs in -P (script) mode, so new headers
# are picked up automatically without CONFIGURE_DEPENDS.
file(GLOB_RECURSE HEADERS
    "${SEARCH_ROOT}/*.h"
    "${SEARCH_ROOT}/*.hpp"
    "${SEARCH_ROOT}/*.hh"
    "${SEARCH_ROOT}/*.hxx"
    "${SEARCH_ROOT}/*.inl"
    "${SEARCH_ROOT}/*.tpp"
    "${SEARCH_ROOT}/*.ipp"
    "${SEARCH_ROOT}/*.ixx"
)

# file(READ) the whole file rather than file(STRINGS), which truncates
# lines at an implementation-defined length and could let a long
# auto-generated header escape detection. Cost is proportional to total
# header bytes — fine for any realistic public-API surface.
set(VIOLATIONS "")
foreach(HEADER IN LISTS HEADERS)
    file(READ "${HEADER}" CONTENT)
    string(REGEX MATCH "${DISALLOWED_PATTERN}" HEADER_HIT "${CONTENT}")
    if(NOT HEADER_HIT STREQUAL "")
        list(APPEND VIOLATIONS "${HEADER}")
    endif()
endforeach()

if(VIOLATIONS)
    message("HeaderPurity violation: public headers must not reference ${DISALLOWED_PATTERN}")
    foreach(V IN LISTS VIOLATIONS)
        message("  - ${V}")
    endforeach()
    message(FATAL_ERROR "HeaderPurity check failed")
endif()

list(LENGTH HEADERS HEADER_COUNT)
message(STATUS "HeaderPurity: scanned ${HEADER_COUNT} headers — no violations")
