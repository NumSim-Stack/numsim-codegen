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

file(GLOB_RECURSE HEADERS
    "${SEARCH_ROOT}/*.h"
    "${SEARCH_ROOT}/*.hpp"
)

set(VIOLATIONS "")
foreach(HEADER IN LISTS HEADERS)
    file(STRINGS "${HEADER}" MATCHES REGEX "${DISALLOWED_PATTERN}")
    if(MATCHES)
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

message(STATUS "HeaderPurity: scanned ${HEADERS} — no violations")
