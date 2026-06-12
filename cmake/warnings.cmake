# ─── First-party warning configuration ──────────────────────────────────────
#
# Defines an INTERFACE target `numsim_codegen_warnings` carrying a comprehensive
# warning set. Link it PRIVATE into first-party COMPILED targets only (the
# library + tests + generators) — never propagate to consumers, and never to
# fetched deps (numsim-cas / tmech / Eigen / gtest), whose headers are included
# with SYSTEM/-isystem so their warnings stay silent.
#
# `-Werror` is enabled when building Debug, or when NUMSIM_CODEGEN_WERROR=ON.
# The intent: a Debug build is the strict gate (warnings = errors); Release and
# day-to-day local builds warn but don't fail, so a newer compiler's novel
# diagnostics don't block work. CI builds Debug (and/or sets the option) to
# enforce zero-warning first-party code.

add_library(numsim_codegen_warnings INTERFACE)

# Shared across GCC and Clang.
set(_ncg_common_warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow                 # a declaration shadowing an outer one
    -Wnon-virtual-dtor       # polymorphic base with non-virtual dtor
    -Woverloaded-virtual     # a derived method hiding a virtual
    -Wold-style-cast         # C-style casts
    -Wcast-align             # casts increasing alignment requirements
    -Wunused                 # anything unused
    -Wconversion             # implicit conversions that may alter a value
    -Wsign-conversion        # implicit sign conversions
    -Wnull-dereference       # a null-pointer dereference
    -Wdouble-promotion       # implicit float -> double
    -Wformat=2               # printf/scanf format checking
    -Wimplicit-fallthrough   # a switch case falling through
)

# Deliberately suppressed (both GCC and Clang). These two -Wextra sub-warnings
# are noise/false-positive-prone for idioms this codebase uses intentionally:
#   * missing-field-initializers — partial aggregate-init (`SymbolDecl{a,b,c}`)
#     where trailing members carry value-init defaults; intentional, not a bug.
set(_ncg_common_suppress
    -Wno-missing-field-initializers
)

# GCC-only extras (Clang either lacks these or spells them differently).
set(_ncg_gcc_warnings
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
)

# GCC-only suppressions: -Wmaybe-uninitialized is optimization-dependent and
# notoriously false-positive-prone (it does not even fire at the Debug -O0 that
# gates -Werror); keep the definite -Wuninitialized, drop the speculative one.
set(_ncg_gcc_suppress
    -Wno-maybe-uninitialized
)

target_compile_options(numsim_codegen_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:${_ncg_common_warnings};${_ncg_gcc_warnings};${_ncg_common_suppress};${_ncg_gcc_suppress}>
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:${_ncg_common_warnings};${_ncg_common_suppress}>
    # Warnings-as-errors in Debug or when explicitly requested.
    $<$<AND:$<OR:$<CONFIG:Debug>,$<BOOL:${NUMSIM_CODEGEN_WERROR}>>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang,AppleClang>>>:-Werror>
)

# Convenience alias matching the project's `numsim::` namespace.
add_library(numsim::codegen_warnings ALIAS numsim_codegen_warnings)
