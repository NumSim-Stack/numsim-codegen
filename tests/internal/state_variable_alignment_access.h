#ifndef NUMSIM_CODEGEN_TESTS_INTERNAL_STATE_VARIABLE_ALIGNMENT_ACCESS_H
#define NUMSIM_CODEGEN_TESTS_INTERNAL_STATE_VARIABLE_ALIGNMENT_ACCESS_H

// Test-only friend access to `ConstitutiveModel`'s private members for
// negative-path tests of the StateVariable ↔ SymbolDecl alignment
// invariant. PR #66 round-3 review #7.
//
// **Why this header exists.** `ConstitutiveModel` grants `friend` to
// `numsim::codegen::testing_detail::state_variable_alignment_access`.
// If two test TUs each defined that struct independently — even with
// the same name and same body — the program would be ODR-violating
// ([basic.def.odr]/12: at most one definition of a non-inline class).
// Factoring the definition here gives the test binary a single ODR
// slot; future test TUs that need the same access `#include` this
// header rather than redefining the struct.
//
// Methods are `static` so the struct itself is never instantiated;
// each method is `inline` so multiple TU includes link cleanly.
// Production code paths cannot accidentally rely on this — the header
// lives under `tests/internal/` and isn't on the public include path.

#include <numsim_codegen/recipe.h>

#include <cstddef>
#include <string>
#include <utility>

namespace numsim::codegen::testing_detail {

struct state_variable_alignment_access {
  // Clobber the name of the SymbolDecl pointed at by a state variable's
  // current/old index. Simulates a mutating pass renaming a symbol
  // without keeping the pair in sync.
  static inline void poison_symbol_name(ConstitutiveModel &m,
                                        std::size_t idx,
                                        std::string new_name) {
    m.m_symbols[idx].name = std::move(new_name);
  }
  // Clobber the category of a symbol.
  static inline void poison_symbol_category(ConstitutiveModel &m,
                                            std::size_t idx,
                                            SymbolDecl::Category new_cat) {
    m.m_symbols[idx].category = new_cat;
  }
  // Clobber the kind of a symbol.
  static inline void poison_symbol_kind(ConstitutiveModel &m,
                                        std::size_t idx,
                                        SymbolDecl::Kind new_kind) {
    m.m_symbols[idx].kind = new_kind;
  }
  // Clobber the dim of a tensor symbol.
  static inline void poison_symbol_dim(ConstitutiveModel &m,
                                       std::size_t idx,
                                       std::size_t new_dim) {
    m.m_symbols[idx].dim = new_dim;
  }
  // Push the current_symbol_idx of a state variable past the end of
  // m_symbols to simulate a vector-shrink that left stale indices.
  static inline void poison_state_var_current_idx(ConstitutiveModel &m,
                                                  std::size_t sv_idx,
                                                  std::size_t new_idx) {
    m.m_state_variables[sv_idx].current_symbol_idx = new_idx;
  }
};

} // namespace numsim::codegen::testing_detail

#endif // NUMSIM_CODEGEN_TESTS_INTERNAL_STATE_VARIABLE_ALIGNMENT_ACCESS_H
