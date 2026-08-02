// MOOSE-API stub (#132): libMesh's warning-suppression bracket. The emitted
// .C wraps third-party linear-algebra includes (Eigen) in this pair so they
// survive MOOSE's -Werror builds. Deliberately no include guard — like the
// real header, it must be usable more than once per TU, always paired with
// libmesh/restore_warnings.h.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
