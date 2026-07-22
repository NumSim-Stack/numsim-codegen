add_executable(linear_elasticity linear_elasticity.cpp)
target_link_libraries(linear_elasticity PRIVATE numsim::codegen)

add_executable(moose_linear_elastic moose_linear_elastic.cpp)
target_link_libraries(moose_linear_elastic PRIVATE numsim::codegen)

add_executable(recipe_registry_gen recipe_registry_gen.cpp)
target_link_libraries(recipe_registry_gen PRIVATE numsim::codegen)

add_executable(resin_curing resin_curing.cpp)
target_link_libraries(resin_curing PRIVATE numsim::codegen)

add_executable(elastic_tangent elastic_tangent.cpp)
target_link_libraries(elastic_tangent PRIVATE numsim::codegen)

add_executable(hencky_hyperelastic hencky_hyperelastic.cpp)
target_link_libraries(hencky_hyperelastic PRIVATE numsim::codegen)

add_executable(first_piola_svk first_piola_svk.cpp)
target_link_libraries(first_piola_svk PRIVATE numsim::codegen)
