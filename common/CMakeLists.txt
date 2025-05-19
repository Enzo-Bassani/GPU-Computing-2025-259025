file(GLOB SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu
)

add_library(common STATIC ${SOURCES})
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(common PUBLIC cuda)
