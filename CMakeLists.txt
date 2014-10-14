project(image_hash)
find_package(OpenCV REQUIRED)
add_executable(image_hash image_hash.cpp)
target_link_libraries(image_hash ${OpenCV_LIBS})
