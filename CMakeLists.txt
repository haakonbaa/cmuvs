cmake_minimum_required(VERSION 3.10)
project(CMUVs VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the main library
add_library(cmuvs SHARED
    src/cmuvs.cpp
    src/util.cpp
    src/se3.cpp
)

# Set include directories
target_include_directories(cmuvs PUBLIC include)

# Install target
install(TARGETS cmuvs
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

if(MSVC)
    target_compile_options(cmuvs PRIVATE /W4 /WX)
else()
    target_compile_options(cmuvs PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()


# Add the test directory
add_subdirectory(tests)
