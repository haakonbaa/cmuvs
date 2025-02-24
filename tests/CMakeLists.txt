# Enable testing
enable_testing()

# Ensure Catch2 is available
set(CATCH2_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/Catch2")

# Add Catch2
add_subdirectory(${CATCH2_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Catch2Build)

# List of test files
set(TEST_SOURCES
    test_main.cpp
)

# Include the library's source files to ensure linkage
set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/util.cpp
)

# Create the test executable
add_executable(test_runner ${TEST_SOURCES})

# Link against Catch2 and the main library
target_link_libraries(test_runner PRIVATE cmuvs Catch2::Catch2WithMain)

# Register the tests with CTest
include(CTest)
add_test(NAME MyLibraryTests COMMAND test_runner)
