# Enable testing
enable_testing()

include(FetchContent)

# Fetch Catch2 if not present
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)

FetchContent_MakeAvailable(Catch2)

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
