PROJECT(main_test)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
SET(CMAKE_CXX_STANDARD 20)

ENABLE_TESTING()
FIND_PACKAGE(benchmark REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(fmt REQUIRED)
FIND_PACKAGE(GTest REQUIRED)

SET(TESTS
    main.cpp
    isample.cpp
    benchmark.cpp
    circular_distributions.cpp
    rs_test_1d.cpp
    rs_test_2d.cpp
    misc_distributions.cpp
    ratio_uniform.cpp
    set_filter.cpp
    slice_sampling.cpp
    spherical_distributions.cpp
    unit_tests.cpp
    envmaps.cpp
)

FOREACH(TEST_CASE ${TESTS})
    GET_FILENAME_COMPONENT(EXE_NAME ${TEST_CASE} NAME_WE)
    ADD_EXECUTABLE(${EXE_NAME} ${TEST_CASE})
    IF(MSVC)
        TARGET_COMPILE_OPTIONS(${EXE_NAME} PRIVATE /W4 /WX)
    ELSE()
        TARGET_COMPILE_OPTIONS(${EXE_NAME} PRIVATE -Wall -Wextra -pedantic -mavx2 -g)
    ENDIF()
    TARGET_LINK_LIBRARIES(${EXE_NAME} importance_distributions ${Boost_LIBRARIES} fmt::fmt tbb)
ENDFOREACH()

# Additional libraries
TARGET_LINK_LIBRARIES(benchmark
    benchmark::benchmark
)

TARGET_LINK_LIBRARIES(unit_tests
    GTest::gtest_main
)

INCLUDE(GoogleTest)
GTEST_DISCOVER_TESTS(unit_tests)
