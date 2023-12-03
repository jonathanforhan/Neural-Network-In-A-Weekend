cmake_minimum_required(VERSION 3.20)
project(NN VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#[[ EXTERN ]]
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_executable REQUIRED NAMES glslc HINTS Vulkan::glslc)

file(GLOB_RECURSE CXX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HXX_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

add_library(NN STATIC ${CXX_SOURCES} ${HXX_HEADERS})
target_link_libraries(NN PUBLIC Vulkan::Vulkan)

if (MSVC)
    set_target_properties(NN PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else ()
    set_target_properties(NN PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif ()

target_include_directories(NN PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(tests)
