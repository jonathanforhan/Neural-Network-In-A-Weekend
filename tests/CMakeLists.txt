cmake_minimum_required(VERSION 3.20)
project(NN_Tests VERSION 1.0.0 LANGUAGES CXX)

#[[ Compile Shaders ]]
execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/spirv"
)

file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
foreach (SHADER ${SHADERS})
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(
            COMMAND ${glslc_executable} ${SHADER} -o "spirv/${FILENAME}.spv"
            OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/spirv/${FILENAME}.spv"
            DEPENDS ${SHADER} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/spirv"
            COMMENT "Compiling ${FILENAME}"
    )
    list(APPEND SPV_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/spirv/${FILENAME}.spv")
endforeach ()
add_custom_target(COMPILE_SHADERS ALL DEPENDS ${SPV_SHADERS})

#[[ Test Function ]]
function(TEST_PROJECT)
    file(GLOB_RECURSE CXX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    add_executable(${DIRNAME})

    if (MSVC)
        set_target_properties(${DIRNAME} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} /W4")
    else ()
        set_target_properties(${DIRNAME} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    endif ()

    target_sources(${DIRNAME} PUBLIC ${CXX_SOURCES})
    target_include_directories(${DIRNAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${DIRNAME} PUBLIC NN)
    add_dependencies(${DIRNAME} COMPILE_SHADERS)
endfunction()

#[[ Recurse Projects ]]
file(GLOB SUBDIRS LIST_DIRECTORIES true "*")
foreach (DIR ${SUBDIRS})
    get_filename_component(FILENAME ${DIR} NAME)
    if (IS_DIRECTORY ${DIR} AND NOT ${FILENAME} STREQUAL "shaders")
        add_subdirectory(${DIR})
    endif ()
endforeach ()