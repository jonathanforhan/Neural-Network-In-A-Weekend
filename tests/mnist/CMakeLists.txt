TEST_PROJECT()

execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/dataset"
        "${CMAKE_CURRENT_BINARY_DIR}/dataset"
)