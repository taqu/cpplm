cmake_minimum_required(VERSION 3.25)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PROJECT_NAME cpplm)
project(${PROJECT_NAME})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(AFTER ${PROJECT_ROOT}/include)
set(HEADER_DIR ${PROJECT_ROOT}/include)
set(SOURCE_DIR ${PROJECT_ROOT}/src)
set(THIRDPARTY_DIR ${PROJECT_ROOT}/thirdparty)

# build thirdparties
configure_file(${THIRDPARTY_DIR}/CMakeLists.txt.in ${THIRDPARTY_DIR}/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${THIRDPARTY_DIR})
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

set(ONEDNN_ROOT ${THIRDPARTY_DIR}/oneDNN)
set(ONEDNN_INCLUDE ${ONEDNN_ROOT}/include)
set(ONEDNN_LIB ${ONEDNN_ROOT}/lib)

include_directories(AFTER ${ONEDNN_INCLUDE})
include_directories(AFTER $ENV{CUDA_PATH}/include)

link_directories(AFTER ${ONEDNN_LIB})
link_directories(AFTER $ENV{CUDA_PATH}/lib/x64)

add_library(OPENCL MODULE IMPORTED)
set_target_properties(OPENCL PROPERTIES IMPORTED_LOCATION_DEBUG "OpenCL.lib" IMPORTED_LOCATION_RELEASE "OpenCL.lib")

add_library(ONEDNN MODULE IMPORTED)
set_target_properties(ONEDNN PROPERTIES IMPORTED_LOCATION_DEBUG "dnnld.lib" IMPORTED_LOCATION_RELEASE "dnnl.lib")

########################################################################
# Sources
set(HEADERS
    "${HEADER_DIR}/cppgpt.h"
    "${HEADER_DIR}/gguf.h")

set(SOURCES
    "${SOURCE_DIR}/cppgpt.cpp"
    "${SOURCE_DIR}/gguf.cpp")

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${SOURCES} "${SOURCE_DIR}/main.cpp")

set(FILES ${HEADERS} ${SOURCES} "${SOURCE_DIR}/main.cpp")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(${PROJECT_NAME} ${FILES})

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MBCS /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd /std:c++20 /std:c11")
    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++20")
    endif()
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1 /Gy /GR /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Oi /GL /GR /DNDEBUG /EHsc-")
    target_link_libraries(${PROJECT_NAME} ONEDNN OPENCL)

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++20 -std=gnu++20 -march=x86-64-v3 -fno-exceptions")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    target_link_libraries(${PROJECT_NAME} ONEDNN OPENCL)
elseif(APPLE)
elseif(APPLE)
endif()

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(test)
#set(CMAKE_FOLDER "oneDNN")
#set(CMAKE_INSTALL_PREFIX ${THIRDPARTY_DIR}/oneDNN)
#set(DNNL_GPU_RUNTIME OCL)
#add_subdirectory(thirdparty/work/oneDNN)
#unset(CMAKE_INSTALL_PREFIX)
#unset(CMAKE_FOLDER)
