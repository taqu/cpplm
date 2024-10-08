cmake_minimum_required(VERSION 3.2)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME test_lm)
project(${PROJECT_NAME})

set(CPPGPT_HEADERS ${HEADERS})
set(CPPGPT_SOURCES ${SOURCES})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HEADERS "catch_amalgamated.hpp")
set(SOURCES
    ${SOURCE_DIR}/catch_amalgamated.cpp
    ${SOURCE_DIR}/test_gguf.cpp
    ${SOURCE_DIR}/test_hash.cpp
    ${SOURCE_DIR}/test_container.cpp
    ${SOURCE_DIR}/main.cpp)

include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR}../include)

source_group("include" FILES ${HEADERS} ${CPPGPT_HEADERS})
source_group("src" FILES ${SOURCES} ${CPPGPT_SOURCES})

set(FILES ${HEADERS} ${SOURCES} ${CPPGPT_HEADERS} ${CPPGPT_SOURCES})

add_executable(${PROJECT_NAME} ${FILES})

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_UNICODE /DUNICODE /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd /std:c++20 /std:c11 /DLG3_GFX_USE_WIN32 /DVK_USE_PLATFORM_WIN32_KHR /DGLEW_STATIC")
    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++20")
    endif()
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /GL /GR- /DNDEBUG /EHsc-")
    target_link_libraries(${PROJECT_NAME} MIMALLOC OPENCL)

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++20 -std=gnu++20 -march=x86-64-v3 -fno-exceptions -DVK_USE_PLATFORM_WIN32_KHR")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    target_link_libraries(${PROJECT_NAME} MIMALLOC OPENCL)
elseif(APPLE)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_dependencies(${PROJECT_NAME} cpplm)

