cmake_minimum_required (VERSION 3.18)
project(cuda VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

set(PROJECT_SOURCES
        mandelbrotwidget.h
        renderthread.h
        main.cpp
        mandelbrotwidget.cpp
        renderthread.cpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(cuda
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET cuda APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(cuda SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(cuda
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(cuda PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

set_target_properties(cuda PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(cuda)
endif()

add_subdirectory ("cuda-library")

target_link_libraries(cuda PUBLIC cuda-library)

# add the library dir to the search path for include files
# so that we will find math_tools.h
target_include_directories(cuda PUBLIC
                          "${PROJECT_SOURCE_DIR}/cuda-library"
                          )

# See https://stackoverflow.com/questions/6594796/how-do-i-make-cmake-output-into-a-bin-dir
# and https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
# Make CMake save the files in a ./bin/Debug (or ./bin/Release) directory
set_target_properties(
    cuda cuda-library
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
