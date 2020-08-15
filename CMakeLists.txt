#[[
	Build options:
	* BUILD_SHARED_LIBS (default off) builds as a shared library (if LIBSEEDFINDING_COMPILE is ON)
    * LIBSEEDFINDING_COMPILE (default off)
    * LIBSEEDFINDING_USE_CUDA (default off)
    * LIBSEEDFINDING_UPDATE_VERSION (default on)

    -------------------------------------------------------------------------------

	After installation with Cmake, a find_package(libseedfinding) is available.
	This creates a libseedfinding::libseedfinding target (if found).
	It can be linked like so:

	target_link_libraries(your_exe libseedfinding::libseedfinding)

	The following will build & install for later use.

	Linux/macOS:

	mkdir -p build
	cd build
	cmake -DCMAKE_BUILD_TYPE=Release ..
	sudo cmake --build . --target install

	Windows:

	mkdir build
	cd build
	cmake ..
	runas /user:Administrator "cmake --build . --config Release --target install"

	-------------------------------------------------------------------------------

	These variables are available after you run find_package(libseedfinding)
	* LIBSEEDFINDING_HEADER_PATH - this is the full path to the installed header (e.g. /usr/include/libseedfinding.h).
	* LIBSEEDFINDING_IS_USING_CUDA - a bool for if Cuda support is enabled.
	* LIBSEEDFINDING_IS_COMPILED - a bool for if the library is compiled, or otherwise header-only.
	* LIBSEEDFINDING_INCLUDE_DIR - the root path to libseedfinding's header (e.g. /usr/include).
	* LIBSEEDFINDING_LIBRARY - the full path to the library if compiled (e.g. /usr/lib/liblibseedfinding.so).
	* LIBSEEDFINDING_VERSION or libseedfinding_VERSION - the project's version string.
	* LIBSEEDFINDING_FOUND - a bool for if the target was found.

    Want to use precompiled headers (Cmake feature since v3.16)?
	It's as simple as doing the following (before linking):

	target_precompile_headers(libseedfinding::libseedfinding INTERFACE "${LIBSEEDFINDING_HEADER_PATH}")

	-------------------------------------------------------------------------------

	FindPython3 requires Cmake v3.12
	ARCH_INDEPENDENT option of write_basic_package_version_file() requires Cmake v3.14
]]
cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)

# On systems without Git installed, there were errors since execute_process seemed to not throw an error without it?
# Also to not check out the version.h its needed
find_package(Git QUIET)
if (Git_FOUND)
    # Gets the latest tag as a string like "v0.6.6"
    # Can silently fail if git isn't on the system
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE _raw_version_string
            ERROR_VARIABLE _git_tag_error
            )
endif ()

# execute_process can fail silenty, so check for an error
# if there was an error, just use the user agent as a version
if (_git_tag_error OR NOT Git_FOUND)
    message(WARNING "cpp-libseedfinding failed to find the latest Git tag, falling back to using the version in the header file.")
    # This is so the we can only bother to update the header
    file(STRINGS include/util.h _raw_version_string REGEX ".*version.*([0-9]+.?)+")
endif ()
# Needed since git tags have "v" prefixing them.
# Also used if the fallback to user agent string is being used.
string(REGEX MATCH "([0-9]+\\.?)+" _libseedfinding_version "${_raw_version_string}")

option(LIBSEEDFINDING_USE_CUDA "If ON the cuda kernel will be compiled" OFF)
if (LIBSEEDFINDING_USE_CUDA)
    set(LIBSEEDFINDING_IS_USING_CUDA TRUE)
    project(libseedfinding VERSION ${_libseedfinding_version} LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA support")
    else ()
        message(STATUS "No CUDA support")
    endif ()
else ()
    project(libseedfinding VERSION ${_libseedfinding_version} LANGUAGES CXX)
endif ()

# Lets you compile the program as a regular library instead of header-only
option(LIBSEEDFINDING_COMPILE "If ON, uses a Python script to split the header into a compilable header & source file (requires Python v3)." OFF)
# Just setting this variable here for people building in-tree
if (LIBSEEDFINDING_COMPILE)
    set(LIBSEEDFINDING_IS_COMPILED TRUE)
endif ()

# Defaults to static library
option(BUILD_SHARED_LIBS "Build the library as a shared library instead of static. Has no effect if using header-only." OFF)
if (BUILD_SHARED_LIBS AND WIN32 AND LIBSEEDFINDING_COMPILE)
    # Necessary for Windows if building shared libs
    # See https://stackoverflow.com/a/40743080
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# populate the version.h
option(LIBSEEDFINDING_UPDATE_VERSION "Download the version manifest from mojang and write it to version.h" ON)
if (LIBSEEDFINDING_UPDATE_VERSION)
    find_package(Python COMPONENTS Interpreter Development)
    if (Python_FOUND)
        execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_version.py ${CMAKE_CURRENT_SOURCE_DIR})
    else ()
        error("You are missing Python the version.h was not generated")
    endif ()
endif ()


# Used for default, common dirs that the end-user can change (if needed)
# like CMAKE_INSTALL_INCLUDEDIR or CMAKE_INSTALL_DATADIR
include(GNUInstallDirs)

if (LIBSEEDFINDING_COMPILE)
    configure_file(cmake/split.py "${CMAKE_CURRENT_BINARY_DIR}/split.py"
            COPYONLY
            )
    # Needs to be in the same dir as the python script
    file(GLOB SEEDFINDING_HEADERS "include/*.h")
    file(COPY ${SEEDFINDING_HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    # Used outside of this if-else
    set(_INTERFACE_OR_PUBLIC PUBLIC)
    # Brings in the Python3_EXECUTABLE path we can use.
    find_package(Python3 REQUIRED)
    # Actually split the file
    # Keeps the output in the build dir to not pollute the main dir
    execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/split.py"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            ERROR_VARIABLE _libseedfinding_split_error
            )
    if (_libseedfinding_split_error)
        message(FATAL_ERROR "Failed when trying to split Cpp-libseedfinding with the Python script.\n${_libseedfinding_split_error}")
    endif ()

    # split.py puts output in "out"
    set(_libseedfinding_build_includedir "${CMAKE_CURRENT_BINARY_DIR}/out")
    # This will automatically be either static or shared based on the value of BUILD_SHARED_LIBS

    add_library(${PROJECT_NAME} "${_libseedfinding_build_includedir}/libseedfinding.cc")
    target_sources(${PROJECT_NAME}
            PUBLIC
            $<BUILD_INTERFACE:${_libseedfinding_build_includedir}/libseedfinding.h>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libseedfinding.h>
            )
else ()
    set(_INTERFACE_OR_PUBLIC INTERFACE)
    add_library(${PROJECT_NAME} INTERFACE)
    set(_libseedfinding_build_includedir "${CMAKE_CURRENT_SOURCE_DIR}")
endif ()
# Lets you address the target with libseedfinding::libseedfinding
# Only useful if building in-tree, versus using it from an installation.
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Might be missing some, but this list is somewhat comprehensive
target_compile_features(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC}
        cxx_std_14
        cxx_nullptr
        cxx_lambdas
        cxx_override
        cxx_defaulted_functions
        cxx_attribute_deprecated
        cxx_auto_type
        cxx_decltype
        cxx_deleted_functions
        cxx_range_for
        cxx_sizeof_member
        )

target_include_directories(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC}
        $<BUILD_INTERFACE:${_libseedfinding_build_includedir}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

# here we could set what libraries we depend on
target_link_libraries(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC})

# Set the definitions to enable optional features
target_compile_definitions(${PROJECT_NAME} ${_INTERFACE_OR_PUBLIC} $<$<BOOL:${LIBSEEDFINDING_IS_USING_CUDA}>:"CPPLIBSEEDFINDING_CUDA_SUPPORT">)

# Cmake's find_package search path is different based on the system
# See https://cmake.org/cmake/help/latest/command/find_package.html for the list
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_TARGET_INSTALL_CMAKEDIR "${CMAKE_INSTALL_PREFIX}/cmake/${PROJECT_NAME}")
else ()
    # On Non-Windows, it should be /usr/lib/cmake/<name>/<name>Config.cmake
    # NOTE: This may or may not work for macOS...
    set(_TARGET_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif ()

include(CMakePackageConfigHelpers)

# Configures the meta-file libseedfindingConfig.cmake.in to replace variables with paths/values/etc.
configure_package_config_file("${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${_TARGET_INSTALL_CMAKEDIR}"
        # Passes the includedir install path
        PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR
        # There aren't any components, so don't use the macro
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
        )

if (LIBSEEDFINDING_COMPILE)
    write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
            # Example: if you find_package(libseedfinding 0.5.4)
            # then anything >= 0.5 and <= 1.0 is accepted
            COMPATIBILITY SameMajorVersion
            )
else ()
    write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
            # Example: if you find_pacrkage(libseedfinding 0.5.4)
            # then anything >= 0.5 and <= 1.0 is accepted
            COMPATIBILITY SameMajorVersion
            # Tells Cmake that it's a header-only lib
            # Mildly useful for end-users :)
            ARCH_INDEPENDENT
            )
endif ()

# Creates the export libseedfindingTargets.cmake
# This is strictly what holds compilation requirements
# and linkage information (doesn't find deps though).
install(TARGETS ${PROJECT_NAME}
        EXPORT libseedfindingTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(FILES
        "${_libseedfinding_build_includedir}/include/lcg.h"
        "${_libseedfinding_build_includedir}/include/simplex.h"
        "${_libseedfinding_build_includedir}/include/util.h"
        "${_libseedfinding_build_includedir}/include/version.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${_TARGET_INSTALL_CMAKEDIR}
        )

# NOTE: This path changes depending on if it's on Windows or Linux
install(EXPORT libseedfindingTargets
        # Puts the targets into the libseedfinding namespace
        # So this makes libseedfinding::libseedfinding linkable after doing find_package(libseedfinding)
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${_TARGET_INSTALL_CMAKEDIR}
        )

if (LIBSEEDFINDING_IS_USING_CUDA)
    add_subdirectory(test_cuda)
else ()
    add_subdirectory(test_cpp)
endif ()


add_custom_target(install_${PROJECT_NAME}
        sudo "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
        DEPENDS ${PROJECT_NAME}
        COMMENT "Installing ${PROJECT_NAME}"
        SOURCES include/lcg.h include/util.h include/simplex.h include/kernels.cuh include/version.h
        USES_TERMINAL)
