include(ExternalProject)

MESSAGE(STATUS "Setting up DARKNET Server")

set(DEPENDENCY_INSTALL_DIR ${PROJECT_BINARY_DIR}/install)
set(DEPENDENCY_INCLUDE_DIR ${DEPENDENCY_INSTALL_DIR}/include)
set(DEPENDENCY_LIB_DIR ${DEPENDENCY_INSTALL_DIR}/lib)

# clangd:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FILE (GLOB CLISRC server.cpp)
IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINRC windows/*.rc)
	LIST (APPEND CLISRC ${WINRC})
ENDIF ()

LIST (SORT CLISRC)

ADD_EXECUTABLE (server ${CLISRC} SnowflakeGenerator_64.cpp)
SET_TARGET_PROPERTIES (server PROPERTIES OUTPUT_NAME "darknet_server")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	# NOTE: Set ASAN_OPTIONS=protect_shadow_gap=0 before running
	# 	e.g.	$ ASAN_OPTIONS=protect_shadow_gap=0 darknet_server ...
	target_compile_options(server PRIVATE -fsanitize=address)
	target_link_options(server PRIVATE -fsanitize=address)
endif()

IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()

# uWebsocket
add_subdirectory(uSockets)
target_include_directories(server PRIVATE uSockets/src)

# mdns: (TODO: use a more professional approach -> Dependency.cmake)
ExternalProject_Add(
    dep-mdns
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mdns-cpp
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${DEPENDENCY_INSTALL_DIR}
		-DCMAKE_INSTALL_INCLUDEDIR=${DEPENDENCY_INCLUDE_DIR}
		-DCMAKE_INSTALL_LIBDIR=${DEPENDENCY_LIB_DIR}
		-DMDNS_BUILD_EXAMPLE=OFF
		-DMDNS_DISABLE_LOGGING=ON
    TEST_COMMAND ""
)

add_dependencies(server dep-mdns)
target_include_directories(server PRIVATE ${DEPENDENCY_INCLUDE_DIR})
target_link_directories(server PRIVATE ${DEPENDENCY_LIB_DIR})

#
TARGET_LINK_LIBRARIES (server PRIVATE darknet ${DARKNET_LINK_LIBS} usocketslib mdns)

IF (UNIX)
	# non-Windows installation is very simple
	INSTALL (TARGETS server DESTINATION bin)
ELSE ()
	# more complicated install for Windows so we also get the .DLL files copied over from vcpkg
	# https://stackoverflow.com/a/72088102/13022
	INSTALL (TARGETS server
		DESTINATION bin
		RUNTIME_DEPENDENCIES
			PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "wpaxholder" "HvsiFileTrust" "PdmUtilities"
			POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
			DIRECTORIES ${CMAKE_BINARY_DIR}/bin
		RUNTIME DESTINATION bin
		)
ENDIF ()

