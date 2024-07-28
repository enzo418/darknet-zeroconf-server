# Darknet object detection framework


# ==
# Darknet CLI is actually quite simple in terms of source files.
# ==
MESSAGE(STATUS "Setting up DARKNET CLI")

# clangd:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FILE (GLOB CLISRC server.cpp)
IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINRC windows/*.rc)
	LIST (APPEND CLISRC ${WINRC})
ENDIF ()

LIST (SORT CLISRC)

ADD_EXECUTABLE (server ${CLISRC} $<TARGET_OBJECTS:darknetobjlib>)
SET_TARGET_PROPERTIES (server PROPERTIES OUTPUT_NAME "darknet_server")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()

# uWebsocket
add_subdirectory(uSockets)
target_include_directories(server PRIVATE uSockets/src)

#
TARGET_LINK_LIBRARIES (server PRIVATE ${DARKNET_LINK_LIBS} usocketslib)


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
