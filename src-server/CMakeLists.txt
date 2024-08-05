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

# Note: these are mutually exclusive.
option(ASAN "Enable AddressSanitizer (ASan)" OFF)
option(MSAN "Enable MemorySanitizer (MSan)" OFF)
option(TSAN "Enable ThreadSanitizer (TSan)" OFF)
option(UBSAN "Enable UndefinedBehaviorSanitizer (UBSan)" OFF)

if(MSAN AND NOT CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang")
  message(SEND_ERROR "MemorySanitizer requires clang. Try again with -DCMAKE_C_COMPILER=clang")
endif()

if(ASAN)
  message(STATUS "Enabling AddressSanitizer (ASan)")
  message(WARNING "Don't forget to set ASAN_OPTIONS=protect_shadow_gap=0 before running or cuda will fail to initialize.")

  list(APPEND darknet_defines __ASAN__=1)
  if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|GNU|Clang")
    # set CMAKE_C_FLAGS, CMAKE_SHARED_LINKER_FLAGS and CMAKE_EXE_LINKER_FLAGS
    # or ...
	target_compile_options(server PRIVATE -fsanitize=address -fno-omit-frame-pointer)
	target_link_options(server PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  elseif(MSVC)
    target_compile_options(server PRIVATE /fsanitize=address)
    target_link_options(server PRIVATE /fsanitize=address)
  else()
    message(SEND_ERROR "AddressSanitizer support requires clang, gcc, or msvc. Try again with -DCMAKE_C_COMPILER.")
  endif()
endif()

if(MSAN)
  list(APPEND darknet_defines __MSAN__=1)
  target_compile_options(server PRIVATE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins=2)
  target_link_options(server PRIVATE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins=2)
endif()

if(TSAN)
  list(APPEND darknet_defines __TSAN__=1)

  message(WARNING "ThreadSanitizer and opencv do not play well together.")

  if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|GNU|Clang")
	target_compile_options(server PRIVATE
		-fsanitize=thread
		-fno-omit-frame-pointer
	)

	target_link_options(server PRIVATE
		-fsanitize=thread
		-fno-omit-frame-pointer
	)
  else()
    message(SEND_ERROR "ThreadSanitizer support requires clang or gcc. Try again with -DCMAKE_C_COMPILER.")
  endif()
endif()

if(UBSAN)
  list(APPEND darknet_defines __UBSAN__=1)
  if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|GNU|Clang")
    target_compile_options(server PRIVATE
		-fsanitize=undefined
		-fno-omit-frame-pointer
	)

	target_link_options(server PRIVATE
		-fsanitize=undefined
		-fno-omit-frame-pointer
	)
  elseif(MSVC)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=undefined")
  else()
    message(SEND_ERROR "UndefinedBehaviorSanitizer support requires clang, gcc, or msvc. Try again with -DCMAKE_C_COMPILER.")
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC AND NOT ASAN AND NOT MSAN AND NOT TSAN AND NOT UBSAN)
	target_compile_options(server PRIVATE -fsanitize=address)
	target_link_options(server PRIVATE -fsanitize=address)
endif()

IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (server PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()

target_compile_definitions(server PRIVATE ${uv_defines})

# uWebsocket
IF (WIN32)
	set(BUILD_SHARED_LIBS FALSE)
endif()

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

