# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ece408/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /build

# Include any dependencies generated for this target.
include CMakeFiles/m3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/m3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/m3.dir/flags.make

CMakeFiles/m3.dir/m3.cc.o: CMakeFiles/m3.dir/flags.make
CMakeFiles/m3.dir/m3.cc.o: /ece408/project/m3.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/m3.dir/m3.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/m3.dir/m3.cc.o -c /ece408/project/m3.cc

CMakeFiles/m3.dir/m3.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/m3.dir/m3.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/m3.cc > CMakeFiles/m3.dir/m3.cc.i

CMakeFiles/m3.dir/m3.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/m3.dir/m3.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/m3.cc -o CMakeFiles/m3.dir/m3.cc.s

CMakeFiles/m3.dir/m3.cc.o.requires:

.PHONY : CMakeFiles/m3.dir/m3.cc.o.requires

CMakeFiles/m3.dir/m3.cc.o.provides: CMakeFiles/m3.dir/m3.cc.o.requires
	$(MAKE) -f CMakeFiles/m3.dir/build.make CMakeFiles/m3.dir/m3.cc.o.provides.build
.PHONY : CMakeFiles/m3.dir/m3.cc.o.provides

CMakeFiles/m3.dir/m3.cc.o.provides.build: CMakeFiles/m3.dir/m3.cc.o


# Object files for target m3
m3_OBJECTS = \
"CMakeFiles/m3.dir/m3.cc.o"

# External object files for target m3
m3_EXTERNAL_OBJECTS =

m3: CMakeFiles/m3.dir/m3.cc.o
m3: CMakeFiles/m3.dir/build.make
m3: /usr/local/cuda/lib64/libcudart_static.a
m3: /usr/lib/x86_64-linux-gnu/librt.so
m3: libece408net.a
m3: src/libMiniDNNLib.a
m3: src/libGpuConv.a
m3: /usr/local/cuda/lib64/libcudart_static.a
m3: /usr/lib/x86_64-linux-gnu/librt.so
m3: CMakeFiles/m3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable m3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/m3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/m3.dir/build: m3

.PHONY : CMakeFiles/m3.dir/build

CMakeFiles/m3.dir/requires: CMakeFiles/m3.dir/m3.cc.o.requires

.PHONY : CMakeFiles/m3.dir/requires

CMakeFiles/m3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/m3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/m3.dir/clean

CMakeFiles/m3.dir/depend:
	cd /build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ece408/project /ece408/project /build /build /build/CMakeFiles/m3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/m3.dir/depend
