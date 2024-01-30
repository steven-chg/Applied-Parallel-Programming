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
include src/CMakeFiles/MiniDNNLib.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/MiniDNNLib.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/MiniDNNLib.dir/flags.make

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o: /ece408/project/src/mnist.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/mnist.cc.o -c /ece408/project/src/mnist.cc

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/mnist.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/mnist.cc > CMakeFiles/MiniDNNLib.dir/mnist.cc.i

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/mnist.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/mnist.cc -o CMakeFiles/MiniDNNLib.dir/mnist.cc.s

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o


src/CMakeFiles/MiniDNNLib.dir/network.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/network.cc.o: /ece408/project/src/network.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/network.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/network.cc.o -c /ece408/project/src/network.cc

src/CMakeFiles/MiniDNNLib.dir/network.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/network.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/network.cc > CMakeFiles/MiniDNNLib.dir/network.cc.i

src/CMakeFiles/MiniDNNLib.dir/network.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/network.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/network.cc -o CMakeFiles/MiniDNNLib.dir/network.cc.s

src/CMakeFiles/MiniDNNLib.dir/network.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/network.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/network.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/network.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/network.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/network.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/network.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/network.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o: /ece408/project/src/layer/ave_pooling.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o -c /ece408/project/src/layer/ave_pooling.cc

src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/ave_pooling.cc > CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/ave_pooling.cc -o CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o: /ece408/project/src/layer/conv.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o -c /ece408/project/src/layer/conv.cc

src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/conv.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/conv.cc > CMakeFiles/MiniDNNLib.dir/layer/conv.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/conv.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/conv.cc -o CMakeFiles/MiniDNNLib.dir/layer/conv.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o: /ece408/project/src/layer/conv_cpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o -c /ece408/project/src/layer/conv_cpu.cc

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/conv_cpu.cc > CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/conv_cpu.cc -o CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o: /ece408/project/src/layer/conv_cust.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o -c /ece408/project/src/layer/conv_cust.cc

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/conv_cust.cc > CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/conv_cust.cc -o CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o: /ece408/project/src/layer/fully_connected.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o -c /ece408/project/src/layer/fully_connected.cc

src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/fully_connected.cc > CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/fully_connected.cc -o CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o: /ece408/project/src/layer/max_pooling.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o -c /ece408/project/src/layer/max_pooling.cc

src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/max_pooling.cc > CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/max_pooling.cc -o CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o: /ece408/project/src/layer/relu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o -c /ece408/project/src/layer/relu.cc

src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/relu.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/relu.cc > CMakeFiles/MiniDNNLib.dir/layer/relu.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/relu.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/relu.cc -o CMakeFiles/MiniDNNLib.dir/layer/relu.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o: /ece408/project/src/layer/sigmoid.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o -c /ece408/project/src/layer/sigmoid.cc

src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/sigmoid.cc > CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/sigmoid.cc -o CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o: /ece408/project/src/layer/softmax.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o -c /ece408/project/src/layer/softmax.cc

src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/softmax.cc > CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/softmax.cc -o CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o


src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o: /ece408/project/src/loss/cross_entropy_loss.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o -c /ece408/project/src/loss/cross_entropy_loss.cc

src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/loss/cross_entropy_loss.cc > CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.i

src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/loss/cross_entropy_loss.cc -o CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.s

src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o


src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o: /ece408/project/src/loss/mse_loss.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o -c /ece408/project/src/loss/mse_loss.cc

src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/loss/mse_loss.cc > CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.i

src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/loss/mse_loss.cc -o CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.s

src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o


src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o: /ece408/project/src/optimizer/sgd.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o -c /ece408/project/src/optimizer/sgd.cc

src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/optimizer/sgd.cc > CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.i

src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/optimizer/sgd.cc -o CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.s

src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o


src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o: src/CMakeFiles/MiniDNNLib.dir/flags.make
src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o: /ece408/project/src/layer/custom/cpu-new-forward.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o"
	cd /build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o -c /ece408/project/src/layer/custom/cpu-new-forward.cc

src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.i"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/src/layer/custom/cpu-new-forward.cc > CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.i

src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.s"
	cd /build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/src/layer/custom/cpu-new-forward.cc -o CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.s

src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.requires:

.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.requires

src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.provides: src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.requires
	$(MAKE) -f src/CMakeFiles/MiniDNNLib.dir/build.make src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.provides.build
.PHONY : src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.provides

src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.provides.build: src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o


# Object files for target MiniDNNLib
MiniDNNLib_OBJECTS = \
"CMakeFiles/MiniDNNLib.dir/mnist.cc.o" \
"CMakeFiles/MiniDNNLib.dir/network.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o" \
"CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o" \
"CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o" \
"CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o" \
"CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o"

# External object files for target MiniDNNLib
MiniDNNLib_EXTERNAL_OBJECTS =

src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/network.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/build.make
src/libMiniDNNLib.a: src/CMakeFiles/MiniDNNLib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Linking CXX static library libMiniDNNLib.a"
	cd /build/src && $(CMAKE_COMMAND) -P CMakeFiles/MiniDNNLib.dir/cmake_clean_target.cmake
	cd /build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MiniDNNLib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/MiniDNNLib.dir/build: src/libMiniDNNLib.a

.PHONY : src/CMakeFiles/MiniDNNLib.dir/build

src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/mnist.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/network.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/ave_pooling.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/conv.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cpu.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/conv_cust.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/fully_connected.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/max_pooling.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/relu.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/sigmoid.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/softmax.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/loss/cross_entropy_loss.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/loss/mse_loss.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/optimizer/sgd.cc.o.requires
src/CMakeFiles/MiniDNNLib.dir/requires: src/CMakeFiles/MiniDNNLib.dir/layer/custom/cpu-new-forward.cc.o.requires

.PHONY : src/CMakeFiles/MiniDNNLib.dir/requires

src/CMakeFiles/MiniDNNLib.dir/clean:
	cd /build/src && $(CMAKE_COMMAND) -P CMakeFiles/MiniDNNLib.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/MiniDNNLib.dir/clean

src/CMakeFiles/MiniDNNLib.dir/depend:
	cd /build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ece408/project /ece408/project/src /build /build/src /build/src/CMakeFiles/MiniDNNLib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/MiniDNNLib.dir/depend

