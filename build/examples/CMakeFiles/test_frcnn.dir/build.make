# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/saa4/Caffe_with_updated_functions

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/saa4/Caffe_with_updated_functions/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/test_frcnn.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/test_frcnn.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/test_frcnn.dir/flags.make

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o: examples/CMakeFiles/test_frcnn.dir/flags.make
examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o: ../examples/FRCNN/test_frcnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saa4/Caffe_with_updated_functions/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o"
	cd /home/saa4/Caffe_with_updated_functions/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o -c /home/saa4/Caffe_with_updated_functions/examples/FRCNN/test_frcnn.cpp

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.i"
	cd /home/saa4/Caffe_with_updated_functions/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saa4/Caffe_with_updated_functions/examples/FRCNN/test_frcnn.cpp > CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.i

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.s"
	cd /home/saa4/Caffe_with_updated_functions/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saa4/Caffe_with_updated_functions/examples/FRCNN/test_frcnn.cpp -o CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.s

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.requires:

.PHONY : examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.requires

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.provides: examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/test_frcnn.dir/build.make examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.provides.build
.PHONY : examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.provides

examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.provides.build: examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o


# Object files for target test_frcnn
test_frcnn_OBJECTS = \
"CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o"

# External object files for target test_frcnn
test_frcnn_EXTERNAL_OBJECTS =

examples/FRCNN/test_frcnn: examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o
examples/FRCNN/test_frcnn: examples/CMakeFiles/test_frcnn.dir/build.make
examples/FRCNN/test_frcnn: src/api/FRCNN/libFRCNN_api.so
examples/FRCNN/test_frcnn: lib/libcaffe.so.1.0.0
examples/FRCNN/test_frcnn: lib/libcaffeproto.a
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libglog.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libsz.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libz.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libdl.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libm.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libglog.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libsz.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libz.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libdl.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libm.so
examples/FRCNN/test_frcnn: /usr/local/lib/libprotobuf.a
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/liblmdb.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libleveldb.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libcudart.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libcurand.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libcublas.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libcudnn.so
examples/FRCNN/test_frcnn: /usr/local/lib/libopencv_highgui.so.2.4.13
examples/FRCNN/test_frcnn: /usr/local/lib/libopencv_imgproc.so.2.4.13
examples/FRCNN/test_frcnn: /usr/local/lib/libopencv_core.so.2.4.13
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libcudart.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libnppc.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libnppi.so
examples/FRCNN/test_frcnn: /usr/local/cuda/lib64/libnpps.so
examples/FRCNN/test_frcnn: /usr/lib/liblapack.so
examples/FRCNN/test_frcnn: /usr/lib/libcblas.so
examples/FRCNN/test_frcnn: /usr/lib/libatlas.so
examples/FRCNN/test_frcnn: /usr/lib/x86_64-linux-gnu/libboost_python.so
examples/FRCNN/test_frcnn: examples/CMakeFiles/test_frcnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/saa4/Caffe_with_updated_functions/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FRCNN/test_frcnn"
	cd /home/saa4/Caffe_with_updated_functions/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_frcnn.dir/link.txt --verbose=$(VERBOSE)
	cd /home/saa4/Caffe_with_updated_functions/build/examples && ln -sf /home/saa4/Caffe_with_updated_functions/build/examples/FRCNN/test_frcnn /home/saa4/Caffe_with_updated_functions/build/examples/FRCNN/test_frcnn.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/test_frcnn.dir/build: examples/FRCNN/test_frcnn

.PHONY : examples/CMakeFiles/test_frcnn.dir/build

examples/CMakeFiles/test_frcnn.dir/requires: examples/CMakeFiles/test_frcnn.dir/FRCNN/test_frcnn.cpp.o.requires

.PHONY : examples/CMakeFiles/test_frcnn.dir/requires

examples/CMakeFiles/test_frcnn.dir/clean:
	cd /home/saa4/Caffe_with_updated_functions/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/test_frcnn.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/test_frcnn.dir/clean

examples/CMakeFiles/test_frcnn.dir/depend:
	cd /home/saa4/Caffe_with_updated_functions/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/saa4/Caffe_with_updated_functions /home/saa4/Caffe_with_updated_functions/examples /home/saa4/Caffe_with_updated_functions/build /home/saa4/Caffe_with_updated_functions/build/examples /home/saa4/Caffe_with_updated_functions/build/examples/CMakeFiles/test_frcnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/test_frcnn.dir/depend

