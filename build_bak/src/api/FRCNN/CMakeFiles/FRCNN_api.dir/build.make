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
CMAKE_SOURCE_DIR = /home/kakadinh/caffe_variances/Caffe_with_updated_functions

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build

# Include any dependencies generated for this target.
include src/api/FRCNN/CMakeFiles/FRCNN_api.dir/depend.make

# Include the progress variables for this target.
include src/api/FRCNN/CMakeFiles/FRCNN_api.dir/progress.make

# Include the compile flags for this target's objects.
include src/api/FRCNN/CMakeFiles/FRCNN_api.dir/flags.make

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/flags.make
src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o: ../src/api/FRCNN/frcnn_api.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o -c /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/frcnn_api.cpp

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.i"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/frcnn_api.cpp > CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.i

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.s"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/frcnn_api.cpp -o CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.s

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.requires:

.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.requires

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.provides: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.requires
	$(MAKE) -f src/api/FRCNN/CMakeFiles/FRCNN_api.dir/build.make src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.provides.build
.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.provides

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.provides.build: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o


src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/flags.make
src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o: ../src/api/FRCNN/rpn_api.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o -c /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/rpn_api.cpp

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FRCNN_api.dir/rpn_api.cpp.i"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/rpn_api.cpp > CMakeFiles/FRCNN_api.dir/rpn_api.cpp.i

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FRCNN_api.dir/rpn_api.cpp.s"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN/rpn_api.cpp -o CMakeFiles/FRCNN_api.dir/rpn_api.cpp.s

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.requires:

.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.requires

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.provides: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.requires
	$(MAKE) -f src/api/FRCNN/CMakeFiles/FRCNN_api.dir/build.make src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.provides.build
.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.provides

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.provides.build: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o


# Object files for target FRCNN_api
FRCNN_api_OBJECTS = \
"CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o" \
"CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o"

# External object files for target FRCNN_api
FRCNN_api_EXTERNAL_OBJECTS =

src/api/FRCNN/libFRCNN_api.so: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o
src/api/FRCNN/libFRCNN_api.so: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o
src/api/FRCNN/libFRCNN_api.so: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/build.make
src/api/FRCNN/libFRCNN_api.so: lib/libcaffe.so.1.0.0
src/api/FRCNN/libFRCNN_api.so: lib/libcaffeproto.a
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libpthread.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libglog.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libgflags.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libsz.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libz.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libdl.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libm.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libpthread.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libglog.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libgflags.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libsz.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libz.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libdl.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libm.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/lib/libprotobuf.a
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/liblmdb.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libleveldb.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libcudart.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libcurand.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libcublas.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libcudnn.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/lib/libopencv_highgui.so.2.4.13
src/api/FRCNN/libFRCNN_api.so: /usr/local/lib/libopencv_imgproc.so.2.4.13
src/api/FRCNN/libFRCNN_api.so: /usr/local/lib/libopencv_core.so.2.4.13
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libcudart.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libnppc.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libnppi.so
src/api/FRCNN/libFRCNN_api.so: /usr/local/cuda/lib64/libnpps.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/liblapack.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/libcblas.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/libatlas.so
src/api/FRCNN/libFRCNN_api.so: /usr/lib/x86_64-linux-gnu/libboost_python.so
src/api/FRCNN/libFRCNN_api.so: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libFRCNN_api.so"
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FRCNN_api.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/api/FRCNN/CMakeFiles/FRCNN_api.dir/build: src/api/FRCNN/libFRCNN_api.so

.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/build

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/requires: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/frcnn_api.cpp.o.requires
src/api/FRCNN/CMakeFiles/FRCNN_api.dir/requires: src/api/FRCNN/CMakeFiles/FRCNN_api.dir/rpn_api.cpp.o.requires

.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/requires

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/clean:
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN && $(CMAKE_COMMAND) -P CMakeFiles/FRCNN_api.dir/cmake_clean.cmake
.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/clean

src/api/FRCNN/CMakeFiles/FRCNN_api.dir/depend:
	cd /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kakadinh/caffe_variances/Caffe_with_updated_functions /home/kakadinh/caffe_variances/Caffe_with_updated_functions/src/api/FRCNN /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN /home/kakadinh/caffe_variances/Caffe_with_updated_functions/build/src/api/FRCNN/CMakeFiles/FRCNN_api.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/api/FRCNN/CMakeFiles/FRCNN_api.dir/depend
