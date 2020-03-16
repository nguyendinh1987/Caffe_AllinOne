FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN     pip install --upgrade pip && \
	python -m pip install scikit-image protobuf
RUN     python -m pip install opencv-python==3.2.0.7

# Need to find a version that support ffmpeg to load videos/image
RUN     python -m pip install opencv-contrib-python==4.1.1.26

# add repository to source.list to install needed packages
RUN     echo "##### Added to install packages in ubuntu based - I got it from default ubuntu" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial main restricted" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial-updates main restricted" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial universe" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial-updates universe" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial multiverse" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial-updates multiverse" | tee -a /etc/apt/sources.list
RUN     echo "deb http://sg.archive.ubuntu.com/ubuntu/ xenial-backports main restricted universe multiverse" | tee -a /etc/apt/sources.list
RUN     echo "deb http://security.ubuntu.com/ubuntu xenial-security main restricted" | tee -a /etc/apt/sources.list
RUN     echo "deb http://security.ubuntu.com/ubuntu xenial-security universe" | tee -a /etc/apt/sources.list
RUN     echo "deb http://security.ubuntu.com/ubuntu xenial-security multiverse" | tee -a /etc/apt/sources.list

RUN     apt-get update \
# ffmpeg - dependence lib which opencv need to load video/image
	&& apt install -y software-properties-common ffmpeg \
# vnc and other softwares
        && apt-get install -y x11vnc xvfb firefox gedit \
# clean unnecessary files to reduce docker image size
       && apt-get autoclean -y \
       && apt-get autoremove -y \
       && rm -rf /var/lib/apt/lists/*
# install tkinter
RUN    apt-get update --fix-missing \
       && apt-get install -y python-tk
# Setup a password
RUN    mkdir /root/.vnc
RUN     x11vnc -storepasswd 1234 /root/.vnc/passwd

# install caffe
WORKDIR /opt
RUN git clone https://github.com/nguyendinh1987/Caffe_AllinOne.git
RUN cd Caffe_AllinOne && \
       mkdir build && \
       cd build && \
       cmake .. &&\
       make -j20 && \
       make install

# Or copy compiled libs
#WORKDIR /opt
#RUN mkdir Caffe_AllinOne \
#    && mkdir Caffe_AllinOne/build \
#    && mkdir Caffe_AllinOne/build/install
#COPY install Caffe_AllinOne/build/install 

ENV CAFFE_ROOT=/opt/Caffe_AllinOne/build/install
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/bin:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# setup fluxbox
## move this path into one install command
RUN     apt-get install -y fluxbox
WORKDIR /root/home
RUN     mkdir .startx
COPY entrypoint.sh .startx
ENTRYPOINT [".startx/entrypoint.sh"]
