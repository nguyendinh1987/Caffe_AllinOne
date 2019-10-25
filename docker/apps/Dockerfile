FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

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

WORKDIR /opt
RUN git clone https://github.com/nguyendinh1987/Caffe_AllinOne.git
RUN cd Caffe_AllinOne && \
       mkdir build && \
       cd build && \
       cmake .. &&\
       make -j20 && \
       make install

ENV CAFFE_ROOT=/opt/va/Caffe_AllinOne/build/install
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $PYCAFFE_ROOT/bin:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# vnc
RUN    apt-get install x11vnc xvfb firefox \
	&& apt-get autoclean -y \
    	&& apt-get autoremove -y \
	&& rm -rf /var/lib/apt/lists/*
# Setup a password
RUN     x11vnc -storepasswd 1234 root/.vnc/passwd

