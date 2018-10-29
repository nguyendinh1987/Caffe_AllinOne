#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

BUILD=/home/kakadinh/caffe_variances/my_modify_caffe/caffe/build/examples/FRCNN/demo_frcnn_api.bin

$BUILD --gpu $gpu \
       --model /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/vgg16/test.proto \
       --weights /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/vgg16_faster_rcnn_final.caffemodel \
       --default_c /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/config/voc_config.json \
       --image_dir /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/images/  \
       --out_dir /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/results/ 
