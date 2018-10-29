#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

CAFFE=/home/kakadinh/caffe_variances/my_modify_caffe/caffe/build/tools/caffe 

time $CAFFE train   \
    --gpu $gpu \
    --solver /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/vgg16/solver.proto \
    --weights /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/VGG16.caffemodel

time python /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/convert_model.py \
    --model /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/vgg16/test.proto \
    --weights /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/snapshot/vgg16_faster_rcnn_iter_20000.caffemodel \
    --config /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/config/voc_config.json \
    --net_out /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/vgg16_faster_rcnn_final.caffemodel
