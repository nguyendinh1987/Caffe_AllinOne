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

pid=$$

BUILD=/home/kakadinh/caffe_variances/my_modify_caffe/caffe/build/examples/FRCNN/test_rpn.bin

time $BUILD --gpu $gpu \
    --model /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/zf_rpn/test.prototxt \
    --weights /home/kakadinh/caffe_variances/my_modify_caffe/caffe/models/FRCNN/zf_rpn_final.caffemodel \
    --default_c /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/config/voc_config.json \
    --image_root VOCdevkit/VOC2007/JPEGImages/ \
    --image_list /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/dataset/voc2007.test \
    --out_file /home/kakadinh/caffe_variances/my_modify_caffe/caffe/examples/FRCNN/results/voc2007_test_${pid}.rpn

CAL_RECALL=examples/FRCNN/calculate_recall.py

time python $CAL_RECALL --gt examples/FRCNN/dataset/voc2007.test \
    --answer examples/FRCNN/results/voc2007_test_${pid}.rpn \
    --overlap 0.5
