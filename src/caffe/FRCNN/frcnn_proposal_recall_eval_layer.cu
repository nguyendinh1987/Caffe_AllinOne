// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Dinh Nguyen Van
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_recall_eval_layer.hpp"

namespace caffe {
namespace Frcnn {    
    template <typename Dtype>
    void RecallEvalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
	this->Forward_cpu(bottom, top);
    }
    template <typename Dtype>
    void RecallEvalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
	NOT_IMPLEMENTED;
    }

INSTANTIATE_LAYER_GPU_FUNCS(RecallEvalLayer);
}// end of Frcnn namespace
}// end of caffe namespace
