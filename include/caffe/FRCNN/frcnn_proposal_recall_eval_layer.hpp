// ------------------------------------------------------------------
// Dinh, Nguyen Van
// 2018/12/04
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PROPOSAL_RECALL_EVAL_LAYER_HPP_
#define CAFFE_FRCNN_PROPOSAL_RECALL_EVAL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
class RecallEvalLayer : public Layer<Dtype> {
public:
	explicit RecallEvalLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "RecallEval"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        vector<float> ious_;
	vector<int> npps_;
};

} // end of Frcnn namespace
} // end of caffe namespace
#endif  // CAFFE_FRCNN_PROPOSAL_RECALL_EVAL_LAYER_HPP_
