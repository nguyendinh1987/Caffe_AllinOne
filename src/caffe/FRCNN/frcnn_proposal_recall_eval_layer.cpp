// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Dinh Nguyen Van
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_recall_eval_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include <typeinfo>

namespace caffe {
namespace Frcnn {    
    template <typename Dtype>
    void RecallEvalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
            ious_.clear();
            npps_.clear();
            if (this->layer_param_.has_recall_based_eval_param()){
                RecallEvalParameter recall_eval_param = this->layer_param_.recall_based_eval_param();
                for (int i = 0; i < recall_eval_param.ious_size(); i++){
                    ious_.push_back(recall_eval_param.ious(i));
                }
                for (int i = 0; i < recall_eval_param.npp_size(); i++){
                    npps_.push_back(recall_eval_param.npp(i));
                }
            }else{
                ious_.push_back(0.5);
                npps_.push_back(100);
            }
            for (int i = 0; i < ious_.size(); i++){
                CHECK_GT(ious_[i],0.0) << ious_[i] << " illegal iou threshold. It has to be greater than 0";
                CHECK_LT(ious_[i],1.0) << ious_[i] << " illegal iou threshold. It has to be less than 1";
            }
            for (int i = 0; i < npps_.size(); i++){
                CHECK_GT(npps_[i],0) << ious_[i] << " illegal number of proposals. It has to be greater than 0";
            }
        }

	template <typename Dtype>
	void RecallEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
			top[0]->Reshape(1,1,npps_.size(),ious_.size());
	}

    template <typename Dtype>
	void RecallEvalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
            // bottom [0]: rpn_bbox_pred
            // bottom [1]: rpn_cls_prob_reshape
            // bottom [2]: gt_boxes
            // bottom [3]: im_info
            // const Dtype* bottom_gt = bottom[2]->cpu_data();
			// const Dtype* bottom_ppb = bottom[0]->cpu_data();
            const Dtype *bottom_rpn_bbox = bottom[0]->cpu_data();            
            const Dtype *bottom_rpn_score = bottom[1]->cpu_data();
            const Dtype *bottom_im_info = bottom[3]->cpu_data();

            const float im_height = bottom_im_info[0];
            const float im_width = bottom_im_info[1];

            CHECK(bottom[0]->num() == 1) << "only single item batches are supported";

            DLOG(ERROR) << "========== get gt boxes : " << bottom[1]->num();
            vector<Point4f<Dtype> > gt_boxes;
            for (int i = 0; i < bottom[2]->num(); i++) {
                gt_boxes.push_back(Point4f<Dtype>(
                bottom[2]->data_at(i, 0, 0, 0),
                bottom[2]->data_at(i, 1, 0, 0),
                bottom[2]->data_at(i, 2, 0, 0),
                bottom[2]->data_at(i, 3, 0, 0)));
                CHECK(gt_boxes[i][0]>=0 && gt_boxes[i][1]>=0);
                CHECK(gt_boxes[i][2]<=im_width && gt_boxes[i][3]<=im_height);
                DLOG(ERROR) << "============= " << i << "  : " << gt_boxes[i][0] << ", " << gt_boxes[i][1] << ", " << gt_boxes[i][2] << ", " << gt_boxes[i][3];
            }

            DLOG(ERROR) << "========== get proposal boxes : " << bottom[0]->channels();
            const int config_n_anchors = FrcnnParam::anchors.size() / 4;
            const int channes = bottom[0]->channels();
            const int height = bottom[0]->height();
            const int width = bottom[0]->width();
            typedef pair<Dtype, int> sort_pair;
            std::vector<sort_pair> sort_vector;
            std::vector<Point4f<Dtype> > anchors;

            int rpn_min_size;
            if (this->phase_ == TRAIN) {
                rpn_min_size = FrcnnParam::rpn_min_size;
            } else {
                rpn_min_size = FrcnnParam::test_rpn_min_size;
            }
            const Dtype min_size = bottom_im_info[2] * rpn_min_size;

            const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height -1 };

            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    for (int k = 0; k < config_n_anchors; k++) {
                        // Get ppb score
                        Dtype score = bottom_rpn_score[config_n_anchors * height * width + k * height * width + j * width + i];
                        // Get anchor
                        Point4f<Dtype> anchor(
                            FrcnnParam::anchors[k * 4 + 0] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
                            FrcnnParam::anchors[k * 4 + 1] + j * FrcnnParam::feat_stride,  // shift_y[i][j];
                            FrcnnParam::anchors[k * 4 + 2] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
                            FrcnnParam::anchors[k * 4 + 3] + j * FrcnnParam::feat_stride); // shift_y[i][j];
                        // Get bbox delta
                        Point4f<Dtype> box_delta(
                            bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
                            bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
                            bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
                            bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i]);
                
                        Point4f<Dtype> cbox = bbox_transform_inv(anchor, box_delta);
                        
                        // Clip predicted boxes to image
                        for (int q = 0; q < 4; q++) {
                            cbox.Point[q] = std::max(Dtype(0), std::min(cbox[q], bounds[q]));
                        }

                        // Remove predicted boxes with either height or width < threshold
                        if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
                            const int now_index = sort_vector.size();
                            sort_vector.push_back(sort_pair(score, now_index)); 
                            anchors.push_back(cbox);
                        }
                    }
                }
            }

            DLOG(ERROR) << "========== after clip and remove size < threshold box " << (int)sort_vector.size();

            // Sort proposal boxes
            std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
            // LOG(INFO)<<"Before nms,"<<sort_vector.size()<<" proposal boxes are generated";
            // nms
            float rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
            std::vector<bool> select(2000, true);
            std::vector<Point4f<Dtype> > box_final;
            std::vector<Dtype> scores_;
            for (int i = 0; i < sort_vector.size() && i < 2000; i++) {
                if (select[i]) 
                {
                    const int cur_i = sort_vector[i].second;
                    for (int j = i + 1; j < sort_vector.size() && j < 2000; j++)
                        if (select[j]) 
                        {
                            const int cur_j = sort_vector[j].second;
                            if (get_iou(anchors[cur_i], anchors[cur_j]) > rpn_nms_thresh) 
                            {
                                select[j] = false;
                            }
                        }
                    box_final.push_back(anchors[cur_i]);
                    scores_.push_back(sort_vector[i].first);
                }
            }
            // LOG(INFO)<<"After nms,"<<box_final.size()<<" proposal boxes are generated";
            // Get n proposal boxes
            int max_n_anchors = std::min((int)box_final.size(), npps_[(int)npps_.size()-1]);
            box_final.erase(box_final.begin() + max_n_anchors, box_final.end());
            // LOG(INFO)<<sort_vector.size()<<" proposal boxes are kept for recall evaluation";
            // LOG(INFO)<<"Image has "<<gt_boxes.size()<<" gt boxes";
            // for (int i = 0; i<10; i++)
                // LOG(INFO)<<"index: "<<sort_vector[i].second<<"; score: "<<sort_vector[i].first;
            
            vector<Point4f<Dtype> > kept_anchors;
            for (int i = 0; i < max_n_anchors; i++){
                kept_anchors.push_back(box_final[i]);
            }

            vector<Dtype> max_overlaps(kept_anchors.size(), -1);
            vector<int> argmax_overlaps(kept_anchors.size(), -1);
            // vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
            // vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

            vector<vector<Dtype> > ious = get_ious(kept_anchors, gt_boxes);

            for (int ia = 0; ia < kept_anchors.size(); ia++) {
                for (size_t igt = 0; igt < gt_boxes.size(); igt++) {
                    if (ious[ia][igt] > max_overlaps[ia]) {
                        max_overlaps[ia] = ious[ia][igt];
                        argmax_overlaps[ia] = igt;
                    }
                }
            }

            vector<float> recall(npps_.size()*ious_.size(),0);

            for (int io = 0; io < ious_.size(); io++){
                float iou_thresh = ious_[io];
                for (int np = 0; np < npps_.size(); np++){
                    vector<int> check_gt(gt_boxes.size(),0);
                    int n_anchors = npps_[np];
                    for (int ia = 0; ia < n_anchors && ia < kept_anchors.size(); ia++) {
                        if (max_overlaps[ia] >= iou_thresh && check_gt[argmax_overlaps[ia]]==0){
                            check_gt[argmax_overlaps[ia]] = 1;
                        }
                    }
                    float n_detect_gt = 0;
                    for (int i = 0; i<gt_boxes.size(); i++) n_detect_gt += check_gt[i];
                    // LOG(INFO)<<"type of n_detect_gt: "<<typeid(n_detect_gt).name();
                    recall[np * ious_.size() + io] = n_detect_gt/check_gt.size();
                }
            }
            
            // for (int i = 0; i<recall.size(); i++) LOG(INFO)<<recall[i];
            // Copy to top layer
            Dtype* top_recall = top[0]->mutable_cpu_data();            
            caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
            for (int io = 0; io < ious_.size(); io++){
                for (int np = 0; np < npps_.size(); np++){
                    top_recall[ top[0]->offset(0,0,np,io) ] = recall[np * ious_.size() + io];
                }
            }


        }
    template <typename Dtype>
	void RecallEvalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
    }

#ifdef CPU_ONLY
	STUB_GPU(RecallEvalLayer);
#endif
INSTANTIATE_CLASS(RecallEvalLayer);
REGISTER_LAYER_CLASS(RecallEval);
}
}