#ifndef CAFFE_ANOTATEDDATA_LAYER_HPP_
#define CAFFE_ANOTATEDDATA_LAYER_HPP_

#include <string>
#include <vector>
#include <boost/thread.hpp>

#include "caffe/blob.hpp"
#include "caffe/SSD/data_reader.hpp"
#include "caffe/SSD/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class AnnotatedDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit AnnotatedDataLayer(const LayerParameter& param);
  virtual ~AnnotatedDataLayer();

  // because I need to initialize ssd_data_transformer which is not exist in BasePrefetchingDataLayer
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // AnnotatedDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "AnnotatedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

  // From BasePrefetchingDataLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader<AnnotatedDatum> reader_;
  bool has_anno_type_;
  AnnotatedDatum_AnnotationType anno_type_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;
  shared_ptr<SSDDataTransformer<Dtype> > ssd_data_transformer_;

  // From BasePrefetchingDataLayer
  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  virtual void InternalThreadEntry();
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Blob<Dtype> transformed_data_;

};

}  // namespace caffe

#endif // CAFFE_DATA_LAYER_HPP_
