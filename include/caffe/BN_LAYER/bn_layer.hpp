#ifndef CAFFE_COMMON_LAYERS_HPP_
#define CAFFE_COMMON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
//#include "caffe/BN_LAYER/data_layers.hpp"
#include "caffe/layer.hpp"
//#include "caffe/BN_LAYER/loss_layers.hpp"
//#include "caffe/BN_LAYER/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"

#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {
/**
 * @brief Batch normalization the input blob along the channel axis while
 *        averaging over the spatial axes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BNLayer : public Layer<Dtype> {
 public:
  explicit BNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void AverageAllExceptChannel(const Dtype* input, Dtype* output);
  void BroadcastChannel(const Dtype* input, Dtype* output);

  bool frozen_;
  Dtype bn_momentum_;
  Dtype bn_eps_;

  int num_;
  int channels_;
  int height_;
  int width_;

  Blob<Dtype> broadcast_buffer_;
  Blob<Dtype> spatial_statistic_;
  Blob<Dtype> batch_statistic_;

  Blob<Dtype> x_norm_;
  Blob<Dtype> x_inv_std_;

  Blob<Dtype> spatial_sum_multiplier_;
  Blob<Dtype> batch_sum_multiplier_;
};


#if defined(USE_CUDNN)
#if CUDNN_VERSION_MIN(5, 0, 0)
/**
 * @brief cuDNN implementation of BNLayer.
 *        Fallback to BNLayer for CPU mode.
 */
template <typename Dtype>
class CuDNNBNLayer : public BNLayer<Dtype> {
 public:
  explicit CuDNNBNLayer(const LayerParameter& param)
      : BNLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNBNLayer();

  virtual inline const char* type() const { return "BN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;

  Blob<Dtype> save_mean_;
  Blob<Dtype> save_inv_variance_;
};
#endif
#endif

#ifdef USE_MPI
template <typename Dtype>
class SyncBNLayer : public Layer<Dtype> {
 public:
  explicit SyncBNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SyncBN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype bn_momentum_;
  Dtype bn_eps_;

  int num_;
  int channels_;
  int height_;
  int width_;

  Blob<Dtype> mean_buffer_;
  Blob<Dtype> var_buffer_;
};
#endif

/**
* @brief Normalizes input to unit-length vector
*/
template <typename Dtype>
class NormalizeLayer : public Layer<Dtype> {
public:
    explicit NormalizeLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Normalize"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> sum_multiplier_, norm_, squared_;
};


template <typename Dtype>
class ScaleLayer: public Layer<Dtype> {
public:
    explicit ScaleLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Scale"; }
    // Scale
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    /**
     * In the below shape specifications, @f$ i @f$ denotes the value of the
     * `axis` field given by `this->layer_param_.scale_param().axis()`, after
     * canonicalization (i.e., conversion from negative to positive index,
     * if applicable).
     *
     * @param bottom input Blob vector (length 2)
     *   -# @f$ (d_0 \times ... \times
     *           d_i \times ... \times d_j \times ... \times d_n) @f$
     *      the first factor @f$ x @f$
     *   -# @f$ (d_i \times ... \times d_j) @f$
     *      the second factor @f$ y @f$
     * @param top output Blob vector (length 1)
     *   -# @f$ (d_0 \times ... \times
     *           d_i \times ... \times d_j \times ... \times d_n) @f$
     *      the product @f$ z = x y @f$ computed after "broadcasting" y.
     *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
     *      then computing the elementwise product.
     */
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    shared_ptr<Layer<Dtype> > bias_layer_;
    vector<Blob<Dtype>*> bias_bottom_vec_;
    vector<bool> bias_propagate_down_;
    int bias_param_id_;

    Blob<Dtype> sum_multiplier_;
    Blob<Dtype> sum_result_;
    Blob<Dtype> temp_;
    int axis_;
    int outer_dim_, scale_dim_, inner_dim_;
};

/**
 * @brief Computes a sum of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        sum.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer.
 */
template <typename Dtype>
class BiasLayer : public Layer<Dtype> {
public:
    explicit BiasLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Bias"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    Blob<Dtype> bias_multiplier_;
    int outer_dim_, bias_dim_, inner_dim_, dim_;
};


/**
 * @brief Compute "reductions" -- operations that return a scalar output Blob
 *        for an input Blob of arbitrary size, such as the sum, absolute sum,
 *        and sum of squares.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BatchReductionLayer : public Layer<Dtype> {
public:
    explicit BatchReductionLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "BatchReduction"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /// @brief the reduction operation performed by the layer
    ReductionParameter_ReductionOp op_;
    /// @brief the index of the first input axis to reduce
    int axis_;
    /// @brief the number of reductions performed
    int num_;
    /// @brief the step of reduction
    int step_;
    /// @brief whether to perform position sensitive learning
    bool pos_;
    /// @brief a helper Blob used for transferring ticks to GPU
    Blob<Dtype> ticks_blob_;
    vector<int> levels_;
    vector<int> ticks_;
    int max_tick_;

    Blob<Dtype> argsort_idx_;
};
}  // namespace caffe

#endif // CAFFE_COMMON_LAYERS_HPP_
