#ifndef CAFFE_SSD_DATA_TRANSFORMER_HPP
#define CAFFE_SSD_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/SSD/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/SSD/bbox_util.hpp"
#include "caffe/SSD/im_transforms.hpp"

#include "google/protobuf/repeated_field.h"
using google::protobuf::RepeatedPtrField;

namespace caffe {
/**
 * @brief Applies from-SSD transformations to the input data, such as
 * <Need to fill>
 */
template <typename Dtype>
class SSDDataTransformer : public DataTransformer<Dtype> {
    public:       
        /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to the annotated data.
     *
     * @param anno_datum
     *    AnnotatedDatum containing the data and annotation to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
     * @param transformed_anno_vec
     *    This is destination annotation.
     */
    explicit SSDDataTransformer(const TransformationParameter& param, Phase phase);
    virtual ~SSDDataTransformer(){}

    /* Original from DataTransformer*/
    // void InitRand();
    // Overided functions
    void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
    void Transform(const Datum& datum, Dtype* transformed_data);
    vector<int> InferBlobShape(const Datum& datum);
#ifdef USE_OPENCV
    void Transform(const cv::Mat& cv_img,Blob<Dtype>* transformed_blob);
    vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif // USE_OPENCV
    //--------------------------------
    void Transform(const AnnotatedDatum& anno_datum,
                    Blob<Dtype>* transformed_blob,
                    RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);//Done
    void Transform(const AnnotatedDatum& anno_datum,
                    Blob<Dtype>* transformed_blob,
                    RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                    bool* do_mirror);//Done
    void Transform(const AnnotatedDatum& anno_datum,
                    Blob<Dtype>* transformed_blob,
                    vector<AnnotationGroup>* transformed_anno_vec,
                    bool* do_mirror);//Done
    void Transform(const AnnotatedDatum& anno_datum,
                    Blob<Dtype>* transformed_blob,
                    vector<AnnotationGroup>* transformed_anno_vec);//Done

    /**
     * @brief Transform the annotation according to the transformation applied
     * to the datum.
     *
     * @param anno_datum
     *    AnnotatedDatum containing the data and annotation to be transformed.
     * @param do_resize
     *    If true, resize the annotation accordingly before crop.
     * @param crop_bbox
     *    The cropped region applied to anno_datum.datum()
     * @param do_mirror
     *    If true, meaning the datum has mirrored.
     * @param transformed_anno_group_all
     *    Stores all transformed AnnotationGroup.
     */
    void TransformAnnotation(
        const AnnotatedDatum& anno_datum, const bool do_resize,
        const NormalizedBBox& crop_bbox, const bool do_mirror,
        RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);//Done

    /**
     * @brief Crops the datum according to bbox.
     */
    void CropImage(const Datum& datum, const NormalizedBBox& bbox,
                    Datum* crop_datum);//Done

    /**
     * @brief Crops the datum and AnnotationGroup according to bbox.
     */
    void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                    AnnotatedDatum* cropped_anno_datum);//Done

    /**
     * @brief Expand the datum.
     */
    void ExpandImage(const Datum& datum, const float expand_ratio,
                    NormalizedBBox* expand_bbox, Datum* expanded_datum);//Done

    /**
     * @brief Expand the datum and adjust AnnotationGroup.
     */
    void ExpandImage(const AnnotatedDatum& anno_datum,
                    AnnotatedDatum* expanded_anno_datum);//Done

    /**
     * @brief Apply distortion to the datum.
     */
    void DistortImage(const Datum& datum, Datum* distort_datum);//Done
#ifdef USE_OPENCV
    void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
                    NormalizedBBox* crop_bbox, bool* do_mirror);//Done
    /**
    * @brief Crops img according to bbox.
    */
    void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
                    cv::Mat* crop_img);//Done

    /**
     * @brief Expand img to include mean value as background.
     */
    void ExpandImage(const cv::Mat& img, const float expand_ratio,
                    NormalizedBBox* expand_bbox, cv::Mat* expand_img); //Done

    void TransformInv(const Blob<Dtype>* blob, vector<cv::Mat>* cv_imgs); //Done
    void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
                        const int width, const int channels);//Done
#endif //USE_OPENCV

    protected:
    // Transform and return the transformation information.
    //Done
    void Transform(const Datum& datum, Dtype* transformed_data,
                    NormalizedBBox* crop_bbox, bool* do_mirror);
    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to the data and return transform information.
     */
    //Done
    void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                    NormalizedBBox* crop_bbox, bool* do_mirror);
    int Rand(int n);
//    TransformationParameter param_;                    
//    Phase phase_;
};

} // end namespace
#endif
