#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <iostream>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// Source code:
// (1) https://github.com/yjxiong/caffe for DSN-TSN network


namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  // Dinh-------------
  // check if we want to use normalize_input
  if (param_.normalize_input() || param_.on_fly_mean()) {
    if (param_.mean_value_size() > 0){
      LOG(INFO)<<"The fixed mean values will be changed. Need to turn false for"
               <<" 'normalize_input' or 'on_fly_mean'"
               <<" if you want to use your mean values";
      LOG(INFO)<<" Press any key to continue";
      getchar();
    }
    if (mean_values_.size()==0)
      mean_values_.push_back(0);//To activate has_mean_value variable
  }
  //------------------------
  //From (1)
  //////////
  //load multiscale info
  max_distort_ = param_.max_distort();
  custom_scale_ratios_.clear();
  for (int i = 0; i < param_.scale_ratios_size(); ++i){
    custom_scale_ratios_.push_back(param_.scale_ratios(i));
  }
  org_size_proc_ = param.original_image();
}
// From (1)
///////////
  /** @build fixed crop offsets for random selection
  */
  void fillFixOffset(int datum_height, int datum_width, int crop_height, int crop_width,
                    bool more_crop,
                    vector<pair<int , int> >& offsets){
    int height_off = (datum_height - crop_height)/4;
    int width_off = (datum_width - crop_width)/4;

    offsets.clear();
    offsets.push_back(pair<int, int>(0, 0)); //upper left
    offsets.push_back(pair<int, int>(0, 4 * width_off)); //upper right
    offsets.push_back(pair<int, int>(4 * height_off, 0)); //lower left
    offsets.push_back(pair<int, int>(4 * height_off, 4 *width_off)); //lower right
    offsets.push_back(pair<int, int>(2 * height_off, 2 * width_off)); //center

    //will be used when more_fix_crop is set to true
    if (more_crop) {
      offsets.push_back(pair<int, int>(0, 2 * width_off)); //top center
      offsets.push_back(pair<int, int>(4 * height_off, 2 * width_off)); //bottom center
      offsets.push_back(pair<int, int>(2 * height_off, 0)); //left center
      offsets.push_back(pair<int, int>(2 * height_off, 4 * width_off)); //right center

      offsets.push_back(pair<int, int>(1 * height_off, 1 * width_off)); //upper left quarter
      offsets.push_back(pair<int, int>(1 * height_off, 3 * width_off)); //upper right quarter
      offsets.push_back(pair<int, int>(3 * height_off, 1 * width_off)); //lower left quarter
      offsets.push_back(pair<int, int>(3 * height_off, 3 * width_off)); //lower right quarter
    }
  }
  float _scale_rates[] = {1.0, .875, .75, .66};
  vector<float> default_scale_rates(_scale_rates, _scale_rates + sizeof(_scale_rates)/ sizeof(_scale_rates[0]) );
  /**
 * @generate crop size when multi-scale cropping is requested
 */
  void fillCropSize(int input_height, int input_width,
                  int net_input_height, int net_input_width,
                  vector<pair<int, int> >& crop_sizes,
                  int max_distort, vector<float>& custom_scale_ratios)
  {
      crop_sizes.clear();
      vector<float>& scale_rates = (custom_scale_ratios.size() > 0)?custom_scale_ratios:default_scale_rates;
      int base_size = std::min(input_height, input_width);
      for (int h = 0; h < scale_rates.size(); ++h)
      {
        int crop_h = int(base_size * scale_rates[h]);
        crop_h = (abs(crop_h - net_input_height) < 3)?net_input_height:crop_h;
        for (int w = 0; w < scale_rates.size(); ++w)
        {
          int crop_w = int(base_size * scale_rates[w]);
          crop_w = (abs(crop_w - net_input_width) < 3)?net_input_width:crop_w;

          //append this cropping size into the list
          if (abs(h-w)<=max_distort) 
          {
            crop_sizes.push_back(pair<int, int>(crop_h, crop_w));
          }
        }
      }
  }
  /**
 * @generate crop size and offset when process original images
 */
  void sampleRandomCropSize(int img_height, int img_width,
                            int& crop_height, int& crop_width,
                            float min_scale=0.08, float max_scale=1.0, float min_as=0.75, float max_as=1.33)
  {

    float total_area = img_height * img_width;
    float area_ratio = 0;
    float target_area = 0;
    float aspect_ratio = 0;
    float flip_coin = 0;

    int attempt = 0;

    while (attempt < 10) 
    {
      // sample scale and area
      caffe_rng_uniform(1, min_scale, max_scale, &area_ratio);
      target_area = total_area * area_ratio;

      caffe_rng_uniform(1, float(0), float(1), &flip_coin);
      if (flip_coin > 0.5)
      {
          std::swap(crop_height, crop_width);
      }

      // sample aspect ratio
      caffe_rng_uniform(1, min_as, max_as, &aspect_ratio);
      crop_height = int(sqrt(target_area / aspect_ratio));
      crop_width = int(sqrt(target_area * aspect_ratio));

      if (crop_height <= img_height && crop_width <= img_width)
      {
        return;
      }
      attempt ++;
    }

    // fallback to normal 256-224 style size crop
    crop_height = img_height / 8 * 7;
    crop_width = img_width / 8 * 7;
  }
////////////
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;
  //From (1)
  //////////
    const bool do_multi_scale = param_.multi_scale();
    vector<pair<int, int> > offset_pairs;
    vector<pair<int, int> > crop_size_pairs;
    cv::Mat multi_scale_bufferM;
  //////////

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    // From (1): To avoid a case of a number of the mean value list is not equal a number of input channel
    ////////////
      if (mean_values_.size() < datum_channels)
      {
        // Replicate the mean_value group to fill up the datum channels
        size_t group_size = mean_values_.size();
        for (size_t c = group_size; c < datum_channels; ++c)
        {
          mean_values_.push_back(mean_values_[c % group_size]);
        }
      }
    ////////////
  }
  // From (1)
  ///////////
    if (!crop_size && do_multi_scale)
    {
      LOG(ERROR)<< "Multi scale augmentation is only activated with crop_size set.";
    }
  ///////////

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  // From (1)
  ///////////
    int crop_height = 0;
    int crop_width = 0;
    bool need_imgproc = false;
  ///////////
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      // From (1)
      ///////////
      // If in training and we need multi-scale cropping, reset the crop size params
        if (do_multi_scale)
        {
          fillCropSize(datum_height, datum_width, crop_size, crop_size, crop_size_pairs,
                      max_distort_, custom_scale_ratios_);
          int sel = Rand(crop_size_pairs.size());
          crop_height = crop_size_pairs[sel].first;
          crop_width = crop_size_pairs[sel].second;
        }
        else
        {
          crop_height = crop_size;
          crop_width = crop_size;
        }
        if (param_.fix_crop())
        {
          fillFixOffset(datum_height, datum_width, crop_height, crop_width,
                        param_.more_fix_crop(), offset_pairs);
          int sel = Rand(offset_pairs.size());
          h_off = offset_pairs[sel].first;
          w_off = offset_pairs[sel].second;
        }
        else
        {
      ///////////
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
        }
    } else {
      // From (1)
      ///////////
        crop_height = crop_size;
        crop_width = crop_size;
      ///////////
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }
  // From (1)
  ///////////
    need_imgproc = do_multi_scale && crop_size && ((crop_height != crop_size) || (crop_width != crop_size));
  ///////////

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    // From (1)
    ///////////
      // image resize etc needed
      if (need_imgproc)
      {
        cv::Mat M(datum_height, datum_width, has_uint8?CV_8UC1:CV_32FC1);

        //put the datum content to a cvMat
        for (int h = 0; h < datum_height; ++h) 
        {
          for (int w = 0; w < datum_width; ++w) {
            int data_index = (c * datum_height + h) * datum_width + w;
            if (has_uint8) {
              M.at<uchar>(h, w) = static_cast<uint8_t>(data[data_index]);
            }else{
              M.at<float>(h, w) = datum.float_data(data_index);
            }
          }
        }

        //resize the cropped patch to network input size
        cv::Mat cropM(M, cv::Rect(w_off, h_off, crop_width, crop_height));
        cv::resize(cropM, multi_scale_bufferM, cv::Size(crop_size, crop_size));
        cropM.release();
      }
    ///////////
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // From (1)
        ///////////
          if (need_imgproc){
            if (has_uint8){
              if (param_.is_flow() && do_mirror && c%2 == 0)
                datum_element = 255 - static_cast<Dtype>(multi_scale_bufferM.at<uint8_t>(h, w));
              else
                datum_element = static_cast<Dtype>(multi_scale_bufferM.at<uint8_t>(h, w));
            }else {
              if (param_.is_flow() && do_mirror && c%2 == 0)
                datum_element = 255 - static_cast<Dtype>(multi_scale_bufferM.at<float>(h, w));
              else
                datum_element = static_cast<Dtype>(multi_scale_bufferM.at<float>(h, w));
            }
          }
          else
          {
        ///////////
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
        // From (1)
        ///////////
          }
          if (has_mean_file) 
          {
            if (do_multi_scale) 
            {
              int fixed_data_index = (c * datum_height +  h) * datum_width + w;
              transformed_data[top_index] =
                  (datum_element - mean[fixed_data_index]) * scale;
            }
            else
            {
              transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
            }
          } 
          else 
          {
            if (has_mean_values) 
            {
              transformed_data[top_index] =
                (datum_element - mean_values_[c]) * scale;
            }
            else 
            {
              transformed_data[top_index] = datum_element * scale;
            }
          }
        ///////////
      }
    }
  }
  // From (1)
  ///////////
    multi_scale_bufferM.release();
  ///////////
}
// From (1): New function?????
/////////////////////////
  template<typename Dtype>
  void DataTransformer<Dtype>::Transform(const Datum& datum_data, const Datum& datum_label, 
                                        Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) 
  {
    CHECK_EQ(datum_data.height(), datum_label.height());
    CHECK_EQ(datum_data.width(), datum_label.width());

    const string& data = datum_data.data();
    const string& label = datum_label.data();
    const int datum_channels = datum_data.channels();
    const int datum_height = datum_data.height();
    const int datum_width = datum_data.width();

    float lower_scale = 1, upper_scale = 1;
    if (param_.scale_ratios_size() == 2)
    {
      lower_scale = param_.scale_ratios(0);
      upper_scale = param_.scale_ratios(1);
    }
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    const bool has_mean_values = mean_values_.size() > 0;
    const int stride = param_.stride();

    CHECK_GT(datum_channels, 0);

    if (has_mean_file) {
      NOT_IMPLEMENTED;
    }
    if (has_mean_values) {
      CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
      "Specify either 1 mean_value or as many as channels: " << datum_channels;
      if (datum_channels > 1 && mean_values_.size() == 1) {
        // Replicate the mean_value for simplicity
        for (int c = 1; c < datum_channels; ++c) {
          mean_values_.push_back(mean_values_[0]);
        }
      }
    }

    float scale_ratios = std::max(Rand(int((upper_scale - lower_scale) * 1000.0) + 1) / 1000.0, 0.0) + lower_scale;

    int height = int(datum_height * scale_ratios + 0.5);
    int width = int(datum_width * scale_ratios + 0.5);


    int crop_height = height / stride * stride;
    int crop_width = width / stride * stride;

    if (param_.has_upper_size())
    {
      crop_height = std::min(crop_height, param_.upper_size());
      crop_width = std::min(crop_width, param_.upper_size());
    }
    else if (param_.has_upper_height() && param_.has_upper_width())
    {
      crop_height = std::min(crop_height, param_.upper_height());
      crop_width = std::min(crop_width, param_.upper_width());
    }


    int h_off = Rand(height - crop_height + 1);
    int w_off = Rand(width - crop_width + 1);

    transformed_data->Reshape(1, datum_channels, crop_height, crop_width);
    transformed_label->Reshape(1, 1, crop_height, crop_width);

    //for image data
    
    Dtype datum_element;
    int top_index;
    Dtype* ptr = transformed_data->mutable_cpu_data();
    for (int c = 0; c < datum_channels; ++c) {
      cv::Mat M(datum_height, datum_width, CV_8UC1);
      for (int h = 0; h < datum_height; ++h)
        for (int w = 0; w < datum_width; ++w) 
        {
          int data_index = (c * datum_height + h) * datum_width + w;
          M.at<uchar>(h, w) = static_cast<uint8_t>(data[data_index]);
        }
      cv::resize(M, M, cv::Size(width, height));
      cv::Mat cropM(M, cv::Rect(w_off, h_off, crop_width, crop_height));
      for (int h = 0; h < crop_height; ++h)
        for (int w = 0; w < crop_width; ++w)
        {

          if (do_mirror) 
            top_index = (c * crop_height + h) * crop_width + (crop_width - 1 - w);
          else 
            top_index = (c * crop_height + h) * crop_width + w;

          datum_element = static_cast<Dtype>(cropM.at<uint8_t>(h, w));
          if (has_mean_file) 
          {
              NOT_IMPLEMENTED;
          } 
          else if (has_mean_values) 
            ptr[top_index] =(datum_element - mean_values_[c]) * scale;
          else 
            ptr[top_index] = datum_element * scale;
        }
      M.release();
      cropM.release();
    }

    //for label

    ptr = transformed_label->mutable_cpu_data();
    cv::Mat M(datum_height, datum_width, CV_8UC1);
    for (int h = 0; h < datum_height; ++h)
      for (int w = 0; w < datum_width; ++w)
      {
        int data_index = h * datum_width + w;
        M.at<uchar>(h, w) = static_cast<uint8_t>(label[data_index]);
      }
    cv::resize(M, M, cv::Size(width, height), 0, 0, CV_INTER_NN);
    cv::Mat cropM(M, cv::Rect(w_off, h_off, crop_width, crop_height));
    for (int h = 0; h < crop_height; ++h)
      for (int w = 0; w < crop_width; ++w) 
      {

        if (do_mirror) 
          top_index = h * crop_width + (crop_width - 1 - w);
        else 
          top_index = h * crop_width + w;

        ptr[top_index] = static_cast<Dtype>(cropM.at<uint8_t>(h, w));
      }
    M.release();
    cropM.release();
  }
/////////////////////////

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decode and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}
// //data augmentation on-the-fly
// template<typename Dtype>
// void DataTransformer<Dtype>::Transform(const cv::Mat& img,
//                                        Blob<Dtype>* transformed_blob) {
//   cv::Mat cv_img;
//   img.copyTo(cv_img);									   
//   const int crop_size = param_.crop_size();
//   const bool display = param_.display();
//   const bool contrast_adjustment = param_.contrast_adjustment();
//   const bool smooth_filtering = param_.smooth_filtering();
//   const bool jpeg_compression = param_.jpeg_compression();


//   const int img_channels = cv_img.channels();
//   const int img_height = cv_img.rows;
//   const int img_width = cv_img.cols;

//   // Check dimensions.
//   const int channels = transformed_blob->channels();
//   const int height = transformed_blob->height();
//   const int width = transformed_blob->width();
//   const int num = transformed_blob->num();

//   CHECK_EQ(channels, img_channels);
//   CHECK_LE(height, img_height);
//   CHECK_LE(width, img_width);
//   CHECK_GE(num, 1);

//   CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

//   const Dtype scale = param_.scale();
//   //const bool do_mirror = param_.mirror() && Rand(2);
//   const bool has_mean_file = param_.has_mean_file();
//   const bool has_mean_values = mean_values_.size() > 0;

//   CHECK_GT(img_channels, 0);
//   CHECK_GE(img_height, crop_size);
//   CHECK_GE(img_width, crop_size);

//   // param for rotation
//   const float rotation_angle_interval = param_.rotation_angle_interval();


//   if (display && phase_ == TRAIN)
// 	  cv::imshow("Source", cv_img);

//   // Flipping and Reflection -----------------------------------------------------------------
// 	int flipping_mode = (Rand(4)) - 1; // -1, 0, 1, 2
// 	bool apply_flipping = (flipping_mode != 2);
// 	if (apply_flipping) {
// 		cv::flip(cv_img,cv_img,flipping_mode);
// 		if (display && phase_ == TRAIN)
// 			cv::imshow("Flipping and Reflection", cv_img);
// 	}


//   // Smooth Filtering -------------------------------------------------------------
//   int smooth_param1 = 3;
//   int apply_smooth = Rand(2);
//   if ( smooth_filtering && apply_smooth ) {
// 	int smooth_type = Rand(4); // see opencv_util.hpp
// 	smooth_param1 = 3 + 2*(Rand(1));
//         switch(smooth_type){
//         case 0:
// 	   //cv::Smooth(cv_img, cv_img, smooth_type, smooth_param1);
// 	   cv::GaussianBlur(cv_img, cv_img, cv::Size(smooth_param1,smooth_param1),0);
//            break;
//         case 1:
//            cv::blur(cv_img, cv_img, cv::Size(smooth_param1,smooth_param1));
//            break;
//         case 2:
//            cv::medianBlur(cv_img, cv_img, smooth_param1);
//            break;
//         case 3:
//            cv::boxFilter(cv_img, cv_img, -1, cv::Size(smooth_param1*2,smooth_param1*2));
//            break;
//         }
// 	if (display && phase_ == TRAIN)
//       cv::imshow("Smooth Filtering", cv_img);
//   }
//   cv::RNG rng;
//   // Contrast and Brightness Adjuestment ----------------------------------------
//   float alpha = 1, beta = 0;
//   int apply_contrast = Rand(2);
//   if ( contrast_adjustment && apply_contrast ) {
//     float min_alpha = 0.8, max_alpha = 1.2;
//     alpha = rng.uniform(min_alpha, max_alpha);
//     beta = (float)(Rand(6));
// 	// flip sign
// 	if ( Rand(2) ) beta = - beta;
//     cv_img.convertTo(cv_img, -1 , alpha, beta);
// 	if (display && phase_ == TRAIN)
//      		cv::imshow("Contrast Adjustment", cv_img);
//   }

//   // JPEG Compression -------------------------------------------------------------
//   // DO NOT use the following code as there is some memory leak which I cann't figure out
//   int QF = 100;
//   int apply_JPEG = Rand(2);
//   if ( jpeg_compression && apply_JPEG ) {
// 	// JPEG quality factor
// 	QF = 95 + 1 * (Rand(6));
//         int cp[] = {1, QF};
// 	vector<int> compression_params(cp,cp + 2);
//         vector<unsigned char> img_jpeg;
// 	//cv::imencode(".jpg", cv_img, img_jpeg);
//         cv::imencode(".jpg", cv_img, img_jpeg, compression_params);
// 	cv::Mat temp = cv::imdecode(img_jpeg, 1);
//         temp.copyTo(cv_img);
// 	if (display && phase_ == TRAIN)
//       cv::imshow("JPEG Compression", cv_img);
//   }

//   // Cropping -------------------------------------------------------------
//   int h_off = 0;
//   int w_off = 0;
//   cv::Mat cv_cropped_img = cv_img;
//   if (crop_size) {
//     CHECK_EQ(crop_size, height);
//     CHECK_EQ(crop_size, width);
//     // We only do random crop when we do training.
//     if (phase_ == TRAIN) {
//       h_off = Rand(img_height - crop_size + 1);
//       w_off = Rand(img_width - crop_size + 1);
//     } else {
//       h_off = (img_height - crop_size) / 2;
//       w_off = (img_width - crop_size) / 2;
//     }
//     cv::Rect roi(w_off, h_off, crop_size, crop_size);
//     cv_cropped_img = cv_img(roi);
//     if (display && phase_ == TRAIN)
//     	cv::imshow("Cropping", cv_cropped_img);
//   } else {
//     CHECK_EQ(img_height, height);
//     CHECK_EQ(img_width, width);
//   }


//   // Rotation -------------------------------------------------------------
//   double rotation_degree;
//   if ( rotation_angle_interval!=1 ) {
//   cv::Mat dst;
//   int interval = 360/rotation_angle_interval;
//   int apply_rotation = Rand(interval);

//   cv::Size dsize = cv::Size(cv_cropped_img.cols*1.5,cv_cropped_img.rows*1.5);
//   cv::Mat resize_img = cv::Mat(dsize,CV_32S);
//   cv::resize(cv_cropped_img, resize_img,dsize);

//   cv::Point2f pt(resize_img.cols/2., resize_img.rows/2.);    
//   rotation_degree = apply_rotation*rotation_angle_interval;
//   cv::Mat r = getRotationMatrix2D(pt, rotation_degree, 1.0);
//   warpAffine(resize_img, dst, r, cv::Size(resize_img.cols, resize_img.rows));


//   cv::Rect myROI(resize_img.cols/6, resize_img.rows/6, cv_cropped_img.cols, cv_cropped_img.rows);
//   cv::Mat crop_after_rotate = dst(myROI);
//   if (display && phase_ == TRAIN)
//       cv::imshow("Rotation", crop_after_rotate);


//   crop_after_rotate.copyTo(cv_img);
//   }
  
//   if (display && phase_ == TRAIN)
//       cv::imshow("Final", cv_img);


  

//   //--------------------!! for debug only !!-------------------
//   if (display && phase_ == TRAIN) {
// 	LOG(INFO) << "----------------------------------------";
// 	LOG(INFO) << "src width: " << width << ", src height: " << height;
// 	LOG(INFO) << "dest width: " << crop_size << ", dest height: " << crop_size;
// 	if (apply_flipping) {
// 		LOG(INFO) << "* parameter for flipping: ";
// 		LOG(INFO) << "  flipping_mode: " << flipping_mode;
// 	}
// 	if ( smooth_filtering && apply_smooth ) {
//           LOG(INFO) << "* parameter for smooth filtering: ";
// 	  //LOG(INFO) << "  smooth type: " << smooth_type << ", smooth param1: " << smooth_param1;
// 	}
// 	if ( contrast_adjustment && apply_contrast ) {
// 	  LOG(INFO) << "* parameter for contrast adjustment: ";
// 	  LOG(INFO) << "  alpha: " << alpha << ", beta: " << beta;
// 	}
// 	if ( jpeg_compression && apply_JPEG ) {
// 	  LOG(INFO) << "* parameter for JPEG compression: ";
// 	  LOG(INFO) << "  QF: " << QF;
// 	}
// 	LOG(INFO) << "* parameter for cropping: ";
// 	LOG(INFO) << "  w: " << w_off << ", h: " << h_off;
// 	LOG(INFO) << "  roi_width: " << crop_size << ", roi_height: " << crop_size;
// 	LOG(INFO) << "* parameter for rotation: ";
// 	LOG(INFO) << "  angle_interval: " << rotation_angle_interval;
// 	LOG(INFO) << "  angle: " << rotation_degree;
//     cvWaitKey(10);
//   }

//   Dtype* mean = NULL;
//   if (has_mean_file) {
//     CHECK_EQ(img_channels, data_mean_.channels());
//     CHECK_EQ(img_height, data_mean_.height());
//     CHECK_EQ(img_width, data_mean_.width());
//     mean = data_mean_.mutable_cpu_data();
//   }
  
//    if (has_mean_values) {
//     CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
//      "Specify either 1 mean_value or as many as channels: " << img_channels;
//     if (img_channels > 1 && mean_values_.size() == 1) {
//       // Replicate the mean_value for simplicity
//       for (int c = 1; c < img_channels; ++c) {
//         mean_values_.push_back(mean_values_[0]);
//       }
//     }
//   }
  
//   Dtype* transformed_data = transformed_blob->mutable_cpu_data();
//   int top_index;
//   for (int h = 0; h < height; ++h) {
//     const uchar* ptr = cv_img.ptr<uchar>(h); // here!!
//     int img_index = 0;
//     for (int w = 0; w < width; ++w) {
//       for (int c = 0; c < img_channels; ++c) {
//         //if (do_mirror) {
//         //  top_index = (c * height + h) * width + (width - 1 - w);
//         //} else {
//           top_index = (c * height + h) * width + w;
//         //}
//         // int top_index = (c * height + h) * width + w;
//         Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
//         if (has_mean_file) {
//           int mean_index = (c * img_height + h) * img_width + w;
//           transformed_data[top_index] =
//             (pixel - mean[mean_index]) * scale;
//         } else {
//           if (has_mean_values) {
//             transformed_data[top_index] =
//               (pixel - mean_values_[c]) * scale;
//           } else {
//             transformed_data[top_index] = pixel * scale;
//           }
//         }
//       }
//     }
//   }

// }

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  // const Dtype scale = param_.scale();
  Dtype scale = param_.scale(); // to change scale later if normalize_input is set
  const bool do_mirror = param_.mirror() && Rand(2);

  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  // Dinh -----
  const bool do_normalize_input = param_.normalize_input();
  vector<Dtype> scales;
  //-----------
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // From (2) -------------------------------------------------------------
  // Smooth Filtering
  if (param_.smooth_filtering()) {
    // LOG(INFO)<< "Doing smoothing image";
    int smooth_param1 = 3;
    int apply_smooth = Rand(2);
  	int smooth_type = Rand(4); // see opencv_util.hpp
	  smooth_param1 = 3 + 2*(Rand(1));
    switch(smooth_type){
      case 0:
        // cv::smooth(cv_img, cv_img, smooth_type, smooth_param1);
        cv::GaussianBlur(cv_img, cv_img, cv::Size(smooth_param1,smooth_param1),0);
        break;
      case 1:
        cv::blur(cv_img, cv_img, cv::Size(smooth_param1,smooth_param1));
        break;
      case 2:
        cv::medianBlur(cv_img, cv_img, smooth_param1);
        break;
      case 3:
        cv::boxFilter(cv_img, cv_img, -1, cv::Size(smooth_param1*2,smooth_param1*2));
        break;
    }
    if (param_.display() && phase_ == TRAIN){
        cv::imshow("Smooth Filtering", cv_img);
        cv::waitKey(0);
    }
  }
  // Contrast and Brightness Adjuestment
  if (param_.contrast_adjustment()) {
    int apply_contrast = Rand(2);
    if (apply_contrast){
      cv::RNG rng;
      float alpha = 1, beta = 0;
      float min_alpha = 0.8, max_alpha = 1.2;
      alpha = rng.uniform(min_alpha, max_alpha);
      beta = (float)(Rand(6));
      // flip sign
      if ( Rand(2) ) beta = - beta;
        cv_img.convertTo(cv_img, -1 , alpha, beta);
      if (param_.display() && phase_ == TRAIN){
          cv::imshow("contrast_adjustment", cv_img);
          cv::waitKey(0);
      }
    }
  }

  // // Rotation
  // if (param_.rotation_angle_interval()!=1) {
    
  //   const float rotation_angle_interval = param_.rotation_angle_interval();  
  //   int interval = 360/rotation_angle_interval;
  //   int apply_rotation = Rand(interval);
    
  //   if (apply_rotation){
  //     cv::Mat dst;
  //     cv::Size dsize = cv::Size(cv_cropped_img.cols*1.5,cv_cropped_img.rows*1.5);
  //     cv::Mat resize_img = cv::Mat(dsize,cv_img.type());
  //     cv::resize(cv_cropped_img, resize_img, dsize);

  //     cv::Point2f pt(resize_img.cols/2., resize_img.rows/2.);    
  //     float rotation_degree = apply_rotation*rotation_angle_interval;
  //     cv::Mat r = getRotationMatrix2D(pt, rotation_degree, 1.0);
  //     warpAffine(resize_img, dst, r, cv::Size(resize_img.cols, resize_img.rows));

  //     cv::Rect myROI(resize_img.cols/6, resize_img.rows/6, cv_cropped_img.cols, cv_cropped_img.rows);
  //     cv::Mat crop_after_rotate = dst(myROI);
  //     crop_after_rotate.copyTo(cv_img);
      
  //     if (display && phase_ == TRAIN)
  //         cv::imshow("Final", cv_img);
  //   }
  // }

  // Dinh: Add noise
  if (param_.add_noise()){
    int apply_add_noise = Rand(2);
    if (apply_add_noise){
      // cv::RNG rng;
      Dtype m_NoiseStdDev=10;//rng.uniform(0, 20);;
      cv::Mat noise(cv_img.size(),cv_img.type());
      randn(noise,0,m_NoiseStdDev);
      cv_img += noise;
      if (param_.display() && phase_ == TRAIN){
          cv::imshow("add_noise", cv_img);
          cv::waitKey(0);
      }
    }
  }
  //---------------------------------------------------------------------------------
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);
  //Dinh ---
  if (param_.on_fly_mean()){
    vector<Dtype> mean_img;
    for (int c = 0; c < img_channels; ++c){
      mean_img.push_back(0);
    }
    for (int h = 0; h < height; ++h){
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w)
        for (int c = 0; c < img_channels; ++c){
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          mean_img[c] += pixel;
        }
    }
    for (int c = 0; c < img_channels; ++c){
      mean_img[c] /= height * width;
    }
    if (param_.global_mean()){
      Dtype global_mean = 0;
      for (int c = 0; c < img_channels; ++c){
        global_mean += mean_img[c]/img_channels;
      }
      for (int c = 0; c < img_channels; ++c){
        mean_img[c] = global_mean;
      }
    }
    //reset mean_values_ vector
    for (int c = 1; c < img_channels; ++c){
      mean_values_[c] = mean_img[c];
    }
  }
  if (do_normalize_input){
      //Dinh--------------------------------
    // substruct mean and divide std:
    // mean_values = mean of whole image
    // scale = 1/std; std of whole image
    // param: normalize_input: true
    // const bool do_normalize_input = param_.normalize_input();
    // Also adding four lines from 47-50 in this programe, a line at 1028
    ///-----------------------------------
    // Get mean and assign to mean_value_
    vector<Dtype> mean_img;
    for (int c = 0; c < img_channels; ++c){
      mean_img.push_back(0);
    }
    for (int h = 0; h < height; ++h){
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w)
        for (int c = 0; c < img_channels; ++c){
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          mean_img[c] += pixel;
        }
    }
    for (int c = 0; c < img_channels; ++c){
      mean_img[c] /= height * width;
    }
    if (param_.global_normalize()){
      Dtype global_mean = 0;
      for (int c = 0; c < img_channels; ++c){
        global_mean += mean_img[c]/img_channels;
      }
      for (int c = 0; c < img_channels; ++c){
        mean_img[c] = global_mean;
      }
    }
    //reset mean_values_ vector
    for (int c = 1; c < img_channels; ++c){
      mean_values_[c] = mean_img[c];
    }
    // Get std and assign to scale
    vector<Dtype> var;
    for (int c = 0; c < img_channels; ++c){
      var.push_back(0);
    }    
    for (int h = 0; h < height; ++h){
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w)
        for (int c = 0; c < img_channels; ++c){
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          var[c] += (pixel - mean_img[c]) * (pixel - mean_img[c]);
        }
    }
    for (int c = 0; c < img_channels; ++c){
      var[c] /= height * width;
      if (!param_.global_normalize()){
        scales.push_back(1/sqrt(var[c] + 0.00000001));
      }
    }
    if (param_.global_normalize()){
      Dtype global_var = 0;
      for (int c = 0; c < img_channels; ++c){
        global_var += var[c]/img_channels;
      }
      global_var = 1/sqrt(global_var + 0.00000001);
      for (int c = 0; c < img_channels; ++c){
        scales.push_back(global_var);
      }
    }

    // LOG(INFO) << "mean_img: "<<mean_img<<" scale: "<<scale;    
    //------------------------------------
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            // Dinh-----
            if (do_normalize_input) scale = scales[c];
            //----------
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

#endif  // USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}
//-------------------- For VolumeDatum - C3D --------------------------------------------
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const VolumeDatum& datum) {
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_length = datum.length();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.height
  CHECK_GT(datum_channels, 0);
  CHECK_GT(datum_length, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(5);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = datum_length;
  shape[3] = (crop_size)? crop_size: datum_height;
  shape[4] = (crop_size)? crop_size: datum_width;
  return shape;
}
template<typename Dtype>
void DataTransformer<Dtype>::VideoTransform(const VolumeDatum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_length = datum.length();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  /*LOG(INFO) << datum_channels << " channels "
		  << datum_length << " length "
		  << datum_height << " height "
		  << datum_width << " width " << data.size() << " size\n";*/

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);


  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.shape(1));
    CHECK_EQ(datum_length, data_mean_.shape(2));
    CHECK_EQ(datum_height, data_mean_.shape(3));
    CHECK_EQ(datum_width, data_mean_.shape(4));
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int length = datum_length;
  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
	for (int l = 0; l < length; ++l) {
		for (int h = 0; h < height; ++h) {
		  for (int w = 0; w < width; ++w) {
			data_index = ((c * datum_length  + l) * datum_height + h_off + h) * datum_width + w_off + w;
			if (do_mirror) {
			  top_index = ((c * length + l) * height + h) * width + (width - 1 - w);
			} else {
			  top_index = ((c * length + l) * height + h) * width + w;
			}
			if (has_uint8) {
			  datum_element =
				static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
			} else {
			  datum_element = datum.float_data(data_index);
			}
			if (has_mean_file) {
			  transformed_data[top_index] =
				(datum_element - mean[data_index]) * scale;
			} else {
			  if (has_mean_values) {
				transformed_data[top_index] =
				  (datum_element - mean_values_[c]) * scale;
			  } else {
				transformed_data[top_index] = datum_element * scale;
			  }
			}
		  }
		}
	}
  }
}
template<typename Dtype>
void DataTransformer<Dtype>::VideoTransform(const VolumeDatum& datum,
                                       Blob<Dtype>* transformed_blob) {
	  const int crop_size = param_.crop_size();
	  const int datum_channels = datum.channels();
	  const int datum_length = datum.length();
	  const int datum_height = datum.height();
	  const int datum_width = datum.width();

	  // Check dimensions.
	  const int num = transformed_blob->shape(0);
	  const int channels = transformed_blob->shape(1);
	  const int length = transformed_blob->shape(2);
	  const int height = transformed_blob->shape(3);
	  const int width = transformed_blob->shape(4);


	  CHECK_EQ(channels, datum_channels);
	  CHECK_LE(length, datum_length);
	  CHECK_LE(height, datum_height);
	  CHECK_LE(width, datum_width);
	  CHECK_GE(num, 1);

	  if (crop_size) {
	    CHECK_EQ(crop_size, height);
	    CHECK_EQ(crop_size, width);
	  } else {
	    CHECK_EQ(datum_height, height);
	    CHECK_EQ(datum_width, width);
	  }

	  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	  VideoTransform(datum, transformed_data);
}
//---------------------------------------------------------------------------------------
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
