#ifndef CAFFE_SSD_UTIL_IO_H_
#define CAFFE_SSD_UTIL_IO_H_

#include "caffe/util/io.hpp"
#include <map>

namespace caffe {

inline void GetTempDirname(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      bool remove_done = boost::filesystem::remove(dir);
      if (remove_done) {
        *temp_dirname = dir.string();
        return;
      }
      LOG(FATAL) << "Failed to remove a temporary directory.";
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void GetTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    GetTempDirname(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadXMLToAnnotatedDatum(const string& labelname, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum);//Done

bool ReadJSONToAnnotatedDatum(const string& labelname, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum);//Done

bool ReadTxtToAnnotatedDatum(const string& labelname, const int height,
    const int width, AnnotatedDatum* anno_datum);//Done

bool ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map);//Done

inline bool ReadLabelFileToLabelMap(const string& filename,
      bool include_background, LabelMap* map) {
  return ReadLabelFileToLabelMap(filename, include_background, " ", map);
}

inline bool ReadLabelFileToLabelMap(const string& filename, LabelMap* map) {
  return ReadLabelFileToLabelMap(filename, true, map);
}

bool MapNameToLabel(const LabelMap& map, const bool strict_check,
                    std::map<string, int>* name_to_label);//Done

inline bool MapNameToLabel(const LabelMap& map,
                           std::map<string, int>* name_to_label) {
  return MapNameToLabel(map, true, name_to_label);
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
                    std::map<int, string>* label_to_name);//Done

inline bool MapLabelToName(const LabelMap& map,
                           std::map<int, string>* label_to_name) {
  return MapLabelToName(map, true, label_to_name);
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
                           std::map<int, string>* label_to_display_name);//Done

inline bool MapLabelToDisplayName(const LabelMap& map,
                              std::map<int, string>* label_to_display_name) {
  return MapLabelToDisplayName(map, true, label_to_display_name);
}

#ifdef USE_OPENCV
void GetImageSize(const string& filename, int* height, int* width);//Done
/////////////////////////////////////////////////////////////////////////////////////
//////////ReadImageToDatumSSD
/////////////////////////////////////////////////////////////////////////////////////
bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum);//Need to implement

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, height, width, min_dim, max_dim,
                          is_color, "", datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    Datum* datum) {
  return ReadImageToDatumSSD(filename, label, height, width, min_dim, max_dim,
                          true, datum);
}
//--> There is another ReadImageToDatum function implemented in original io.cpp
inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, height, width, 0, 0, is_color,
                          encoding, datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatumSSD(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatumSSD(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatumSSD(filename, label, 0, 0, true, encoding, datum);
}
/////////////////////////////////////////////////////////////////////////////////////
bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelname, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const std::string& encoding, const AnnotatedDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum); //Done

inline bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelname, const int height, const int width,
    const bool is_color, const std::string & encoding,
    const AnnotatedDatum_AnnotationType type, const string& labeltype,
    const std::map<string, int>& name_to_label, AnnotatedDatum* anno_datum) {
  return ReadRichImageToAnnotatedDatum(filename, labelname, height, width, 0, 0,
                      is_color, encoding, type, labeltype, name_to_label,
                      anno_datum);
}
cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim);

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum);

#endif  // USE_OPENCV
} // namespace caffe


#endif // end define
