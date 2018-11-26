# Caffe_with_updated_functions
This is a modified version of caffe which includes a lot of additional functions supporting for FCNN, video loader, data agumentation and so on. Some of modifications are inheritated from other developers in github (Thank for sharing)
## Targets:
- Add FRCNN layers
- Add Video loader layers
- Add BN Layer
- Add on-fly data argumentation for improving training process
- Add YoLo implementation (planing)
- Add SSD implementation (planing)
- Add layers for flownet (planing) [Source](https://github.com/lmb-freiburg/flownet2)
- continue

## Modified
- Add FRCNN layers (see folder FRCNN in src/ and include/)  
  [Source](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)  
  Modified:  caffe.proto and add FRCNN folder into src/caffe and include/caffe  
	                           add api folder into src and include
- Add Video loader layers (for implementing DSN and TSN networks)  
  [Source](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)  
  Modified: caffe.proto and add ACTION_REC folder into src/caffe and include/caffe  

- Add BN layer    
  [Source](https://github.com/yjxiong/caffe)     
  Modified: caffe.proto and add BN_LAYER folder into src/caffe and include/caffe  

- Add on-fly data argumentation  
  [Source 1](https://github.com/yjxiong/caffe) Modified function:  
  void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data)  
  [Source 2](https://github.com/kevinlin311tw/caffe-augmentation) Add "contrast_adjustment, smooth_filtering"  
  Other augmentation: Add noise  
  Modified function: void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob)
## How to install
- Check "Dependencies_installation_guide" to install dependent libraries of caffe
- Move to project folder and run:  
  >> mkdir build && cd build && cmake ..  
  >> make -j12  
  >> make install  
- All distributed packaged is in /build/install. Give this directory to your any package finding function to look for your caffe lib  

## Updated package:  
Please see the branch version_1.0 for the updated version. I am working on this version and will merger to the branch master later after checking all states. Thank you !!!!!
