# Caffe_All_in_One (All developed deep learning layers in caffe)
This is a modified version of caffe which includes a lot of additional functions supporting for FCNN, video loader, data agumentation and so on. Some of modifications are inheritated from other developers in github (Thank for sharing)
## Targets:
- Add FRCNN layers
- Add Video loader layers
- Add BN Layer
- Add on-fly data argumentation for improving training process
- Add C3D (3D convolution), 3D pooling, video layer from facebook
- Add RoiAlignment layer for mask rcnn
- Add YoLo implementation (planing)
- Add SSD implementation (planing)
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

- Add C3D (3D convolution), 3D pooling, video layer from facebook  
  [source](https://github.com/facebook/C3D)  
  
  Modified: Add folders "C3D", "Pooling3D" and "VIDEO_LAYER_FB" in src/ and include/. Modify caffe.proto

- Add on-fly data argumentation  
  [Source 1](https://github.com/yjxiong/caffe) Modified function:  
  void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data)  
  [Source 2](https://github.com/kevinlin311tw/caffe-augmentation) Add "contrast_adjustment, smooth_filtering"  
  Other augmentation: Add noise  
  Modified function: void DataTransformer::Transform(const cv::Mat& cv_img, Blob* transformed_blob)
	
- Add RoiAlign layer for Mask-RCNN  
  [source](https://github.com/jasjeetIM/Mask-RCNN)
  
## How to install
- Check "Dependencies_installation_guide" to install dependent libraries of caffe
- Move to project folder and run:  
  >> mkdir build && cd build && cmake ..  
  >> make -j12  
  >> make install  
- All distributed packaged is in /build/install. Give this directory to your any package finding function to look for your caffe lib  

## How to use  
Please take a look at "sample_protos" for examples of using added layers

## [Advance] If you want to design a new layer  
(Let me used RecallEval layer for an example)  
The RecallEval layer has two params as IoUs and Npps where IoUs refers to Intersection over Union threshold and Npps refers to number of proposals selected for evaluation. The IoUs and Npps will be assigned by users by net proto file.  
- Step 1: Modify caffe.proto file  
>> Provide definition of your layer parameter under the message "LayerParameter":  
    <span style="background-color:green"> optional RecallEvalParameter recall_based_eval_param = 210;</span>  
>> Provide definition of your layer parameter components follow message format:  
    message RecallEvalParameter {  
      repeated float IoUs = 1; // Intersection over Union  
      repeated int32 Npp = 2; // Number of object for evaluation  
    }
- Step 2: Add the header file (.hpp) of your own layer in "include/caffe/<your path>/<your file>" (inlcude/caffe/FRCNN/frcnn_proposal_recall_eval_layer.hpp)
- Step 3: Add the source files (.cpp and .cu) of your own layer in "src/caffe/<your path>/<your file>" (src/caffe/FRCNN/frcnn_proposal_recall_eval_layer.cpp; src/caffe/FRCNN/frcnn_proposal_recall_eval_layer.cu)
- Mandatory contents of header and source file:
