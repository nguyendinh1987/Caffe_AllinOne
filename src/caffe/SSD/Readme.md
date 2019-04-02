# Caffe_All_in_One (All developed deep learning layers in caffe)
# All components of SSD implementation have been separated from original caffe implementation and stored in this folder, excluding implementation of SSD solver and parallel straty. The differences between SSD parallel and original caffe parallel is that it does not inherit from NCCL library. Instead, they implemented their own method. The SSD solver supports to load train net and val net from different files as train.prototxt and test.prototxt, while original solver supports to read train net and test net from a file as train_val.prototxt

1. What I have changed to add SSD components: 
- Update caffe.proto
- Add all .cpp files to src/caffe/SSD and .hpp files to include/caffe/SSD
- Add regex components to find_package for boost in cmake/Dependencies.cmake because it is asked by SSD layers

2. What are interesting things in SSD implementation
- They provided a lot of on-fly image augmentation in their AnnotatedDataLayer. To seperate their annotation functions from original DataTransformer class, I added another class called SSDDataTransformer class which is an inheritance of DataTransformer. In order to use, you need to access this class instead of DataTransformer. A list of their on-fly augmentation includes 
+ Resize  
+ Noise has decolorize (BGR->GRAY->BGR), gaussianBlur (window 7x7, gama=1.5), saltpepper, color histogram equal, clathe, jpeg compression, erode, posterize, inverse (cv2::bitwise_not), convert_to_hsv, convert_to_lab
+ Distortion has randomBrightness, randomContrast, randomSaturation, randomHue, randomReorderingChannel 
+ Image expansion
+ Emitconstraint
- In data_reader.cpp, we can learn how we can manage data loader in Threads

