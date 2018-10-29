# Caffe_with_updated_functions
This is a modified version of caffe which includes a lot of additional functions supporting for FCNN, video loader, data agumentation and so on. Some of modifications are inheritated from other developer in github (Thank for sharing)
## Targets:
- Add FRCNN layers
- Add Video loader layers
- Add on-fly data argumentation for improving training process
- Add YoLo implementation
- Add SSD implementation

## Modified
- Add FRCNN layers (see folder FRCNN in src/ and include/)
  [Source](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)
  Modified:  caffe.proto and add FRCNN folder into src/caffe and include/caffe
	                           add api folder into src and include
- Add Video loader layers (for implementing DSN and TSN networks)
  [Source](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)
  Modified: caffe.proto and add ACTION_REC folder into src/caffe and include/caffe

- Add on-fly data argumentation
  [Source 1](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)
  [Source 2](https://github.com/kevinlin311tw/caffe-augmentation) (have not upgraded yet)
