docker run \
       --rm \
       -ti \
       --gpus all \
       --name "Dinh_cuda_caffe" \
       -p 5900:5900 \
       -v /opt/share:/opt/share \
       cuda_caffe_dinh #x11vnc -forever -create -rfbauth /root/.vnc/passwd 

