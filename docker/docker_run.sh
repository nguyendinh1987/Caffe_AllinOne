docker run \
       --rm \
       -ti \
       --gpus all \
       --name "Dinh_cuda_caffe" \
       -p 5900:5900 \
       -e HOME=/ \
       -v /opt/share:/opt/share \
       -v /opt/localmedia:/media/kakadinh \
       cuda_caffe_dinh x11vnc -forever -create -rfbauth /root/.vnc/passwd

