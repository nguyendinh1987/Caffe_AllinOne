docker run \
       --rm \
       -ti \
       --name "test_cuda_caffe" \
       -p 5900:5900 \
       -e HOME=/ \
       -v /opt/share:/opt/share \
       caffe_allinone x11vnc -forever -create -rfbauth root/.vnc/passwd
