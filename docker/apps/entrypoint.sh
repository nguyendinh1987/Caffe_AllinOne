#!/bin/bash
# opt 1
#Xvfb -screen 0 900x900x16 -ac &
#sleep 15
#env DISPLAY=:0.0 x11vnc -noxrecord -noxfixes -noxdamage -forever -display :0 -create -rfbauth /root/.vnc/passwd & #-noxrecord -noxfixes -noxdamage -forever -display :0 &
#env DISPLAY=:0.0 fluxbox

# opt 2
export DISPLAY=${DISPLAY:-:0} # Select screen 0 by default.
! pgrep -a Xvfb && Xvfb $DISPLAY -screen 0 1024x768x16 &
sleep 1
xdpyinfo
if which x11vnc &>/dev/null; then
  ! pgrep -a x11vnc && x11vnc -forever -rfbauth /root/.vnc/passwd -quiet -display WAIT$DISPLAY &
fi
if which fluxbox &>/dev/null; then
  ! pgrep -a fluxbox && fluxbox 2>/dev/null &
fi
echo "IP: $(hostname -I) ($(hostname))"
bash
