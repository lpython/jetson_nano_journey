cat /sys/class/video4linux/video0/name; 
v4l2-ctl -d /dev/video0 --list-formats 2>&1 | head -8; 
nproc;
cat /proc/device-tree/model; 
echo 2>&1 | grep -v -e WARNING -e 'This session' -e 'server may' 

# vi-output, imx219 8-0010
# ioctl: VIDIOC_ENUM_FMT
# 	Type: Video Capture

# 	[0]: 'RG10' (10-bit Bayer RGRG/GBGB)


# nproc 
# 4