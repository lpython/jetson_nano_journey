 
lsb_release -ds 
gcc --version | head -1
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | g
rep -c GLIBCXX_3.4.2[5-9]; /usr/local/cuda/bin/nvcc --version | tail -1
gst-launch-1.0 --version | head -1; df -h / | tail -1
cat /etc/nvpmodel.conf >/dev/null 2>&1 && sudo -n nvpmodel -q 2>&1 | head -2
ls /dev/video* 2>&1' 2>&1 | grep -v -e WARNING -e 'This session' -e 'server may'

# Ubuntu 20.04.6 LTS
# gcc (Ubuntu/Linaro 8.4.0-3ubuntu2) 8.4.0
# 4
# Build cuda_10.2_r440.TC440_70.29663091_0
# gst-launch-1.0 version 1.16.3
# /dev/mmcblk0p1   58G   24G   32G  43% /
# NVPM WARN: fan mode is not set!
# NV Power Mode: MAXN
# /dev/video0