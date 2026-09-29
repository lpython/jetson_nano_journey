Disocery compared to last project

Last project dropped no frames.
Here gstreamer should be the top process.
Tee and queue should be combined so that the downstream python process will not hold up the video stream.
See docs/tee_demo.md
