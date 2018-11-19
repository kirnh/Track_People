This repository contains code for tracking muliple people in a video.

This extends the code provided with DEEP SORT tracking paper that can be found [here](https://arxiv.org/abs/1703.07402) by using YOLOv3 for creating the required detections obtained [here](https://github.com/eriklindernoren/PyTorch-YOLOv3) and acts as a framework for experimenting with various detection networks alongside DEEP SORT. 

First, download the required model weights for both YOLOv3 and re-id feature extractor from [this link]() and extract them into the deep_sort root directory. 

To run the tracker, use the command "python deep_sort_app.py --video_path={input video path} --output_file={output video file path}" from a terminal within the deep_sort root directory.