# ai-hik-tracking
Control DMX Laser using Deep Learning object detection


## Playing Card Control keys
| Key  | Effect  |
|:----------|:----------|
| Num 1-9 | Change YOLO Profile |
| Ctrl+Num 1-9 | Change YOLO Profile and backend between darknet / OpenCV |
| p | OpenCV DNN toggle FP32 / FP16    |
| Ctrl+s | Save current frame with recognize result |
| Ctrl+c | Toggle using CPU / GPU preprocess |
| b | Preview split / black result |
| z | reduce preprocess alpha 0.05 |
| x | increase preprocess alpha 0.05 |
| c | reduce preprocess beta 5 |
| v | increase preprocess beta 5 |
| g | reduce drawing approx thresh 0.001 |
| h | increase drawing approx thresh 0.001 |
| F1 | Change split analyize mode |


## Shared Control keys
| Key  | Effect  |
|:----------|:----------|
| Num 1-9 | Change Palcon-D PTZ Preset |
| Shift+Num 1-9 | Save Current Position to Palcon-D PTZ Preset |
| U/D/L/R | Cal Mode: Move Laser |
| i/j/k/l | Cal Mode: Move PTZ |
| Ctrl+z | Pop last cal result |
| +/- | zoom in/zoom out laser |
| Escape | Switch between cal / test mode |



## Training Data Refereence
* https://github.com/okmd/playing-card-dataset.git


## Use components
* https://github.com/magicbear/py-hikevent


## Data Prepare Toolkit
* https://github.com/spytensor/prepare_detection_dataset.git



## Project Reference
* https://github.com/AlexeyAB/darknet

* https://github.com/facebookresearch/detectron2

* https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector

* https://github.com/a-arbabian/playing-cards-object-detection
