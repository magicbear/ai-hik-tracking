import datetime

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
import cv2
import time
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import threading
# from google.colab.patches import cv2_imshow
from stupidArtnet import StupidArtnet

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import hikevent
import struct
import sdl2
import ctypes
import socket
import sys, getopt

from laser_control import ArtNetThread, GUIThread, TerminatedState

terminated = TerminatedState()

projection_ratio = 1.0

def mapfloat(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class AnalyizeThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.mutex = threading.Lock()
        self.frame = None
        self.artnet = ArtNetThread(terminated, '192.168.20.16', '192.168.20.15')
        self.artnet.start()
        self.gui = GUIThread(self.artnet, self, terminated)
        self.gui.start()

    def run(self):
        global terminated

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        while not terminated.get():
            if self.frame is not None:
                self.mutex.acquire()
                if self.frame.shape[1] > 1280:
                    frame = cv2.resize(self.frame, (1280, 720))
                else:
                    frame = self.frame
                self.frame = None
                self.mutex.release()

                if not self.artnet.cal_mode and self.artnet.test_mode is None:
                    poslist = []
                    outputs = predictor(frame)

                    # We can use `Visualizer` to draw the predictions on the image.
                    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
                    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
                    pred_class_names = list(map(lambda x: class_names[x], pred_classes))
                    boxes = outputs["instances"].pred_boxes

                    # cv2_imshow(out.get_image()[:, :, ::-1])
                    # Display the resulting frame
                    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    out = v.draw_instance_predictions(outputs["instances"][outputs["instances"].pred_classes == 0].to("cpu"))
                    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    self.gui.mutex.acquire()
                    self.gui.frame = out.get_image()[:, :, ::-1]
                    self.gui.mutex.release()
                    for x in range(0,len(pred_classes)):
                        if pred_class_names[x] == "person":
                            # print("pred_classes", pred_classes, pred_class_names)
                            size = (boxes[x].tensor[:, 2:] - boxes[x].tensor[:, :2]).cpu().tolist()
                            pos = boxes[x].get_centers().cpu().tolist()
                            X = self.gui.get_cal_point(pos[0][0] * 1280. / frame.shape[1], pos[0][1] * 720. / frame.shape[0], self.artnet.ptz_preset, True if len(outputs["instances"][outputs["instances"].pred_classes == 0]) == 1 else False)
                            laser_x = X[0]
                            laser_y = X[1]

                            print(pos, size)
                            if laser_x < 0 or laser_y < 0 or laser_x > 255 or laser_y > 255:
                                continue
                            poslist.append(X);

                    self.artnet.mutex.acquire()
                    if not self.artnet.cal_mode and self.artnet.test_mode is None:
                        self.artnet.pos = poslist if len(poslist) > 0 else None
                    self.artnet.mutex.release()
                else:
                    self.gui.mutex.acquire()
                    self.gui.frame = frame
                    self.gui.mutex.release()

            time.sleep(0.01)


analyizeThread = AnalyizeThread()
analyizeThread.start()

lastCheck = 0

try:
    opts, _ = getopt.getopt(sys.argv[1:], "u:p:H:", ["user=", "passwd=", "ip="])
    ip = None
    user = None
    passwd = None
    for opt, arg in opts:
        if opt in ["-u", "--user"]:
            user = arg
        if opt in ["-p", "--passwd"]:
            passwd = arg
        if opt in ["-H", "--ip"]:
            ip = arg
except getopt.GetoptError:
    show_help()
    sys.exit(2)

# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/channels/101" % (user, passwd, ip))
# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/channels/302" % (user, passwd, ip))
# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/tracks/301?starttime=20220310T152930z" % (user, passwd, ip))
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#
# while not terminated:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if time.time() - lastCheck >= 0.05:
#         lastCheck = time.time()
#         analyizeThread.mutex.acquire()
#         analyizeThread.frame = frame
#         analyizeThread.mutex.release()

cam = hikevent.hikevent(ip, user, passwd)
cam.startRealPlay(1, 0)

analyizeThread = AnalyizeThread()
analyizeThread.start()

while not terminated.get():
    # Capture frame-by-frame
    evt = cam.getevent()
    if evt is not None:
        if evt['command'] == "DVR_VIDEO_DATA":
            size = struct.unpack("=LL", evt['payload'][0:8])
            if time.time() - lastCheck >= 0.1:
                frame = np.frombuffer(evt['payload'][8:], dtype=np.uint8).reshape((size[1], size[0], 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

                lastCheck = time.time()
                analyizeThread.mutex.acquire()
                analyizeThread.frame = frame
                analyizeThread.mutex.release()
    else:
        time.sleep(0.001)
