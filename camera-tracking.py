import numpy as np
import os, json, cv2, random
import threading
import time, datetime
import hikevent
import struct
import imutils
import sdl2
from imutils import contours, perspective
from imutils.object_detection import non_max_suppression
from queue import Queue
import colorsys

import base64, functools
import math
from labelme import utils
import sys, getopt
from laser_control import ArtNetThread, GUIThread, TerminatedState, OpenCV_dnnThread, DarknetThread, Detectron2Thread, QueueFrame
import darknet
from simple_pid import PID

terminated = TerminatedState()

projection_ratio = 1.0
minCardSizeRatio = 0.04
minCardAreaRatio = 0.04 * 0.08  # Card Area size required
maxCardSizeRatio = 5  # Width / Height Ratio for overlap Cards
approxThresh = 0.04  # Approx of edges for split
maxAllowShape = 8  # Max allow shape for split
acceptApproxContourRange = [0.8, 1.1]
detailAnalyizeMode = 0  # 0 No Split    1 Split by Block   2 Split by Object
cvCudaProcess = cv2.cuda.getCudaEnabledDeviceCount() > 0

data_file = 'darknet/cfg/coco.data'
cfg_file = None
weightFile = None

def mapfloat(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class AnalyizeThread(threading.Thread):
    def __init__(self, cam, channel):
        threading.Thread.__init__(self)
        self.mutex = threading.Lock()
        self.frame = None
        self.paused = False
        self.gui = GUIThread(None, self, terminated)
        self.gui.start()
        self.gui.onSDL_Event = self.onSDL_Event
        self.darknet_height = 0
        self.darknet_width = 0
        self.lastFrame = None
        self.detectResult = []
        self.displayThreshold = 0
        self.alpha = 1.0
        self.beta = 0
        self.approx_thresh = 0.01  # Approx For Build shape (not for split)
        self.detection_done = 0
        self.t_detect = 0
        self.queue = Queue()
        self.threads = []
        self.detections_adjusted = []  # Offset: Center X, Center Y,
        self.available_cards = []
        self.useCudaProcess = cvCudaProcess
        self.backend = "opencv"
        self.default_pref = cv2.dnn.DNN_TARGET_CUDA_FP16
        self.load_dnn_networks()
        self.sat_thresh = 60
        self.bright_thresh = 60
        self.ptzSpeed = 7
        self.lastSpeed = 1
        self.lastPTZCommand = None
        self.adjustAttributes = {}
        self.pidAdjustAttributes = {}
        self.max_frame_width = 1920
        self.max_frame_height = 1080
        self.moveDirFlags = [False, False, False, False, False, False]
        self.xPID = PID(15, 1, 0.2, setpoint=0)
        self.xPID.output_limits = (-7, 7)
        self.yPID = PID(15, 1, 0.2, setpoint=0)
        self.yPID.output_limits = (-7, 7)
        self.zoomPID = PID(5, 0.1, 0.2, setpoint=0.5)
        self.zoomPID.output_limits = (-7, 7)
        self.ptzOperating = [False, False, False, False, False, False]
        self.fullDuplexZoom = False
        self.cam = cam
        self.channel = channel

    def load_dnn_networks(self, load_cfg=0):
        global cfg_file, weightFile

        config_set = [
            ['darknet/cfg/yolov4.cfg', 'pre-trained/yolov4.weights'],
            ['darknet/cfg/yolov4-tiny.cfg', 'pre-trained/yolov4-tiny.weights'],
            ['darknet/cfg/yolov4-csp.cfg', 'pre-trained/yolov4-csp.weights'],
            ['darknet/cfg/yolov4-csp-swish.cfg', 'pre-trained/yolov4-csp-swish.weights'],
        ]
        if self.backend == "detectron2":
            from detectron2 import model_zoo
            config_set = [
                [model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"), model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")],
                [model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"),
                 model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")],
                [model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"),
                 model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")]
            ]
        if load_cfg >= len(config_set):
            load_cfg = load_cfg % len(config_set)
        cfg_file = config_set[load_cfg][0]
        weightFile = config_set[load_cfg][1]
        self.mutex.acquire()
        while self.queue.qsize():
            self.queue.get()

        old_threads = self.threads
        for i in range(len(old_threads)):
            old_threads[i].stop()

        self.threads = []
        self.detection_done = 0xff

        if self.backend == "darknet":
            for i in range(1):
                th_darknet = DarknetThread(self, terminated, cfg_file, data_file, weightFile)
                th_darknet.start()
                self.threads.append(th_darknet)
        elif self.backend == "opencv":
            for i in range(1):
                th_darknet = OpenCV_dnnThread(self, terminated, cfg_file, data_file, weightFile)
                th_darknet.start()
                self.threads.append(th_darknet)
        elif self.backend == "detectron2":
            for i in range(1):
                th_darknet = Detectron2Thread(self, terminated, cfg_file, data_file, weightFile)
                th_darknet.start()
                self.threads.append(th_darknet)

        self.darknet_width = self.threads[0].darknet_width
        self.darknet_height = self.threads[0].darknet_height

        self.mutex.release()

    def updatePTZCommand(self):
        do_ptz_command = None
        if self.moveDirFlags[0] and self.moveDirFlags[3]:
            do_ptz_command = 25     # UP_LEFT
            if self.moveDirFlags[4]:
                do_ptz_command = 64  # UP_LEFT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 65  # UP_LEFT_ZOOM_OUT
        elif self.moveDirFlags[0] and self.moveDirFlags[1]:
            do_ptz_command = 26     # UP_RIGHT
            if self.moveDirFlags[4]:
                do_ptz_command = 66  # UP_RIGHT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 67  # UP_RIGHT_ZOOM_OUT
        elif self.moveDirFlags[2] and self.moveDirFlags[3]:
            do_ptz_command = 27     # DOWN_LEFT
            if self.moveDirFlags[4]:
                do_ptz_command = 68  # DOWN_LEFT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 69  # DOWN_LEFT_ZOOM_OUT
        elif self.moveDirFlags[2] and self.moveDirFlags[1]:
            do_ptz_command = 28     # DOWN_RIGHT
            if self.moveDirFlags[4]:
                do_ptz_command = 70  # DOWN_RIGHT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 71  # DOWN_RIGHT_ZOOM_OUT
        elif self.moveDirFlags[0]:
            do_ptz_command = 21     # TILT_UP
            if self.moveDirFlags[4]:
                do_ptz_command = 72  # TILT_UP_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 73  # TILT_UP_ZOOM_OUT
        elif self.moveDirFlags[2]:
            do_ptz_command = 22     # TILT_DOWN
            if self.moveDirFlags[4]:
                do_ptz_command = 58  # TILT_DOWN_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 59  # TILT_DOWN_ZOOM_OUT
        elif self.moveDirFlags[1]:
            do_ptz_command = 24     # PAN_RIGHT
            if self.moveDirFlags[4]:
                do_ptz_command = 62  # PAN_RIGHT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 63  # PAN_RIGHT_ZOOM_OUT
        elif self.moveDirFlags[3]:
            do_ptz_command = 23     # PAN_LEFT
            if self.moveDirFlags[4]:
                do_ptz_command = 60  # PAN_LEFT_ZOOM_IN
            elif self.moveDirFlags[5]:
                do_ptz_command = 61  # PAN_LEFT_ZOOM_OUT
        elif self.moveDirFlags[4]:
            do_ptz_command = 11  # ZOOM_IN
        elif self.moveDirFlags[5]:
            do_ptz_command = 12  # ZOOM_OUT
        else:
            do_ptz_command = None

        cmdMapping = {"UP_LEFT": 25, "UP_LEFT_ZOOM_IN": 64, "UP_LEFT_ZOOM_OUT": 65, "UP_RIGHT": 26, "UP_RIGHT_ZOOM_IN": 66, "UP_RIGHT_ZOOM_OUT": 67, "DOWN_LEFT": 27, "DOWN_LEFT_ZOOM_IN": 68, "DOWN_LEFT_ZOOM_OUT": 69, "DOWN_RIGHT": 28, "DOWN_RIGHT_ZOOM_IN": 70, "DOWN_RIGHT_ZOOM_OUT": 71, "TILT_UP": 21, "TILT_UP_ZOOM_IN": 72, "TILT_UP_ZOOM_OUT": 73, "TILT_DOWN": 22, "TILT_DOWN_ZOOM_IN": 58, "TILT_DOWN_ZOOM_OUT": 59, "PAN_RIGHT": 24, "PAN_RIGHT_ZOOM_IN": 62, "PAN_RIGHT_ZOOM_OUT": 63, "PAN_LEFT": 23, "PAN_LEFT_ZOOM_IN": 60, "PAN_LEFT_ZOOM_OUT": 61, "ZOOM_IN": 11, "ZOOM_OUT": 12}
        if self.lastPTZCommand != do_ptz_command or self.lastSpeed != self.ptzSpeed:
            self.lastSpeed = self.ptzSpeed
            if self.lastPTZCommand is not None:
                print("\033[31m%s\033[0m S %d " % (list(cmdMapping.keys())[list(cmdMapping.values()).index(self.lastPTZCommand)], self.ptzSpeed))
                self.cam.ptzControl(self.channel, self.lastPTZCommand, True, self.ptzSpeed)
            if do_ptz_command is not None:
                print("\033[32m%s\033[0m S %d " % (list(cmdMapping.keys())[list(cmdMapping.values()).index(do_ptz_command)], self.ptzSpeed))
                self.cam.ptzControl(self.channel, do_ptz_command, False, self.ptzSpeed)
            self.lastPTZCommand = do_ptz_command


    def onSDL_Event(self, event):
        try:
            if event.type == sdl2.SDL_KEYUP:
                if event.key.keysym.sym >= sdl2.SDLK_1 and event.key.keysym.sym <= sdl2.SDLK_9:
                    global cfg_file, weightFile
                    if event.key.keysym.mod & sdl2.KMOD_CTRL:
                        backends = ["opencv", "darknet", "detectron2"]
                        self.backend = backends[(backends.index(self.backend)+1) % len(backends)]

                    self.load_dnn_networks(event.key.keysym.sym - sdl2.SDLK_1)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_p:
                    if self.backend == "opencv":
                        self.default_pref = cv2.dnn.DNN_TARGET_CUDA if self.default_pref == cv2.dnn.DNN_TARGET_CUDA_FP16 else cv2.dnn.DNN_TARGET_CUDA_FP16
                        for th in self.threads:
                            th.net.setPreferableTarget(self.default_pref)
                        print("Set Preferable Target %s" % (
                            "DNN_TARGET_CUDA" if self.default_pref == cv2.dnn.DNN_TARGET_CUDA else "DNN_TARGET_CUDA_FP16"))
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_s and event.key.keysym.mod & sdl2.KMOD_LCTRL:
                    print("Save Result")
                    now_ts = int(time.time())
                    cv2.imwrite("data/saved/%d.jpg" % (now_ts), cv2.cvtColor(self.lastFrame, cv2.COLOR_RGB2BGR))
                    with open("data/saved/%d.jpg" % (now_ts), "rb") as f:
                        imageData = f.read()
                    labelme_format = {"version": "3.6.16", "flags": {}, "lineColor": [0, 255, 0, 128],
                                      "fillColor": [255, 0, 0, 128], "imagePath": "%d.jpg" % (now_ts),
                                      "imageHeight": self.lastFrame.shape[0], "imageWidth": self.lastFrame.shape[1],
                                      "imageData": base64.b64encode(imageData).decode('utf-8')}
                    shapes = []
                    print(self.detectResult)
                    for shape in self.detectResult:
                        if float(shape[1]) <= 80:
                            continue
                        pos = shape[2]
                        s = {"label": shape[0], "line_color": None, "fill_color": None, "shape_type": "rectangle"}
                        points = [
                            [pos[0] - pos[2] / 2, pos[1] - pos[3] / 2],
                            [pos[0] - pos[2] / 2 + pos[2], pos[1] - pos[3] / 2 + pos[3]]
                        ]
                        s["points"] = points
                        shapes.append(s)
                    labelme_format["shapes"] = shapes
                    json.dump(labelme_format, open("data/saved/%d.json" % now_ts, "w"), ensure_ascii=False, indent=2)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_w or event.key.keysym.sym == sdl2.SDLK_s:
                    self.mutex.acquire()
                    del self.adjustAttributes['sat_thresh']
                    self.mutex.release()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_a or event.key.keysym.sym == sdl2.SDLK_d:
                    self.mutex.acquire()
                    del self.adjustAttributes['bright_thresh']
                    self.mutex.release()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_c and (event.key.keysym.mod & sdl2.KMOD_CTRL):
                    self.useCudaProcess = not self.useCudaProcess
                    print("Using Cuda: %s" % ("YES" if self.useCudaProcess else "No"))
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_b:
                    self.displayThreshold = (self.displayThreshold + 1) % 4
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_z:
                    self.alpha -= 0.05
                    print("Alpha %.03f" % self.alpha)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_x:
                    self.alpha += 0.05
                    print("Alpha %.03f" % self.alpha)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_c:
                    self.beta -= 5
                    print("Beta %d" % self.beta)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_v:
                    self.beta += 5
                    print("Beta %d" % self.beta)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_g:
                    self.approx_thresh -= 0.001
                    print("Approx Thresh %.4f" % self.approx_thresh)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_h:
                    self.approx_thresh += 0.001
                    print("Approx Thresh %.4f" % self.approx_thresh)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_F1:
                    global detailAnalyizeMode
                    detailAnalyizeMode = (detailAnalyizeMode+1)%3
                    print("Using Split Analyize Mode: %d" % detailAnalyizeMode)
                elif event.key.keysym.sym == sdl2.SDLK_UP:
                    self.moveDirFlags[0] = False
                    self.moveDirFlags[2] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                    self.moveDirFlags[0] = False
                    self.moveDirFlags[2] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                    self.moveDirFlags[1] = False
                    self.moveDirFlags[3] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    self.moveDirFlags[1] = False
                    self.moveDirFlags[3] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_EQUALS:
                    # cam.ptzControl(channel, 11, True)
                    self.moveDirFlags[4] = False
                    self.moveDirFlags[5] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_MINUS:
                    # cam.ptzControl(channel, 12, True)
                    self.moveDirFlags[4] = False
                    self.moveDirFlags[5] = False
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_u:
                    del self.pidAdjustAttributes['Kp']
                elif event.key.keysym.sym == sdl2.SDLK_j:
                    del self.pidAdjustAttributes['Kp']
                elif event.key.keysym.sym == sdl2.SDLK_i:
                    del self.pidAdjustAttributes['Ki']
                elif event.key.keysym.sym == sdl2.SDLK_k:
                    del self.pidAdjustAttributes['Ki']
                elif event.key.keysym.sym == sdl2.SDLK_o:
                    del self.pidAdjustAttributes['Kd']
                elif event.key.keysym.sym == sdl2.SDLK_l:
                    del self.pidAdjustAttributes['Kd']
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_w:
                    self.adjustAttributes['sat_thresh'] = True
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_s:
                    self.adjustAttributes['sat_thresh'] = False
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_a:
                    self.adjustAttributes['bright_thresh'] = False
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_d:
                    self.adjustAttributes['bright_thresh'] = True
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_u:
                    self.pidAdjustAttributes['Kp'] = True
                elif event.key.keysym.sym == sdl2.SDLK_j:
                    self.pidAdjustAttributes['Kp'] = False
                elif event.key.keysym.sym == sdl2.SDLK_i:
                    self.pidAdjustAttributes['Ki'] = True
                elif event.key.keysym.sym == sdl2.SDLK_k:
                    self.pidAdjustAttributes['Ki'] = False
                elif event.key.keysym.sym == sdl2.SDLK_o:
                    self.pidAdjustAttributes['Kd'] = True
                elif event.key.keysym.sym == sdl2.SDLK_l:
                    self.pidAdjustAttributes['Kd'] = False
                elif event.key.keysym.sym == sdl2.SDLK_UP:
                    self.moveDirFlags[0] = True
                    self.moveDirFlags[2] = False
                    self.ptzSpeed = 4
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                    self.moveDirFlags[0] = False
                    self.moveDirFlags[2] = True
                    self.ptzSpeed = 4
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                    self.moveDirFlags[1] = False
                    self.moveDirFlags[3] = True
                    self.ptzSpeed = 4
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    self.moveDirFlags[1] = True
                    self.moveDirFlags[3] = False
                    self.ptzSpeed = 4
                    self.updatePTZCommand()
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_EQUALS:
                    self.moveDirFlags[4] = True
                    self.moveDirFlags[5] = False
                    self.ptzSpeed = 7
                    self.updatePTZCommand()
                    # cam.ptzControl(channel, 11, False, 1)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_MINUS:
                    self.moveDirFlags[4] = False
                    self.moveDirFlags[5] = True
                    self.ptzSpeed = 7
                    self.updatePTZCommand()
                    # cam.ptzControl(channel, 12, False, 1)
                    return True
                elif event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    self.paused = not self.paused
                    return True
            return False
        except Exception as e:
            print(e)
            return False

    def nms_detections(self, lastIdentify, identifyConfidence, identifyObjects):
        if lastIdentify != "person":
            return []

        if len(identifyObjects) >= 3:
            confidence_sorted = []
            for confidence_id in range(len(identifyConfidence)):
                confidence_sorted.append([identifyConfidence[confidence_id], identifyObjects[confidence_id]])
            confidence_sorted.sort(key=lambda x: x[0], reverse=True)

            sorted_result = np.array(confidence_sorted[:2], dtype=object)
            identifyConfidence = sorted_result[:, 0].tolist()
            identifyObjects = sorted_result[:, 1].tolist()
            # print("Ignore ",lastIdentify, identifyObjects, identifyObjects, identifyConfidence)

        # Darknet Detect Result
        # Center X, Center Y, Width, Height
        identifyObjects = np.asarray(identifyObjects, dtype=float)
        identifyObjects[:, 0:2] -= identifyObjects[:, 2:4] / 2
        identifyObjects[:, 2:4] += identifyObjects[:, 0:2]

        # Convert To x1, y1, x2, y2 Rect for NMS
        pick = imutils.object_detection.non_max_suppression(identifyObjects, probs=None,
                                                            overlapThresh=0.1)

        cw = (np.max(identifyObjects[:, 2]) - np.min(identifyObjects[:, 0]))
        ch = (np.max(identifyObjects[:, 3]) - np.min(identifyObjects[:, 1]))
        mid_points_x = np.min(identifyObjects[:, 0]) + cw / 2
        mid_points_y = np.min(identifyObjects[:, 1]) + ch / 2

        rc = []
        for rect in pick:
            (x, y, w, h) = rect
            w = w - x
            h = h - y
            x = x + w / 2
            y = y + h / 2
            # Recovery to darknet format
            rc.append((lastIdentify, "%.02f" % np.average(identifyConfidence), (x, y, w, h)))
        return rc

    def run(self):
        for th in self.threads:
            while th.darknet_width is None:
                time.sleep(0.1)

        lastDetect = 0
        lastAnalyize = 0
        moveDutyCycle = 0
        moveDutyReset = time.time()
        moveDutyCheck = 0
        white = None
        while not terminated.get():
            if self.frame is not None:
                self.mutex.acquire()
                if self.frame.shape[1] > self.max_frame_width or self.frame.shape[0] > self.max_frame_height:
                    frame_width = self.max_frame_width
                    frame_height = self.max_frame_height
                else:
                    frame_width = self.frame.shape[1]
                    frame_height = self.frame.shape[0]

                frame = self.frame
                self.frame = None
                for adjustKey in self.adjustAttributes:
                    self.__dict__[adjustKey] += 1 if self.adjustAttributes[adjustKey] else -1
                    if self.__dict__[adjustKey] > 255:
                        self.__dict__[adjustKey] = 255
                    elif self.__dict__[adjustKey] < 0:
                        self.__dict__[adjustKey] = 0
                    print("Set %s -> %d" % (adjustKey, self.__dict__[adjustKey]))

                for adjustKey in self.pidAdjustAttributes:
                    self.xPID.__dict__[adjustKey] += 0.05 if self.pidAdjustAttributes[adjustKey] else -0.05
                    if self.xPID.__dict__[adjustKey] > 255:
                        self.xPID.__dict__[adjustKey] = 255
                    elif self.xPID.__dict__[adjustKey] < 0:
                        self.xPID.__dict__[adjustKey] = 0
                    self.yPID.__dict__[adjustKey] = self.xPID.__dict__[adjustKey]
                    print("Set %s -> %.02f" % (adjustKey, self.xPID.__dict__[adjustKey]))

                self.mutex.release()

                now = time.time()
                # if True or not self.artnet.cal_mode and self.artnet.test_mode is None:
                if not self.paused:
                    prev_time = time.time()

                    # frame = frame[120:720, 0:1066, :]

                    self.lastFrame = frame

                    analyizeDebugColor = (255, 0, 0)
                    if lastDetect is None and now - lastAnalyize < 0.5 and self.displayThreshold == 0:
                        analyizeDebugColor = (0, 255, 0)
                    else:
                        if self.useCudaProcess:
                            if white is None:
                                gpu_resize_frame = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC3)
                                gpu_frame = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC3)
                                gpu_contract_frame = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC3)
                                gpu_zero = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC3)
                                gpu_white = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1)
                                hsv_bin = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1)
                                gpu_blur = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1)
                                gpu_hsv = cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC3)
                                gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                                d_hsv = [
                                    cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1),
                                    cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1),
                                    cv2.cuda_GpuMat(frame_height, frame_width, cv2.CV_8UC1)
                                ]
                                gpu_zero.upload(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

                            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                                gpu_resize_frame.upload(frame)
                                cv2.cuda.resize(gpu_resize_frame, (frame_width, frame_height), gpu_contract_frame)
                            else:
                                gpu_contract_frame.upload(frame)

                            if self.alpha != 1 or self.beta != 0:
                                cv2.cuda.addWeighted(gpu_contract_frame, self.alpha, gpu_zero, 0, self.beta, gpu_frame)
                                cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2HSV, gpu_hsv)
                                frame = gpu_frame.download()
                            else:
                                cv2.cuda.cvtColor(gpu_contract_frame, cv2.COLOR_RGB2HSV, gpu_hsv)
                                frame = gpu_contract_frame.download()

                            cv2.cuda.split(gpu_hsv, d_hsv)

                            cv2.cuda.threshold(d_hsv[1], self.sat_thresh, 1, cv2.THRESH_BINARY_INV,
                                               hsv_bin)  # white = np.where(hsv[:, :, 1] < 50, hsv[:, :, 2], 0)
                            cv2.cuda.multiply(d_hsv[2], hsv_bin, gpu_white)
                            cv2.cuda.threshold(d_hsv[0], self.bright_thresh, 1, cv2.THRESH_BINARY_INV,
                                               hsv_bin)  # white = np.where(hsv[:, :, 0] < 40, white, 0)
                            cv2.cuda.multiply(gpu_white, hsv_bin, gpu_white)

                            gpu_gaussian.apply(gpu_white, gpu_blur)
                            retval, thresh_image = cv2.threshold(gpu_blur.download().astype(np.uint8), 0, 255,
                                                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        else:
                            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                                frame = cv2.resize(self.frame, (frame_width, frame_height))
                            frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
                            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                            retval, sat_bin = cv2.threshold(hsv[:, :, 1], self.sat_thresh, 1,
                                                            cv2.THRESH_BINARY_INV)  # white = np.where(hsv[:, :, 1] < 50, hsv[:, :, 2], 0)
                            retval, hue_bin = cv2.threshold(hsv[:, :, 0], self.bright_thresh, 1,
                                                            cv2.THRESH_BINARY_INV)  # white = np.where(hsv[:, :, 0] < 40, white, 0)
                            white = hsv[:, :, 2] * sat_bin * hue_bin
                            blur = cv2.GaussianBlur(white, (5, 5), 0)
                            retval, thresh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # print("USing ",1000*(time.time()-prev_time))
                        boundingBoxing = []

                        self.detections_adjusted = []
                        self.t_detect = 0
                        self.detection_done = 0
                        detection_queue = 0

                        x_expend = 60
                        y_expend = 60

                        final_contours = None

                        if self.displayThreshold == 1:
                            draw_frame = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
                        elif self.displayThreshold == 2:
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
                            opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=1)

                            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            dilate = cv2.dilate(opening, dilate_kernel, iterations=4)

                            draw_frame = cv2.cvtColor(dilate, cv2.COLOR_GRAY2RGB)
                        elif self.displayThreshold == 3:
                            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                            white = np.where(hsv[:, :, 1] < self.sat_thresh, hsv[:, :, 2], np.zeros_like(hsv[:, :, 2]))
                            white = np.where(hsv[:, :, 0] < self.bright_thresh, white, np.zeros_like(white))
                            draw_frame = cv2.cvtColor(white, cv2.COLOR_GRAY2RGB)
                        else:
                            draw_frame = frame

                        if detailAnalyizeMode == 1:
                            lastAnalyize = now
                            final_pick = []

                            for x in range(0, frame.shape[1], self.darknet_width - x_expend):
                                for y in range(0, frame.shape[0], self.darknet_height - y_expend):
                                    x2 = x + self.darknet_width if x + self.darknet_width < frame.shape[1] else frame.shape[
                                        1]
                                    y2 = y + self.darknet_height if y + self.darknet_height < frame.shape[0] else \
                                    frame.shape[0]
                                    x -= x_expend if x > x_expend else 0
                                    y -= y_expend if y > y_expend else 0
                                    x2 += x_expend if x2 < frame.shape[1] - x_expend else 0
                                    y2 += y_expend if y2 < frame.shape[0] - y_expend else 0

                                    if y2 - y != x2 - x:
                                        diff = (y2 - y) - (x2 - x)
                                        if diff > 0:
                                            if x2 + diff <= frame.shape[1]:
                                                x2 += diff
                                            elif x - diff >= 0:
                                                x -= diff
                                        elif diff < 0:
                                            if y2 - diff <= frame.shape[0]:
                                                y2 -= diff
                                            elif y + diff >= 0:
                                                y += diff
                                        # print("Fixed: Scale %3s matched %4d, %4d, %4d, %4d   S: %dx%d  I: %dx%d Diff %d  D: %dx%d" % ("is" if (y2-y)==(x2-x) else "not", x, y, x2, y2, x2 - x, y2 - y, frame.shape[1], frame.shape[0], diff,  self.self.darknet_width, self.darknet_height))

                                    final_pick.append((x, y, x2, y2))

                                    pick_frame = frame[y:y2, x:x2, :]

                                    queue = QueueFrame(pick_frame, (x, y, x2, y2))

                                    detection_queue += 1
                                    self.mutex.acquire()
                                    self.queue.put(queue)
                                    self.mutex.release()

                            if self.displayThreshold == 1:
                                for i in range(len(final_pick)):
                                    (x, y, w, h) = final_pick[i]
                                    cv2.putText(draw_frame, "%d %d %d %d" % (x, y, w, h), (x, h),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1,
                                                (i * 40 % 255, 0, 0),
                                                1,
                                                1)
                                    cv2.rectangle(draw_frame, pt1=(x, y), pt2=(w, h), color=(i * 40 % 255, 0, 0),
                                                  thickness=1)

                        elif detailAnalyizeMode == 2:
                            lastAnalyize = now
                            # Split by contours
                            contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            final_contours = []

                            for c in contours:
                                if cv2.contourArea(c) >= minCardAreaRatio * frame.shape[0] * frame.shape[1]:
                                    peri = cv2.arcLength(c, True)
                                    approx = cv2.approxPolyDP(c, approxThresh * peri, True)
                                    approxRatio = cv2.contourArea(approx) / cv2.contourArea(c)
                                    approxRect = cv2.boundingRect(approx)
                                    if self.displayThreshold == 1:
                                        cv2.drawContours(draw_frame, [approx], -1, (255, 160, 0), 4)
                                        cv2.putText(draw_frame,
                                                    "R %.02f E %d" % (
                                                    cv2.contourArea(approx) / cv2.contourArea(c), len(approx)),
                                                    (int(approxRect[0] + approxRect[2] / 2), approxRect[1] + int(approxRect[3]/2)),
                                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    fontScale=1, color=(160, 255, 0), thickness=2)

                                    if len(approx) <= maxAllowShape:
                                        if approxRatio > acceptApproxContourRange[1] or approxRatio < \
                                                acceptApproxContourRange[0]:
                                            continue
                                        if approxRect[2] < minCardSizeRatio * frame.shape[1] or approxRect[
                                            3] < minCardSizeRatio * frame.shape[0]:
                                            continue
                                        if approxRect[2] / approxRect[3] > maxCardSizeRatio or approxRect[3] / approxRect[
                                            2] > maxCardSizeRatio:
                                            continue
                                        final_contours.append(c)
                                        boundingBoxing.append(approxRect)
                                        if self.displayThreshold == 1:
                                            cv2.drawContours(draw_frame, [approx], -1, (160, 255, 0), 4)

                            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boundingBoxing])

                            pick = imutils.object_detection.non_max_suppression(rects, probs=None, overlapThresh=0.1)
                            if not isinstance(pick, list):
                                pick = pick.tolist()
                            pick.sort(
                                key=lambda x: (x[1] - x[1] % int(frame.shape[0] / 10)) * 10000 + x[0])  # Sort by Y offset
                            final_pick = []

                            queue_pick = []
                            for i in range(0, len(pick)):
                                queue_pick.append(pick[i])
                                #
                                while len(queue_pick) > 0:
                                    qp = np.array(queue_pick)
                                    w = max(qp[:, 2]) - min(qp[:, 0])
                                    h = max(qp[:, 3]) - min(qp[:, 1])

                                    if w >= self.darknet_width or h >= self.darknet_height or i == len(pick) - 1:
                                        if len(queue_pick) > 1:
                                            queue_pick.pop()
                                            qp = np.array(queue_pick)

                                            x = min(qp[:, 0])
                                            y = min(qp[:, 1])
                                            x2 = max(qp[:, 2])
                                            y2 = max(qp[:, 3])

                                            queue_pick = [pick[i]]
                                        else:
                                            x, y, x2, y2 = pick[i]
                                            queue_pick = []

                                        if y2 - y != x2 - x:
                                            diff = (y2 - y) - (x2 - x)
                                            if diff > 0:
                                                if x2 + diff <= frame.shape[1]:
                                                    x2 += diff
                                                elif x - diff >= 0:
                                                    x -= diff
                                            elif diff < 0:
                                                if y2 - diff <= frame.shape[0]:
                                                    y2 -= diff
                                                elif y + diff >= 0:
                                                    y += diff

                                        final_pick.append((x, y, x2, y2))

                                        pick_frame = frame[y:y2, x:x2, :]
                                        queue = QueueFrame(pick_frame, (x, y, x2, y2))

                                        detection_queue += 1
                                        self.mutex.acquire()
                                        self.queue.put(queue)
                                        self.mutex.release()
                                    else:
                                        break

                            # for i in range(len(final_pick)):
                            #     (x, y, w, h) = final_pick[i]
                            #     cv2.putText(frame, "%d x %d" % (w - x, h - y), (x, h), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(128, 128, 0), fontScale=1, thickness=2)
                            #     cv2.rectangle(frame, pt1=(x, y), pt2=(w, h), color=(128, 0, 0), thickness=2)

                        else:
                            lastAnalyize = now
                            queue = QueueFrame(frame, None)
                            detection_queue += 1
                            self.mutex.acquire()
                            self.queue.put(queue)
                            self.mutex.release()

                        ti = time.time() - prev_time

                        while self.detection_done < detection_queue:
                            # print("%3d / %3d" % (self.detection_done, detection_queue), end="\r")
                            time.sleep(0.001)

                    self.detections_adjusted.sort(key=lambda x: x[0])
                    lastIdentify = None
                    identifyObjects = []
                    identifyConfidence = []
                    draw_detections = []

                    for index in range(len(self.detections_adjusted)):
                        (label, confidence, bbox) = self.detections_adjusted[index]
                        (cx, cy, cw, ch) = bbox
                        if cw == 0 or ch == 0:
                            continue

                        if label != lastIdentify or index == len(self.detections_adjusted) - 1:
                            if index == len(self.detections_adjusted) - 1:
                                lastIdentify = label
                                identifyConfidence.append(float(confidence))
                                identifyObjects.append(bbox)

                            if lastIdentify is not None:
                                draw_detections = draw_detections + self.nms_detections(lastIdentify,
                                                                                        identifyConfidence,
                                                                                        identifyObjects)
                                identifyObjects = []
                                identifyConfidence = []

                            lastIdentify = label
                        identifyConfidence.append(float(confidence))
                        identifyObjects.append(bbox)

                    self.detectResult = draw_detections

                    if final_contours is not None:
                        contours = final_contours
                    else:
                        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        if cv2.contourArea(c) >= minCardAreaRatio * draw_frame.shape[0] * draw_frame.shape[1]:
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, self.approx_thresh * peri, True)
                            if final_contours is not None or len(approx) <= maxAllowShape:
                                approxRect = cv2.boundingRect(approx)
                                if final_contours is None:
                                    if approxRect[2] < minCardSizeRatio * draw_frame.shape[1] or approxRect[3] < minCardSizeRatio * draw_frame.shape[0]:
                                        continue

                                    if approxRect[2] / approxRect[3] > maxCardSizeRatio or approxRect[3] / approxRect[2] > maxCardSizeRatio:
                                        continue

                                cv2.drawContours(draw_frame, [approx], -1, (255, 0, 0),
                                                     1)  # ---set the last parameter to -1

                    if self.detection_done == 0xff:
                        continue

                    if time.time() - moveDutyReset >= 60:
                        moveDutyReset = time.time()
                        moveDutyCycle = 0

                    if len(draw_detections) > 0:
                        persons_bbox = np.array([[bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] for labels, cofidence, bbox in draw_detections])
                        w = np.max(persons_bbox[:, 2]) - np.min(persons_bbox[:, 0])
                        h = np.max(persons_bbox[:, 3]) - np.min(persons_bbox[:, 1])
                        cx = int(np.min(persons_bbox[:, 0]) + w / 2)
                        cy = int(np.min(persons_bbox[:, 1]) + h / 2)

                        cv2.line(draw_frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 5)
                        cv2.line(draw_frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 5)
                        x_error = cx / draw_frame.shape[1] - 0.5
                        y_error = cy / draw_frame.shape[0] - 0.5
                        z_error = max(w / draw_frame.shape[1], h / draw_frame.shape[0])
                        # print(np.min(persons_bbox[:, 0]), np.min(persons_bbox[:, 1]), np.max(persons_bbox[:, 2]), np.max(persons_bbox[:, 3]))

                        if moveDutyCycle < 20:
                            self.xPID.set_auto_mode(True)
                            self.yPID.set_auto_mode(True)
                            self.zoomPID.set_auto_mode(True)

                            ctrl_x = self.xPID(x_error)
                            ctrl_y = self.yPID(y_error)
                            ctrl_z = self.zoomPID(z_error)
                            if ctrl_x >= 7:
                                ctrl_x = 7
                            elif ctrl_x <= -7:
                                ctrl_x = -7
                            if ctrl_y >= 7:
                                ctrl_y = 7
                            elif ctrl_y <= -7:
                                ctrl_y = -7

                            self.ptzSpeed = int(min(abs(ctrl_x), abs(ctrl_y)) + abs(abs(ctrl_x)-abs(ctrl_y)) / 2)
                            if self.ptzSpeed == 0:
                                self.ptzSpeed = 1
                            self.moveDirFlags[2] = ctrl_y <= -self.ptzSpeed / 2
                            self.moveDirFlags[0] = ctrl_y >= self.ptzSpeed / 2
                            self.moveDirFlags[1] = ctrl_x <= -self.ptzSpeed / 2
                            self.moveDirFlags[3] = ctrl_x >= self.ptzSpeed / 2
                            self.moveDirFlags[4] = ctrl_z >= self.ptzSpeed / 2
                            self.moveDirFlags[5] = ctrl_z <= -self.ptzSpeed / 2

                            if not self.fullDuplexZoom and (self.moveDirFlags[4] or self.moveDirFlags[5]) and abs(ctrl_z) >= self.ptzSpeed:
                                print("Force Execute Zoom")
                                self.moveDirFlags[0] = False
                                self.moveDirFlags[1] = False
                                self.moveDirFlags[2] = False
                                self.moveDirFlags[3] = False
                                self.ptzSpeed = int(abs(ctrl_z))
                            elif not self.fullDuplexZoom:
                                self.moveDirFlags[4] = False
                                self.moveDirFlags[5] = False

                            if functools.reduce(lambda a,b: a|b, self.moveDirFlags) and time.time() - moveDutyCheck >= 1:
                                moveDutyCheck = time.time()
                                moveDutyCycle += 1

                            try:
                                print("%s -> X %.02f Err %.03f / Y %.02f Err %.03f / Z %.02f Err %.03f / Speed %d                 "  % (datetime.datetime.now(), ctrl_x,x_error,  ctrl_y, y_error, ctrl_z, z_error, self.ptzSpeed), end="\r")
                                self.updatePTZCommand()
                            except Exception as e:
                                if e == "NET_DVR_PTZControlWithSpeed_Other error, 23: Device does not support this function.":
                                    self.fullDuplexZoom = False
                                print(e)
                                pass
                        else:
                            if functools.reduce(lambda a,b: a|b, self.moveDirFlags):
                                self.moveDirFlags[0] = False
                                self.moveDirFlags[2] = False
                                self.moveDirFlags[1] = False
                                self.moveDirFlags[3] = False
                                self.moveDirFlags[4] = False
                                self.moveDirFlags[5] = False
                                try:
                                    self.updatePTZCommand()
                                except Exception as e:
                                    print(e)
                                    pass
                            print("%s X %.02f Err %.03f / Y %.02f Err %.03f / Z %.02f Err %.03f / Speed %d Ignored         " % (
                                datetime.datetime.now(), ctrl_x, x_error, ctrl_y, y_error, ctrl_z, z_error,
                                self.ptzSpeed), end="\r")

                        lastDetect = time.time()
                    elif lastDetect is not None and time.time() - lastDetect >= 5:
                        lastDetect = None
                        self.xPID.reset()
                        self.yPID.reset()
                        self.zoomPID.reset()
                        try:
                            self.cam.ptzPreset(self.channel, 1)
                        except Exception as e:
                            print(e)
                            pass
                    elif lastDetect is not None and time.time() - lastDetect >= 1 and time.time() - lastDetect < 2:
                        self.moveDirFlags[0] = False
                        self.moveDirFlags[2] = False
                        self.moveDirFlags[1] = False
                        self.moveDirFlags[3] = False
                        self.moveDirFlags[4] = False
                        self.moveDirFlags[5] = False
                        try:
                            self.updatePTZCommand()
                        except Exception as e:
                            print(e)
                            pass
                        if self.ptzOperating[4]:
                            self.cam.ptzControl(self.channel, 11, True)
                        elif self.ptzOperating[5]:
                            self.cam.ptzControl(self.channel, 12, True)
                        self.xPID.set_auto_mode(False)
                        self.yPID.set_auto_mode(False)


                    image = darknet.draw_boxes(draw_detections,
                                               draw_frame,
                                               self.threads[0].class_colors)

                    output_str = '%s FPS %d  T i %d / g %d / t %d ms  Seg %d  Detected: %d Weight: %s' % (
                        self.backend, int(1 / (time.time() - prev_time)), int(ti * 1000),
                        self.t_detect * 1000,
                        (time.time() - prev_time) * 1000,
                        self.detection_done,
                        len(self.detectResult),
                        weightFile.split('/')[-1]
                    )

                    cv2.putText(image, output_str,
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                3,
                                2)
                    cv2.putText(image, output_str,
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                analyizeDebugColor,
                                1,
                                2)

                    self.gui.mutex.acquire()
                    self.gui.frame = image
                    self.gui.mutex.release()
                else:
                    # self.moveDirFlags[0] = False
                    # self.moveDirFlags[2] = False
                    # self.moveDirFlags[1] = False
                    # self.moveDirFlags[3] = False
                    # self.moveDirFlags[4] = False
                    # self.moveDirFlags[5] = False
                    # try:
                    #     self.updatePTZCommand()
                    # except Exception as e:
                    #     print(e)
                    #     pass
                    lastDetect = None
                    self.xPID.reset()
                    self.yPID.reset()
                    self.zoomPID.reset()
                    self.gui.mutex.acquire()
                    self.gui.frame = frame
                    self.gui.mutex.release()

            time.sleep(0.01)


lastCheck = 0

try:
    opts, _ = getopt.getopt(sys.argv[1:], "u:p:H:c:", ["user=", "passwd=", "ip=", "channel="])
    ip = None
    user = None
    passwd = None
    channel = 1
    for opt, arg in opts:
        if opt in ["-u", "--user"]:
            user = arg
        if opt in ["-p", "--passwd"]:
            passwd = arg
        if opt in ["-H", "--ip"]:
            ip = arg
        if opt in ["-c", "--channel"]:
            channel = int(arg)
except getopt.GetoptError:
    show_help()
    sys.exit(2)

# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/channels/101" % (user, passwd, ip))
# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/channels/302" % (user, passwd, ip))

cam = hikevent.hikevent(ip, user, passwd)

analyizeThread = AnalyizeThread(cam, channel)
analyizeThread.start()
#
# cap = cv2.VideoCapture("rtsp://%s:%s@%s/Streaming/tracks/4001?starttime=20220324T182930z" % (user, passwd, ip))
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#
# while not terminated.get():
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if time.time() - lastCheck >= 0.05:
#         lastCheck = time.time()
#         analyizeThread.mutex.acquire()
#         analyizeThread.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         analyizeThread.mutex.release()
#     else:
#         time.sleep(0.001)

cam.startRealPlay(channel, 0)

while not terminated.get():
    # Capture frame-by-frame
    evt = cam.getevent()
    if evt is not None:
        if evt['command'] == "DVR_VIDEO_DATA":
            size = struct.unpack("=LL", evt['payload'][0:8])
            if time.time() - lastCheck >= 0.1:
                frame = np.frombuffer(evt['payload'][8:], dtype=np.uint8).reshape((size[1], size[0], 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

                # frame = cv2.imread("cam.jpg")
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lastCheck = time.time()
                analyizeThread.mutex.acquire()
                analyizeThread.frame = frame
                analyizeThread.mutex.release()
    else:
        time.sleep(0.001)
