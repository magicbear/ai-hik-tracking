import numpy as np
import os, json, cv2, random
import threading
import time, datetime
import hikevent
import struct
import darknet
import imutils
import sdl2
from imutils import contours, perspective
from imutils.object_detection import non_max_suppression
from queue import Queue
import colorsys

import base64
import math
from labelme import utils
import sys, getopt
from laser_control import ArtNetThread, GUIThread, TerminatedState, OpenCV_dnnThread, DarknetThread, QueueFrame, nms_detections

terminated = TerminatedState()

projection_ratio = 1.0
minCardSizeRatio = 0.04
minCardAreaRatio = 0.04 * 0.08  # Card Area size required
maxCardSizeRatio = 5  # Width / Height Ratio for overlap Cards
approxThresh = 0.04  # Approx of edges for split
maxAllowShape = 8  # Max allow shape for split
acceptApproxContourRange = [0.8, 1.1]
detailAnalyizeMode = 1  # 0 No Split    1 Split by Block   2 Split by Object
cvCudaProcess = cv2.cuda.getCudaEnabledDeviceCount() > 0

data_file = 'data/cards.data'
cfg_file = None
weightFile = None

def mapfloat(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class AnalyizeThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.mutex = threading.Lock()
        self.frame = None
        self.artnet = ArtNetThread(terminated, '192.168.20.16')
        self.artnet.start()
        self.gui = GUIThread(self.artnet, self, terminated)
        self.gui.start()
        self.gui.onSDL_Event = self.onSDL_Event
        self.darknet_height = 0
        self.darknet_width = 0
        self.lastFrame = None
        self.detectResult = []
        self.displayThreshold = 0
        self.alpha = 1.3
        self.beta = -40
        self.approx_thresh = 0.01  # Approx For Build shape (not for split)
        self.detection_done = 0
        self.t_detect = 0
        self.queue = Queue()
        self.threads = []
        self.detections_adjusted = []  # Offset: Center X, Center Y,
        self.available_cards = []
        self.useCudaProcess = cvCudaProcess
        self.backend = "darknet"
        self.default_pref = cv2.dnn.DNN_TARGET_CUDA_FP16
        self.load_dnn_networks()
        self.sat_thresh = 60
        self.bright_thresh = 60
        self.max_frame_width = 1280
        self.max_frame_height = 720
        self.adjustAttributes = {}

    def load_dnn_networks(self, load_cfg=0):
        global cfg_file, weightFile
        config_set = [
            ['data/cfg/yolov4-cards.cfg', 'backup/ai-fake-real-yolov4-cards_30000.weights'],
            ['data/cfg/yolov4-tiny-cards.cfg', 'backup/ai-mixed-yolov4-tiny-cards_final.weights'],
            ['data/cfg/yolov4-tiny-3l-cards.cfg', 'backup/ai-mixed-yolov4-tiny-3l-cards_final.weights'],
            ['data/cfg/yolov4-cards.cfg', 'backup/yolov4-cards_last.weights'],
            ['data/cfg/yolov4-cards.cfg', 'backup/yolov4-cards_best.weights'],
            ['data/cfg/yolov4-cards.cfg', 'backup/yolov4-cards_10000.weights'],
            ['data/cfg/yolov4-cards.cfg', 'backup/yolov4-cards_20000.weights'],
            ['data/cfg/yolov4-cards.cfg', 'backup/yolov4-cards_30000.weights'],
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
        else:
            for i in range(4):
                th_darknet = OpenCV_dnnThread(self, terminated, cfg_file, data_file, weightFile)
                th_darknet.start()
                self.threads.append(th_darknet)

        self.darknet_width = self.threads[0].darknet_width
        self.darknet_height = self.threads[0].darknet_height

        self.mutex.release()

    def onSDL_Event(self, event):
        if event.type == sdl2.SDL_KEYUP:
            if event.key.keysym.sym >= sdl2.SDLK_1 and event.key.keysym.sym <= sdl2.SDLK_9:
                global cfg_file, weightFile
                if event.key.keysym.mod & sdl2.KMOD_CTRL:
                    self.backend = "opencv" if self.backend == "darknet" else "darknet"

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
        return False

    def nms_detections(self, lastIdentify, identifyConfidence, identifyObjects, card_location):
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

        card_location[lastIdentify] = (
        float(mid_points_x), float(mid_points_y), float(cw), float(ch), np.max(identifyConfidence))
        #
        # # Obtain birds' eye view of image
        # if displayCnt is not None:
        #     warped = perspective.four_point_transform(gray, displayCnt.reshape(4, 2))
        #
        #     # print(edges)
        #     cv2.imshow(label, warped)
        #     cv2.waitKey(1)

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

    def getcardsize(self, card):
        if card[0] == '1':
            return 10
        elif card[0] == 'A':
            return 1
        elif card[0] == 'J':
            return 11
        elif card[0] == 'Q':
            return 12
        elif card[0] == 'K':
            return 13
        return int(card[0])

    def find_rect_cards(self, rect):
        # rect -> boundingRect  (x, y, w, h)
        insideCards = []
        for c in self.detections_adjusted:
            (dc_x, dc_y, dc_w, dc_h) = c[2]
            if dc_x >= rect[0] and dc_x <= rect[0] + rect[2] and dc_y >= rect[1] and dc_y <= rect[1] + rect[3]:
                insideCards.append(c)

        return insideCards

    def calc_rect_cards_possible(self, frame, rect, card, matched_card):
        equalsize_count = 0
        outofrange_count = 0
        start_size = min(self.getcardsize(card), self.getcardsize(matched_card))
        end_size = max(self.getcardsize(card), self.getcardsize(matched_card))
        start_card = matched_card if self.getcardsize(card) > self.getcardsize(
            matched_card) else card
        end_card = matched_card if start_card == card else card
        for avail_card in self.available_cards:
            if self.getcardsize(avail_card) == self.getcardsize(card) or self.getcardsize(
                    avail_card) == self.getcardsize(matched_card):
                equalsize_count += 1
            if self.getcardsize(avail_card) <= start_size or self.getcardsize(
                    avail_card) >= end_size:
                outofrange_count += 1

        return

        cv2.putText(frame, "Pay %.02f%%" % (outofrange_count / len(self.available_cards) * 100),
                    (int(rect[0]), int(rect[3] - 50)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255),
                    1,
                    1)
        cv2.putText(frame, "Double %.02f%%" % (equalsize_count / len(self.available_cards) * 100),
                    (int(rect[0]), int(rect[3] - 30)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255),
                    1,
                    1)
        cv2.putText(frame, start_card + " / " + end_card, (int(rect[0]), int(rect[3])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    1)
        cv2.putText(frame, "Pay %.02f%%" % (outofrange_count / len(self.available_cards) * 100),
                    (int(rect[0]) - 1, int(rect[3] - 50) - 1), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 0, 0),
                    1,
                    1)
        cv2.putText(frame, "Double %.02f%%" % (equalsize_count / len(self.available_cards) * 100),
                    (int(rect[0]) - 1, int(rect[3] - 30) - 1), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 0, 0),
                    1,
                    1)
        cv2.putText(frame, start_card + " / " + end_card, (int(rect[0]) - 1, int(rect[3]) - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    1)

    def run(self):
        cards = open('data/cards.names', 'r').read().strip().split("\n")
        class_colors = {
            cards[i]: tuple((np.array(colorsys.hsv_to_rgb(i/len(cards), 0.65, 0.6))*255).tolist()) for i in range(len(cards))
        }

        for th in self.threads:
            while th.darknet_width is None:
                time.sleep(0.1)

        white = None
        while not terminated.get():
            if self.frame is not None:
                self.mutex.acquire()
                # if self.frame.shape[1] > 1280:
                #     frame = cv2.resize(self.frame, (1280, 720))
                # else:
                #
                frame = self.frame
                self.frame = None
                for adjustKey in self.adjustAttributes:
                    self.__dict__[adjustKey] += 1 if self.adjustAttributes[adjustKey] else -1
                    if self.__dict__[adjustKey] > 255:
                        self.__dict__[adjustKey] = 255
                    elif self.__dict__[adjustKey] < 0:
                        self.__dict__[adjustKey] = 0
                    print("Set %s -> %d" % (adjustKey, self.__dict__[adjustKey]))

                self.mutex.release()

                if not self.artnet.cal_mode and self.artnet.test_mode is None:
                    poslist = []
                    prev_time = time.time()

                    # frame = frame[120:720, 0:1066, :]

                    self.lastFrame = frame

                    if self.useCudaProcess:
                        if white is None:
                            gpu_frame = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC3)
                            gpu_contract_frame = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC3)
                            gpu_zero = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC3)
                            gpu_white = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1)
                            hsv_bin = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1)
                            gpu_blur = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1)
                            gpu_hsv = cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC3)
                            gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                            d_hsv = [
                                cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1),
                                cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1),
                                cv2.cuda_GpuMat(frame.shape[0], frame.shape[1], cv2.CV_8UC1)
                            ]
                            gpu_zero.upload(np.zeros_like(frame))

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
                    card_location = {}

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
                                                                                        identifyObjects,
                                                                                        card_location)
                                identifyObjects = []
                                identifyConfidence = []

                            lastIdentify = label
                        identifyConfidence.append(float(confidence))
                        identifyObjects.append(bbox)

                    self.detectResult = draw_detections

                    card_distance = {}
                    standard_distance = []
                    card_distance_ordering = []
                    for card, offset in card_location.items():
                        card_distance[card] = []
                        cv2.putText(frame, card, (int(offset[0]), int(offset[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1,
                                    (255, 192, 0),
                                    1,
                                    1)
                        for card2, offset2 in card_location.items():
                            if card == card2:
                                continue
                            card_distance[card].append({
                                'card': card2,
                                'distance': math.sqrt(
                                    math.pow(offset[0] - offset2[0], 2) + math.pow(offset[1] - offset2[1], 2))
                            })

                        if len(card_distance[card]) > 0:
                            card_distance[card].sort(key=lambda x: x['distance'])
                            card_distance_ordering.append({
                                'card': card,
                                'card2': card_distance[card][0]['card'],
                                'distance': card_distance[card][0]['distance']
                            })
                        if len(card_distance[card]) > 1:
                            standard_distance.append(card_distance[card][1]['distance'])

                    card_distance_ordering.sort(key=lambda x: x['distance'])

                    self.available_cards = []
                    for card in cards:
                        if card not in card_location:
                            self.available_cards.append(card)
                    self.available_cards.sort(key=self.getcardsize)

                    dumped_pair = {}
                    standard_distance = np.median(standard_distance)

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
                                    if approxRect[2] < minCardSizeRatio * draw_frame.shape[1] or approxRect[
                                        3] < minCardSizeRatio * draw_frame.shape[0]:
                                        continue

                                    if approxRect[2] / approxRect[3] > maxCardSizeRatio or approxRect[3] / approxRect[
                                        2] > maxCardSizeRatio:
                                        continue

                                find_cards = self.find_rect_cards(approxRect)
                                if len(find_cards) > 0:
                                    cv2.drawContours(draw_frame, [approx], -1, (0, 255, 0),
                                                     1)  # ---set the last parameter to -1

                                    if len(find_cards) > 1:
                                        find_cards.sort(key=lambda x: x[0])

                                        find_cards = np.array(find_cards, dtype=object)
                                        cards_name = find_cards[:, 0]
                                        card = cards_name[0]
                                        matched_card = cards_name[len(find_cards) - 1]
                                        dumped_pair[card] = True
                                        dumped_pair[matched_card] = True

                                        self.calc_rect_cards_possible(draw_frame, (
                                        approxRect[0], approxRect[1], approxRect[0] + approxRect[2],
                                        approxRect[1] + approxRect[3] / 2), card,
                                                                      matched_card)
                                else:
                                    cv2.drawContours(draw_frame, [approx], -1, (255, 0, 0),
                                                     1)  # ---set the last parameter to -1

                    for card_dist_order in card_distance_ordering:
                        card = card_dist_order['card']
                        distanceinfo = card_distance[card]
                        matched_card_pair = []
                        for dist_info in distanceinfo:
                            if dist_info['distance'] <= standard_distance:
                                if card_location[card][4] < 0.9 and card_location[card][4] < \
                                        card_location[dist_info['card']][4]:
                                    # Ignore Not good identify
                                    break
                            if dist_info['distance'] >= standard_distance * 1.5:
                                break
                            matched_card_pair.append(dist_info)

                        if len(matched_card_pair) >= 1:
                            matched_card = matched_card_pair[0]['card']
                            if card in dumped_pair or matched_card in dumped_pair:
                                continue
                            dumped_pair[card] = True
                            dumped_pair[matched_card] = True
                            bonding_x1 = min(card_location[card][0] - card_location[card][2] / 2,
                                             card_location[matched_card][0] - card_location[matched_card][2] / 2)
                            bonding_y1 = min(card_location[card][1] - card_location[card][3] / 2,
                                             card_location[matched_card][1] - card_location[matched_card][3] / 2)
                            bonding_x2 = max(card_location[card][0] + card_location[card][2] / 2,
                                             card_location[matched_card][0] + card_location[matched_card][2] / 2)
                            bonding_y2 = max(card_location[card][1] + card_location[card][3] / 2,
                                             card_location[matched_card][1] + card_location[matched_card][3] / 2)

                            self.calc_rect_cards_possible(draw_frame, (int(bonding_x1), int(bonding_y1), int(bonding_x2), int(bonding_y2)), card, matched_card)
                            # cv2.rectangle(draw_frame, (int(bonding_x1), int(bonding_y1)),
                            #               (int(bonding_x2), int(bonding_y2)),
                            #               (255, 0, 0), 2)

                    if self.detection_done == 0xff:
                        continue
                    image = darknet.draw_boxes(draw_detections,
                                               draw_frame,
                                               class_colors)
                    # image = darknet.draw_boxes(draw_detections, image, class_colors)

                    cv2.putText(image, '%s FPS %d  Ti %d ms Td %d ms Tg %d ms  Segment %d  Card: %d  Weight: %s' % (
                        self.backend, int(1 / (time.time() - prev_time)), int(ti * 1000),
                        (time.time() - prev_time) * 1000, self.t_detect * 1000,
                        self.detection_done, len(card_distance_ordering),
                        weightFile.split('/')[-1]
                    ),
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                1,
                                2)

                    self.gui.mutex.acquire()
                    self.gui.frame = image
                    self.gui.mutex.release()

                    self.artnet.mutex.acquire()
                    if not self.artnet.cal_mode and self.artnet.test_mode is None:
                        self.artnet.pos = poslist if len(poslist) > 0 else None
                    self.artnet.mutex.release()
                else:
                    self.gui.mutex.acquire()
                    self.gui.frame = draw_frame
                    self.gui.mutex.release()

            time.sleep(0.01)


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

                # frame = cv2.imread("cam.jpg")
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lastCheck = time.time()
                analyizeThread.mutex.acquire()
                analyizeThread.frame = frame
                analyizeThread.mutex.release()
    else:
        time.sleep(0.001)
