import sdl2
import os, json, cv2, random, threading
import socket
from stupidArtnet import StupidArtnet
import numpy as np
import struct
import ctypes
import datetime, time

class TerminatedState:
    def __init__(self):
        self.terminated = False

    def get(self):
        return self.terminated

    def set(self, term):
        self.terminated = term

class ArtNetThread(threading.Thread):
    def __init__(self, ts, target_ip, pelco_ip = None):
        threading.Thread.__init__(self)
        self.mutex = threading.Lock()
        self.pos = None
        self.cal_mode = False
        self.test_mode = False
        self.duty_cycle = 0
        self.zoom_size = 245
        self.color = 15
        self.sharp = 30
        self.shape1 = 44
        self.shape2 = 40
        self.ptz_preset = 2
        self.terminated_state = ts
        self.target_ip = target_ip
        self.pelco_ip = pelco_ip

    def build_pelco(self, command, pan_speed, tilt_speed):
        return struct.pack(">BBHBBB", 0xFF, 0x01, command, pan_speed, tilt_speed,
                           (0x01 + ((command >> 8) & 0xff) + (command & 0xff) + pan_speed + tilt_speed) % 256)

    def run(self):
        universe = 0  # see docs
        packet_size = 100  # it is not necessary to send whole universe

        a = StupidArtnet(self.target_ip, universe, packet_size, 30, True, True)

        if self.pelco_ip is not None:
            pelco_port = 26
            self.pelco_fd = socket.create_connection((self.pelco_ip, pelco_port))

            self.pelco_fd.send(self.build_pelco(0x07, 0x00, int(self.ptz_preset)))

        # CHECK INIT
        print(a)
        a.blackout()  # send single packet with all channels at 0

        while not self.terminated_state.get():
            self.mutex.acquire()
            poslist = self.pos
            self.mutex.release()

            if self.test_mode is not None:
                if time.time() > self.test_mode:
                    self.test_mode = None

            if poslist is not None:
                a.set_single_value(1, 255)  # set channel 1 to 255
                a.set_single_value(2, self.duty_cycle)  # Duty Cycle
                a.set_single_value(3, self.zoom_size)  # Zoom Size
                a.set_single_value(6, self.color)  # Color
                a.set_single_value(7, self.sharp)  # Sharp
                a.set_single_value(8, self.shape1)  # set channel 1 to 255 250, 40
                a.set_single_value(9, self.shape2)

                for pos in poslist:
                    if len(pos) == 6:
                        tpass = (time.time() - pos[4])
                        if tpass > pos[5]:
                            tpass = pos[5]
                        # pos[]
                        laser_x = pos[2] - (pos[2] - pos[0]) * (tpass / pos[5])
                        laser_y = pos[3] - (pos[3] - pos[1]) * (tpass / pos[5])
                        if tpass != pos[5]:
                            print("%d, %d  Real: %d, %d  Moving %3d%%" % (pos[0], pos[1], laser_x, laser_y, tpass / pos[5] * 100))
                    else:
                        laser_x = pos[0]
                        laser_y = pos[1]
                    if laser_x < 0:
                        laser_x = 0
                    if laser_x > 255:
                        laser_x = 255
                    if laser_y < 0:
                        laser_y = 0
                    if laser_y > 255:
                        laser_y = 255

                    a.set_single_value(4, int(laser_x))
                    a.set_single_value(5, int(laser_y))
                    a.show()  # send data
                    time.sleep(0.001)
            else:
                a.set_single_value(1, 0)  # set channel 1 to 255
                a.show()
                time.sleep(0.03)



class GUIThread(threading.Thread):
    def __init__(self, artnet, analyizer, ts):
        threading.Thread.__init__(self)
        self.mutex = threading.Lock()
        self.frame = None
        self.win = None
        self.ren = None
        self.analyizer = analyizer
        self.terminated_state = ts
        self.artnet = artnet
        self.cal_mtx = []
        self.window_width = 1920
        self.window_height = 1080
        cal_params = json.load(open("cam_cal.json", "r"))
        self.objpoints = cal_params['objpoints']
        self.imgpoints = cal_params['imgpoints']
        self.onSDL_Event = None
        self.cal()

    def cal(self):
        if self.artnet.ptz_preset >= len(self.objpoints) or len(self.objpoints[self.artnet.ptz_preset]) == 0:
            return
        # 标定、去畸变
        # 输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
        # 输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []
        for i in range(0, len(self.objpoints)):
            if len(self.objpoints[i]) == 0:
                self.mtx.append([])
                self.dist.append([])
                self.rvecs.append([])
                self.tvecs.append([])
                continue
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array([self.objpoints[i]], dtype=np.float32),
                                                           np.array([self.imgpoints[i]], dtype=np.float32),
                                                           (self.window_width, self.window_height),
                                                           None, None)
            self.mtx.append(mtx)
            self.dist.append(dist)
            self.rvecs.append(rvecs[0])
            self.tvecs.append(tvecs[0])

        # mtx：内参数矩阵
        # dist：畸变系数
        # rvecs：旋转向量 （外参数）
        # tvecs ：平移向量 （外参数）
        # print(("ret:"), ret)
        # print(("mtx:\n"), mtx)  # 内参数矩阵
        # print(("dist:\n"), dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        # print(("rvecs:\n"), rvecs)  # 旋转向量  # 外参数
        # print(("tvecs:\n"), tvecs)  # 平移向量  # 外参数

    def get_cal_point(self, x, y, force_ptz_preset = None, try_ptz = True):
        using_ptz_preset = force_ptz_preset if force_ptz_preset is not None else self.artnet.ptz_preset
        if len(self.mtx[using_ptz_preset]) == 0:
            return [
                mapfloat(x, 0, self.window_width, 0, 255),
                mapfloat(y, 0, self.window_height, 0, 220)
            ]
        else:
            Lcam = self.mtx[using_ptz_preset].dot(np.hstack((cv2.Rodrigues(self.rvecs[using_ptz_preset])[0], self.tvecs[using_ptz_preset])))
            Z = 0
            X = np.linalg.inv(
                np.hstack((Lcam[:, 0:2], np.array([[-1 * x], [-1 * y], [-1]])))).dot(
                (-Z * Lcam[:, 2] - Lcam[:, 3]))

            if try_ptz:
                if using_ptz_preset >= 2 and X[0] < 10:
                    print("Trying ptz preset ",using_ptz_preset - 1)
                    return self.get_cal_point(x, y, using_ptz_preset - 1)
                elif using_ptz_preset <= 2 and X[0] > 240:
                    print("Trying ptz preset ",using_ptz_preset + 1)
                    return self.get_cal_point(x, y, using_ptz_preset + 1)

                if using_ptz_preset != self.artnet.ptz_preset and self.artnet.pelco_ip is not None:
                    moving_duration = abs(using_ptz_preset - self.artnet.ptz_preset) * 0.6 + 0.2
                    orig_offset = self.get_cal_point(x, y, self.artnet.ptz_preset, False)
                    self.artnet.ptz_preset = using_ptz_preset
                    print("Call ptz preset ", using_ptz_preset, " from ptz ", orig_offset, "to ", X)
                    self.artnet.pelco_fd.send(self.artnet.build_pelco(0x07, 0x00, int(self.artnet.ptz_preset)))
                    return [X[0], X[1], orig_offset[0], orig_offset[1], datetime.datetime.now().timestamp(), moving_duration]

            return [X[0], X[1]]

    def run(self):
        window_width = self.window_width
        window_height = self.window_height

        x = 0
        y = 0
        move_x = None
        move_y = None
        move_ptz_x = None
        move_ptz_y = None
        zoom_in = None

        while not self.terminated_state.get():
            if self.frame is not None:
                self.mutex.acquire()
                if self.frame.shape[1] > 1280:
                    frame = cv2.resize(self.frame, (1280, 720))
                else:
                    frame = self.frame
                self.frame = None
                self.mutex.release()

                if self.win is None:
                    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
                    self.win = sdl2.SDL_CreateWindow(
                        b"Renderer",
                        sdl2.SDL_WINDOWPOS_UNDEFINED,
                        sdl2.SDL_WINDOWPOS_UNDEFINED,
                        window_width,
                        window_height,
                        sdl2.SDL_WINDOW_SHOWN
                    )
                    self.ren = sdl2.SDL_CreateRenderer(
                        self.win,
                        -1,
                        sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
                    )

                    sdl2.SDL_SetRenderDrawBlendMode(self.ren, sdl2.SDL_BLENDMODE_BLEND)
                    rgb_texture = sdl2.SDL_CreateTexture(self.ren, sdl2.SDL_PIXELFORMAT_RGB24,
                                                         sdl2.SDL_TEXTUREACCESS_STREAMING, window_width, window_height)

                output_buffer = np.zeros((window_height, window_width, 4), np.uint8)
                lp_output_buffer = output_buffer.ctypes.data_as(ctypes.c_void_p)

                lp_pixel = ctypes.c_void_p()
                pitch = ctypes.c_int()

                if self.artnet.cal_mode:
                    if move_x is not None:
                        x = x + (5 if move_x else -5)
                    if move_y is not None:
                        y = y + (5 if move_y else -5)
                    if zoom_in is not None:
                        self.artnet.zoom_size += 5 if zoom_in else -5

                    if x < 0:
                        x = 0
                    if x > 255:
                        x = 255
                    if y < 0:
                        y = 0
                    if y > 255:
                        y = 255
                    if self.artnet.zoom_size < 0:
                        self.artnet.zoom_size = 0
                    if self.artnet.zoom_size > 255:
                        self.artnet.zoom_size = 255

                    if move_ptz_x is not None and self.artnet.pelco_ip is not None:
                        if move_ptz_x:
                            self.artnet.pelco_fd.send(self.artnet.build_pelco(0x0002, 0xff, 0xff))
                        else:
                            self.artnet.pelco_fd.send(self.artnet.build_pelco(0x0004, 0xff, 0xff))

                    if move_ptz_y is not None and self.artnet.pelco_ip is not None:
                        if move_ptz_y:
                            self.artnet.pelco_fd.send(self.artnet.build_pelco(0x0010, 0xff, 0xff))
                        else:
                            self.artnet.pelco_fd.send(self.artnet.build_pelco(0x0008, 0xff, 0xff))

                    self.artnet.pos = [
                        [x,y]
                    ]
                    frame = cv2.resize(frame, (window_width, window_height))
                    frame = frame.copy()
                    cv2.putText(frame, 'CAL %3d, %3d, %3d  CAL Points: %d  PTZ: %d' % (x, y, self.artnet.zoom_size, len(self.objpoints[self.artnet.ptz_preset]) if self.artnet.ptz_preset < len(self.objpoints) else 0, self.artnet.ptz_preset),
                                (6, frame.shape[0] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                                1)

                    cv2.putText(frame, 'CAL %3d, %3d, %3d  CAL Points: %d  PTZ: %d' % (x, y, self.artnet.zoom_size, len(self.objpoints[self.artnet.ptz_preset]) if self.artnet.ptz_preset < len(self.objpoints) else 0, self.artnet.ptz_preset),
                                (5, frame.shape[0] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                                1)

                sdl2.SDL_LockTexture(rgb_texture, None, ctypes.byref(lp_pixel), ctypes.byref(pitch))
                src_pixels = np.ctypeslib.as_array(ctypes.cast(lp_pixel, ctypes.POINTER(ctypes.c_ubyte)),
                                                      shape=(window_height, window_width, 3))
                src_pixels[:, :, 0:3] = cv2.resize(frame, (window_width, window_height))

                sdl2.SDL_UnlockTexture(rgb_texture)
                sdl2.SDL_RenderClear(self.ren)
                sdl2.SDL_RenderCopy(self.ren, rgb_texture, None, None)
                sdl2.SDL_RenderReadPixels(self.ren, None, sdl2.SDL_PIXELFORMAT_ARGB8888, lp_output_buffer, frame.shape[1] * 4)

                sdl2.SDL_RenderPresent(self.ren)
            else:
                event = sdl2.SDL_Event()
                in_ctrl = False
                while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
                    if self.onSDL_Event is not None and self.onSDL_Event(event):
                        continue
                    if event.type == sdl2.SDL_KEYUP:
                        # elif event.key.keysym.sym == sdl2.SDLK_v:
                        #     monitor_output.value = 1 + monitor_output.value % (4 + len(cfg["sources"]))
                        if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                            self.artnet.cal_mode = not self.artnet.cal_mode
                            if not self.artnet.cal_mode:
                                json.dump({"window_width": self.window_width, "window_height": self.window_height, "objpoints": self.objpoints, 'imgpoints': self.imgpoints},
                                          open("cam_cal.json", "w"))
                                self.cal()

                        elif event.key.keysym.sym == sdl2.SDLK_LCTRL or event.key.keysym.sym == sdl2.SDLK_RCTRL:
                            in_ctrl = False
                        elif event.key.keysym.sym >= sdl2.SDLK_1 and event.key.keysym.sym <= sdl2.SDLK_9 and event.key.keysym.mod & (sdl2.KMOD_SHIFT):
                            self.artnet.ptz_preset = event.key.keysym.sym - 48
                            if self.artnet.pelco_ip is not None:
                                print("Set Preset ", self.artnet.ptz_preset)
                                self.artnet.pelco_fd.send(self.artnet.build_pelco(0x03, 0x00, int(self.artnet.ptz_preset)))
                            self.cal()
                        elif event.key.keysym.sym >= sdl2.SDLK_1 and event.key.keysym.sym <= sdl2.SDLK_9:
                            self.artnet.ptz_preset = event.key.keysym.sym - 48
                            print("Call Preset ", self.artnet.ptz_preset)
                            if self.artnet.pelco_ip is not None:
                                self.artnet.pelco_fd.send(self.artnet.build_pelco(0x07, 0x00, int(self.artnet.ptz_preset)))
                            self.cal()
                        elif event.key.keysym.sym == sdl2.SDLK_DOWN or event.key.keysym.sym == sdl2.SDLK_UP:
                            move_y = None
                        elif event.key.keysym.sym == sdl2.SDLK_LEFT or event.key.keysym.sym == sdl2.SDLK_RIGHT:
                            move_x = None
                        elif event.key.keysym.sym == sdl2.SDLK_i or event.key.keysym.sym == sdl2.SDLK_k:
                            move_ptz_y = None
                            if self.artnet.pelco_ip is not None:
                                self.artnet.pelco_fd.send(self.artnet.build_pelco(0, 0, 0))
                        elif event.key.keysym.sym == sdl2.SDLK_j or event.key.keysym.sym == sdl2.SDLK_l:
                            move_ptz_x = None
                            if self.artnet.pelco_ip is not None:
                                self.artnet.pelco_fd.send(self.artnet.build_pelco(0, 0, 0))
                        elif event.key.keysym.sym == sdl2.SDLK_z and event.key.keysym.mod & sdl2.KMOD_CTRL:
                            if len(self.objpoints) >= self.artnet.ptz_preset and len(self.objpoints[self.artnet.ptz_preset])>0:
                                self.objpoints[self.artnet.ptz_preset].pop()
                                self.imgpoints[self.artnet.ptz_preset].pop()
                        elif event.key.keysym.sym == sdl2.SDLK_r:
                            if len(self.objpoints) >= self.artnet.ptz_preset:
                                self.objpoints[self.artnet.ptz_preset] = []
                                self.imgpoints[self.artnet.ptz_preset] = []
                        elif event.key.keysym.sym == sdl2.SDLK_EQUALS or event.key.keysym.sym == sdl2.SDLK_MINUS:
                            zoom_in = None
                        else:
                            print("sym key => ", event.key.keysym.sym)
                        # elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                        #     if danmu_transparent.value - 0.05 < 0.05:
                        #         danmu_transparent.value = 0.05
                        #     else:
                        #         danmu_transparent.value = danmu_transparent.value - 0.05
                        # elif event.key.keysym.sym == sdl2.SDLK_UP:
                        #     if danmu_transparent.value + 0.05 > 1:
                        #         danmu_transparent.value = 1
                        #     else:
                        #         danmu_transparent.value = danmu_transparent.value + 0.05
                    elif event.type == sdl2.SDL_KEYDOWN:
                        if event.key.keysym.sym == sdl2.SDLK_LCTRL or event.key.keysym.sym == sdl2.SDLK_RCTRL:
                            in_ctrl = True
                        elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                            move_y = True
                        elif event.key.keysym.sym == sdl2.SDLK_UP:
                            move_y = False
                        elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                            move_x = False
                        elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                            move_x = True
                        elif event.key.keysym.sym == sdl2.SDLK_i:
                            move_ptz_y = False
                        elif event.key.keysym.sym == sdl2.SDLK_k:
                            move_ptz_y = True
                        elif event.key.keysym.sym == sdl2.SDLK_j:
                            move_ptz_x = False
                        elif event.key.keysym.sym == sdl2.SDLK_l:
                            move_ptz_x = True
                        elif event.key.keysym.sym == sdl2.SDLK_EQUALS:
                            zoom_in = True
                        elif event.key.keysym.sym == sdl2.SDLK_MINUS:
                            zoom_in = False
                    elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                        if self.artnet.cal_mode:
                            if self.artnet.ptz_preset not in self.objpoints:
                                while len(self.objpoints) <= self.artnet.ptz_preset:
                                    self.objpoints.append([])
                            self.objpoints[self.artnet.ptz_preset].append([
                                x, y, 0.
                            ])
                            if self.artnet.ptz_preset not in self.imgpoints:
                                while len(self.imgpoints) <= self.artnet.ptz_preset:
                                    self.imgpoints.append([])
                            self.imgpoints[self.artnet.ptz_preset].append([
                                event.button.x, event.button.y
                            ])

                            print([x,y,0.],[event.button.x, event.button.y])
                            # print(event.button.x, event.button.y)
                        else:
                            # print(cv2.projectPoints(np.array([[event.button.x, event.button.y, 1]]).astype(np.float), self.rvecs[0], self.tvecs[0], self.mtx, self.dist))
                            X = self.get_cal_point(event.button.x, event.button.y)
                            self.artnet.mutex.acquire()
                            self.artnet.test_mode = datetime.datetime.now().timestamp() + 5
                            self.artnet.pos = [
                                X
                            ]
                            self.artnet.mutex.release()
                            # print(cv2.undistortPoints(np.array([[event.button.x, event.button.y]]).astype(np.float), self.mtx, self.dist))
                            print("Test Point at %d -> %3d, %3d"  % (self.artnet.ptz_preset, X[0], X[1]))
                            # 点（u, v, 1) 对应代码里的 [605,341,1]
                        break
                    elif event.type == sdl2.SDL_QUIT:
                        break