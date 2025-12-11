#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import socket
import struct
import time
import numpy as np
import mediapipe as mp

# =================== CONFIG GENERAL ===================
CAM_INDEX = 0
TARGET_FPS = 30.0
SHOW_PREVIEW = True

EMA_ALPHA = 0.25
CALIB_WARMUP_S = 2.0
GAMMA_NONLINEAR_FING = 1.2
GAMMA_NONLINEAR_WRIST = 1.0
HUE_OFFSET = 30

# UDP destino (Ubuntu / Tailscale)
UBUNTU_TAILSCALE_IP = "100.119.213.117"   # <-- PON AQUÍ LA IP TAILSCALE DE UBUNTU
UBUNTU_UDP_PORT     = 5005

NET_SEND_FPS = 30.0   # frecuencia máxima de envío por red

# =================== MediaPipe ===================
mp_hands = mp.solutions.hands

# =================== Suavizado EMA ===================
class EMASmoother:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.state = {}

    def reset(self):
        self.state.clear()

    def __call__(self, d):
        out = {}
        for k, v in d.items():
            v = np.array(v, dtype=np.float32)
            if k not in self.state:
                self.state[k] = v.copy()
            else:
                self.state[k] = self.alpha * v + (1.0 - self.alpha) * self.state[k]
            out[k] = self.state[k]
        # limpia claves que ya no existan
        for k in list(self.state.keys()):
            if k not in d:
                del self.state[k]
        return out

smoother = EMASmoother()

# =================== Utilidades geométricas ===================
def angle_between(u, v):
    u = np.array(u, dtype=np.float32)
    v = np.array(v, dtype=np.float32)
    nu = np.linalg.norm(u) + 1e-9
    nv = np.linalg.norm(v) + 1e-9
    u = u / nu
    v = v / nv
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def build_hand_frame(lm):
    WRIST = 0; IN_MCP = 5; PI_MCP = 17
    pw = np.array(lm[WRIST], dtype=np.float32)
    pi = np.array(lm[IN_MCP], dtype=np.float32)
    pp = np.array(lm[PI_MCP], dtype=np.float32)

    xh = pi - pw
    xh = xh / (np.linalg.norm(xh) + 1e-9)
    yp_raw = pp - pw
    zp = np.cross(xh, yp_raw)
    zh = zp / (np.linalg.norm(zp) + 1e-9)
    yh = np.cross(zh, xh)
    yh = yh / (np.linalg.norm(yh) + 1e-9)
    return xh, yh, zh

def euler_like_from_frame(xh, yh, zh):
    adduction = math.degrees(math.atan2(xh[1], xh[0]))
    flexion = math.degrees(math.atan2(-xh[2], math.sqrt(xh[0] ** 2 + xh[1] ** 2)))
    roll = math.degrees(math.atan2(yh[2], zh[2]))
    return roll, adduction, flexion

def finger_flex_metric(lm):
    TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
    IN_MCP, IN_PIP, IN_DIP, IN_TIP = 5, 6, 7, 8
    MI_MCP, MI_PIP, MI_DIP, MI_TIP = 9,10,11,12
    RI_MCP, RI_PIP, RI_DIP, RI_TIP = 13,14,15,16
    PI_MCP, PI_PIP, PI_DIP, PI_TIP = 17,18,19,20

    def ang(a, b, c):
        va = np.array(lm[a]) - np.array(lm[b])
        vc = np.array(lm[c]) - np.array(lm[b])
        return angle_between(va, vc)

    flex_index  = ang(IN_MCP, IN_PIP, IN_DIP) + ang(IN_PIP, IN_DIP, IN_TIP)
    flex_middle = ang(MI_MCP, MI_PIP, MI_DIP) + ang(MI_PIP, MI_DIP, MI_TIP)
    flex_ring   = ang(RI_MCP, RI_PIP, RI_DIP) + ang(RI_PIP, RI_DIP, RI_TIP)
    flex_pinky  = ang(PI_MCP, PI_PIP, PI_DIP) + ang(PI_PIP, PI_DIP, PI_TIP)
    flex_thumb  = ang(TH_CMC, TH_MCP, TH_IP) + ang(TH_MCP, TH_IP, TH_TIP)

    return {
        "thumb":  flex_thumb,
        "index":  flex_index,
        "middle": flex_middle,
        "ring":   flex_ring,
        "pinky":  flex_pinky,
    }

def thumb_adduction_metric(lm):
    TH_MCP, TH_TIP = 2, 4
    xh, yh, zh = build_hand_frame(lm)
    v = np.array(lm[TH_TIP]) - np.array(lm[TH_MCP])
    v_plane = v - np.dot(v, zh) * zh
    sign = np.sign(np.dot(v_plane, yh))
    mag = np.linalg.norm(v_plane) / (np.linalg.norm(v) + 1e-9)
    angle = sign * math.degrees(math.asin(np.clip(mag, -1.0, 1.0)))
    return angle

# =================== Auto-calibrador ===================
class AutoCalibrator:
    def __init__(self):
        self.t0 = time.time()
        self.ready = False
        self.min_flex = {k: 1e9 for k in ["thumb","index","middle","ring","pinky"]}
        self.max_flex = {k: -1e9 for k in ["thumb","index","middle","ring","pinky"]}
        self.min_wrist = {k: 1e9 for k in ["roll","adduction","flexion","th_add"]}
        self.max_wrist = {k: -1e9 for k in ["roll","adduction","flexion","th_add"]}

    def update(self, flex_dict, roll, add, flex, th_add):
        now = time.time()
        for k, v in flex_dict.items():
            self.min_flex[k] = min(self.min_flex[k], v)
            self.max_flex[k] = max(self.max_flex[k], v)

        self.min_wrist["roll"]      = min(self.min_wrist["roll"],      roll)
        self.max_wrist["roll"]      = max(self.max_wrist["roll"],      roll)
        self.min_wrist["adduction"] = min(self.min_wrist["adduction"], add)
        self.max_wrist["adduction"] = max(self.max_wrist["adduction"], add)
        self.min_wrist["flexion"]   = min(self.min_wrist["flexion"],   flex)
        self.max_wrist["flexion"]   = max(self.max_wrist["flexion"],   flex)
        self.min_wrist["th_add"]    = min(self.min_wrist["th_add"],    th_add)
        self.max_wrist["th_add"]    = max(self.max_wrist["th_add"],    th_add)

        if (now - self.t0) >= CALIB_WARMUP_S:
            self.ready = True

    def norm01_fingers(self, flex_dict):
        out = {}
        for k, v in flex_dict.items():
            lo, hi = self.min_flex[k], self.max_flex[k]
            if hi - lo < 1e-3:
                p = 0.0
            else:
                p = (hi - v) / (hi - lo)  # más flexión = más cerrado = 0
            p = float(np.clip(p, 0.0, 1.0))
            if GAMMA_NONLINEAR_FING != 1.0:
                p = p ** GAMMA_NONLINEAR_FING
            out[k] = p
        return out

    def norm01_wrist(self, roll, add, flex, th_add):
        def n01(x, lo, hi):
            if hi - lo < 1e-3:
                return 0.5
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        w = {}
        w["roll"]      = n01(roll,   self.min_wrist["roll"],      self.max_wrist["roll"])
        w["adduction"] = n01(add,    self.min_wrist["adduction"], self.max_wrist["adduction"])
        w["flexion"]   = n01(flex,   self.min_wrist["flexion"],   self.max_wrist["flexion"])
        w["th_add"]    = n01(th_add, self.min_wrist["th_add"],    self.max_wrist["th_add"])

        if GAMMA_NONLINEAR_WRIST != 1.0:
            for k in w:
                w[k] = w[k] ** GAMMA_NONLINEAR_WRIST
        return w

calib = AutoCalibrator()

# =================== UDP CLIENT ===================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = (UBUNTU_TAILSCALE_IP, UBUNTU_UDP_PORT)

def send_state_udp(seq, open01, wrist_norm):
    thumb  = float(open01["thumb"])
    index  = float(open01["index"])
    middle = float(open01["middle"])
    ring   = float(open01["ring"])
    pinky  = float(open01["pinky"])

    w_roll = float(wrist_norm["roll"])
    w_add  = float(wrist_norm["adduction"])
    w_flex = float(wrist_norm["flexion"])
    th_add = float(wrist_norm["th_add"])

    packet = struct.pack("<I9f", int(seq),
                         thumb, index, middle, ring, pinky,
                         w_roll, w_add, w_flex, th_add)
    sock.sendto(packet, server_addr)

# =================== MAIN LOOP ===================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    last_send_time = 0.0
    seq = 0

    print(f"Enviando datos a {server_addr} (UDP, binario)")

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        dt_frame = 1.0 / TARGET_FPS
        dt_net   = 1.0 / NET_SEND_FPS

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # === HUE OFFSET para guante ===
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            h = cv2.add(h, HUE_OFFSET)
            hsv_shifted = cv2.merge([h, s, v])
            rgb = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)

            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            text_status = ""

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = {i: (hand.landmark[i].x,
                          hand.landmark[i].y,
                          hand.landmark[i].z) for i in range(21)}
                lm = smoother(lm)

                flex_dict = finger_flex_metric(lm)
                xh, yh, zh = build_hand_frame(lm)
                roll, add, flx = euler_like_from_frame(xh, yh, zh)
                th_add = thumb_adduction_metric(lm)

                calib.update(flex_dict, roll, add, flx, th_add)

                if calib.ready:
                    open01 = calib.norm01_fingers(flex_dict)
                    wrist_norm = calib.norm01_wrist(roll, add, flx, th_add)

                    now = time.time()
                    if (now - last_send_time) >= dt_net:
                        seq += 1
                        send_state_udp(seq, open01, wrist_norm)
                        last_send_time = now

                    text_status = "Calibrado - enviando"
                else:
                    text_status = "Calibrando... abre/cierra dedos y mueve muñeca"

                if SHOW_PREVIEW:
                    h_img, w_img = frame.shape[:2]
                    for i, p in lm.items():
                        cv2.circle(frame,
                                   (int(p[0] * w_img), int(p[1] * h_img)),
                                   3, (0, 255, 255), -1)
            else:
                smoother.reset()
                text_status = "Sin mano (o calibrando)"

            if SHOW_PREVIEW:
                cv2.putText(frame, text_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow("mediapipe_udp", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break

            time.sleep(max(0.0, dt_frame))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
