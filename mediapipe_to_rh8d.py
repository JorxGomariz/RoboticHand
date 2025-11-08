#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mediapipe_to_rh8d.py
--------------------
Nodo ROS que:
  - Abre la webcam (una sola cámara)
  - Detecta 21 landmarks de mano con MediaPipe
  - Calcula apertura [0..1] para pulgar, índice, medio, anular, meñique
  - Estima muñeca: rotación (pronación/supinación), aducción (desv. radial/cubital), flexión (palmar/dorsal)
  - Calcula aducción del pulgar
  - Publica comandos a la RH8D vía seed_robotics: JointListSetSpeedPos en R_speed_position/L_speed_position

NOTAS:
- Usa nombres EXACTOS de joints tal como aparecen en tu YAML (lowercase).
- Incluye auto-calibración (2 s) para dedos y muñeca (ángulos min/max observados).
- Safe pose si se pierde tracking > SAFE_TIMEOUT_S.
"""

import os
import time
import math
import cv2
import numpy as np
import rospy

from seed_robotics.msg import JointListSetSpeedPos, JointSetSpeedPos

# =================== CONFIGURACIÓN GENERAL ===================
SIDE                  = "R"      # "R" o "L": define el topic de comandos
TOPIC_CMD             = f"{SIDE}_speed_position"
CAM_INDEX             = 0
TARGET_FPS            = 30.0
SHOW_PREVIEW          = True
EMA_ALPHA             = 0.25     # suavizado de landmarks [0..1]
CALIB_WARMUP_S        = 2.0      # segundos de auto-calibración inicial
GAMMA_NONLINEAR_FING  = 1.2      # curva n^gamma para dedos (1.0 = lineal)
GAMMA_NONLINEAR_WRIST = 1.0
DEFAULT_SPEED         = 300      # 0..1023 (ver README seed_robotics)
SAFE_TIMEOUT_S        = 0.5      # si perdemos tracking este tiempo, publicar postura segura
SAFE_POSE_FACTOR      = 0.1      # % de apertura para dedos en safe
HUE_OFFSET           = 30
# ===== Mapeo nombres de joints (desde tu YAML) =====
# Según el YAML facilitado:
# joint_mapping: {
#  r_main_board: 30,
#  r_w_rotation: 31,
#  r_w_adduction: 32,
#  r_w_flexion: 33,
#  r_th_adduction: 34,
#  r_th_flexion: 35,
#  r_ix_flexion: 36,
#  r_middle_flexion: 37,
#  r_ring_ltl_flexion: 38,
# }
JOINTS = {
    # Muñeca
    "w_rotation":     "r_w_rotation",
    "w_adduction":    "r_w_adduction",
    "w_flexion":      "r_w_flexion",
    # Pulgar
    "th_adduction":   "r_th_adduction",
    "th_flexion":     "r_th_flexion",
    # Dedos
    "ix_flexion":     "r_ix_flexion",
    "middle_flexion": "r_middle_flexion",
    "ring_ltl":       "r_ring_ltl_flexion",   # anular + meñique acoplados
}

# RANGOS de ticks conservadores (ajústalos a tu mecánica/telemetría)
# Se mapean valores [0..1] a [lo..hi].
JOINT_RANGE = {
    # Muñeca (ejemplos conservadores)
    "r_w_rotation":     (800, 3300),   # 1200, 3000
    "r_w_adduction":    (800, 3300),   # 1200, 3000
    "r_w_flexion":      (800, 3300),   # 1200, 3000
    # Pulgar
    "r_th_adduction":   (600, 3300),   # 1200, 3000
    "r_th_flexion":     (400,  3500),   # 800, 3300
    # Dedos
    "r_ix_flexion":     (400,  4000),   # 800, 3500
    "r_middle_flexion": (400,  4000),   # 800, 3500
    "r_ring_ltl_flexion": (400, 4000),  # 800, 3200
}

# Asociación "dedo lógico" -> joint de flexión (para el cálculo de ticks)
FINGER_TO_JOINT = {
    "thumb":  JOINTS["th_flexion"],
    "index":  JOINTS["ix_flexion"],
    "middle": JOINTS["middle_flexion"],
    "ring":   JOINTS["ring_ltl"],
    "pinky":  JOINTS["ring_ltl"],  # compartido -> luego promediamos
}

# =================== MediaPipe ===================
import mediapipe as mp
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
        for k in list(self.state.keys()):
            if k not in d:
                del self.state[k]
        return out

smoother = EMASmoother()

# =================== Utilidades geométricas ===================
def angle_between(u, v):
    u = np.array(u, dtype=np.float32); v = np.array(v, dtype=np.float32)
    nu = np.linalg.norm(u) + 1e-9; nv = np.linalg.norm(v) + 1e-9
    u = u/nu; v = v/nv
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def build_hand_frame(lm):
    """
    Construye una base ortonormal de la mano:
      x_h: Wrist->Index_MCP (radial)
      y_h: componente ortogonal de Wrist->Pinky_MCP
      z_h: x_h × y_h (normal de la palma)
    Coordenadas están en el "espacio cámara" (x derecha, y abajo, z relativa de MediaPipe).
    """
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
    return xh, yh, zh  # ejes mano en coords cámara

def euler_like_from_frame(xh, yh, zh):
    """
    Devuelve (roll, adduction, flexion) en grados a partir de la base de la mano.
    No son Tait-Bryan puros; están elegidos para controlar muñeca de forma intuitiva:

      - roll        ~ pronación/supinación alrededor del eje del antebrazo.
                     Aproximamos como giro alrededor del eje x_h (ángulo entre y_h y el plano xz).
      - adduction   ~ desviación radial/cubital (lateral). Aproximamos con yaw en el plano imagen:
                     atan2(xh_y, xh_x).
      - flexion     ~ palmar/dorsal. Aproximamos inclinación hacia cámara:
                     atan2(-xh_z, sqrt(xh_x^2 + xh_y^2)).

    Estas aproximaciones funcionan bien con 1 cámara y landmarks 3D relativos de MediaPipe.
    """
    # aducción (lateral, en imagen)
    adduction = math.degrees(math.atan2(xh[1], xh[0]))
    # flexión (inclinación hacia/desde cámara)
    flexion = math.degrees(math.atan2(-xh[2], math.sqrt(xh[0]**2 + xh[1]**2)))
    # roll (pron/sup) – giro de la palma en torno al eje que apunta hacia índice
    # medimos cuánto "sale" y_h del plano de la cámara frente al normal de la palma
    roll = math.degrees(math.atan2(yh[2], zh[2]))
    return roll, adduction, flexion

# =================== Cálculo de aperturas dedo ===================
def finger_flex_metric(lm):
    """
    Métrica de "flexión" (grados) por dedo: mayor => más cerrado.
    Usa PIP y DIP (y MCP/IP en pulgar).
    """
    # Índices MediaPipe
    TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
    IN_MCP, IN_PIP, IN_DIP, IN_TIP = 5, 6, 7, 8
    MI_MCP, MI_PIP, MI_DIP, MI_TIP = 9,10,11,12
    RI_MCP, RI_PIP, RI_DIP, RI_TIP = 13,14,15,16
    PI_MCP, PI_PIP, PI_DIP, PI_TIP = 17,18,19,20

    def ang(a,b,c):
        va = np.array(lm[a]) - np.array(lm[b])
        vc = np.array(lm[c]) - np.array(lm[b])
        return angle_between(va, vc)

    flex_index  = ang(IN_MCP, IN_PIP, IN_DIP) + ang(IN_PIP, IN_DIP, IN_TIP)
    flex_middle = ang(MI_MCP, MI_PIP, MI_DIP) + ang(MI_PIP, MI_DIP, MI_TIP)
    flex_ring   = ang(RI_MCP, RI_PIP, RI_DIP) + ang(RI_PIP, RI_DIP, RI_TIP)
    flex_pinky  = ang(PI_MCP, PI_PIP, PI_DIP) + ang(PI_PIP, PI_DIP, PI_TIP)
    # Pulgar: MCP + IP
    flex_thumb  = ang(TH_CMC, TH_MCP, TH_IP) + ang(TH_MCP, TH_IP, TH_TIP)

    return {
        "thumb":  flex_thumb,
        "index":  flex_index,
        "middle": flex_middle,
        "ring":   flex_ring,
        "pinky":  flex_pinky,
    }

def thumb_adduction_metric(lm):
    """
    Métrica lateral del pulgar en el plano de la palma:
      Proyecta dirección pulgar (MCP->TIP) sobre el plano (x_h, y_h)
      y toma componente lateral ~ aducción.
    """
    TH_MCP, TH_TIP = 2, 4
    xh, yh, zh = build_hand_frame(lm)
    v = np.array(lm[TH_TIP]) - np.array(lm[TH_MCP])
    # component in hand plane
    v_plane = v - np.dot(v, zh) * zh
    # aducción ~ componente sobre +y_h (ulnar)
    sign = np.sign(np.dot(v_plane, yh))
    mag  = np.linalg.norm(v_plane) / (np.linalg.norm(v) + 1e-9)
    # devolver ángulo "lateral" en grados a partir de la proyección
    angle = sign * math.degrees(math.asin(np.clip(mag, -1.0, 1.0)))
    return angle

# =================== Auto-calibrador ===================
class AutoCalibrator:
    """
    Guarda min/max observados para:
      - flexión dedos (grados)
      - ángulos muñeca (roll, adduction, flexion) en grados
      - aducción pulgar (grados)
    Durante CALIB_WARMUP_S se actualizan límites; luego se fijan.
    """
    def __init__(self):
        self.t0 = time.time()
        self.ready = False
        self.min_flex = {k:  1e9 for k in ["thumb","index","middle","ring","pinky"]}
        self.max_flex = {k: -1e9 for k in ["thumb","index","middle","ring","pinky"]}
        self.min_wrist = {k:  1e9 for k in ["roll","adduction","flexion","th_add"]}
        self.max_wrist = {k: -1e9 for k in ["roll","adduction","flexion","th_add"]}

    def update(self, flex_dict, roll, add, flex, th_add):
        now = time.time()
        for k, v in flex_dict.items():
            self.min_flex[k] = min(self.min_flex[k], v)
            self.max_flex[k] = max(self.max_flex[k], v)
        self.min_wrist["roll"] = min(self.min_wrist["roll"], roll)
        self.max_wrist["roll"] = max(self.max_wrist["roll"], roll)
        self.min_wrist["adduction"] = min(self.min_wrist["adduction"], add)
        self.max_wrist["adduction"] = max(self.max_wrist["adduction"], add)
        self.min_wrist["flexion"] = min(self.min_wrist["flexion"], flex)
        self.max_wrist["flexion"] = max(self.max_wrist["flexion"], flex)
        self.min_wrist["th_add"] = min(self.min_wrist["th_add"], th_add)
        self.max_wrist["th_add"] = max(self.max_wrist["th_add"], th_add)
        if (now - self.t0) >= CALIB_WARMUP_S:
            self.ready = True

    def norm01_fingers(self, flex_dict):
        """
        Convierte "flexión (grados)" -> apertura [0..1] por dedo.
        Más flexión (cerrado, valor alto) -> 0
        Menos flexión (abierto, valor bajo) -> 1
        """
        out = {}
        for k, v in flex_dict.items():
            lo, hi = self.min_flex[k], self.max_flex[k]
            if hi - lo < 1e-3:
                p = 0.0
            else:
                p = (hi - v) / (hi - lo)
            p = float(np.clip(p, 0.0, 1.0))
            if GAMMA_NONLINEAR_FING != 1.0:
                p = p ** GAMMA_NONLINEAR_FING
            out[k] = p
        return out

    def norm01_wrist(self, roll, add, flex, th_add):
        """
        Normaliza muñeca y aducción pulgar a [0..1] con min/max observados.
        """
        def n01(x, lo, hi):
            if hi - lo < 1e-3: return 0.5
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
        w = {}
        w["roll"]      = n01(roll,    self.min_wrist["roll"],      self.max_wrist["roll"])
        w["adduction"] = n01(add,     self.min_wrist["adduction"], self.max_wrist["adduction"])
        w["flexion"]   = n01(flex,    self.min_wrist["flexion"],   self.max_wrist["flexion"])
        w["th_add"]    = n01(th_add,  self.min_wrist["th_add"],    self.max_wrist["th_add"])
        if GAMMA_NONLINEAR_WRIST != 1.0:
            for k in w:
                w[k] = w[k] ** GAMMA_NONLINEAR_WRIST
        return w

calib = AutoCalibrator()

# =================== Utilidades ROS ===================
def map01_to_ticks(joint_name, val01):
    lo, hi = JOINT_RANGE[joint_name]
    val01 = float(np.clip(val01, 0.0, 1.0))
    return int(round(lo + val01 * (hi - lo)))

def publish_ticks(pub, name_to_val01):
    lst = []
    for jname, v01 in name_to_val01.items():
        ticks = map01_to_ticks(jname, v01)
        msg = JointSetSpeedPos()
        msg.name = jname
        msg.target_pos = ticks
        msg.target_speed = DEFAULT_SPEED
        lst.append(msg)
    out = JointListSetSpeedPos()
    out.joints = lst
    pub.publish(out)

# =================== MAIN ===================
def main():
    rospy.init_node("mediapipe_to_rh8d", anonymous=False)
    pub = rospy.Publisher(TOPIC_CMD, JointListSetSpeedPos, queue_size=1)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    last_seen = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        rate_dt = 1.0 / float(TARGET_FPS)
        while not rospy.is_shutdown():
            ok, frame = cap.read()
            if not ok:
                rospy.logwarn("Frame no válido")
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # espejo
            #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ## Glove config
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]

            # Desplaza el tono (cv2.add evita overflow)
            h = cv2.add(h, HUE_OFFSET)
            hsv_shifted = cv2.merge([h, s, v])
            rgb = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)

            ##
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            sent_safe = False
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = {i: (hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z) for i in range(21)}
                lm = smoother(lm)

                # Dedos
                flex_dict = finger_flex_metric(lm)

                # Muñeca y aducción pulgar
                xh, yh, zh = build_hand_frame(lm)
                roll, add, flex = euler_like_from_frame(xh, yh, zh)
                th_add = thumb_adduction_metric(lm)

                # Actualiza calibración
                calib.update(flex_dict, roll, add, flex, th_add)

                # Si calibrado, publica
                if calib.ready:
                    last_seen = time.time()

                    # 1) Apertura dedos [0..1]
                    open01 = calib.norm01_fingers(flex_dict)

                    # 2) Resolver joint compartido (ring + little)
                    ring_pinky = (open01["ring"] + open01["pinky"]) * 0.5

                    # 3) Muñeca / Aducción pulgar normalizadas [0..1]
                    w = calib.norm01_wrist(roll, add, flex, th_add)

                    # 4) Construye diccionario joint->val01
                    name_to_val01 = {
                        # Muñeca
                        JOINTS["w_rotation"]:     w["roll"],
                        JOINTS["w_adduction"]:    w["adduction"],
                        JOINTS["w_flexion"]:      w["flexion"],
                        # Pulgar
                        JOINTS["th_adduction"]:   w["th_add"],
                        JOINTS["th_flexion"]:     open01["thumb"],
                        # Dedos
                        JOINTS["ix_flexion"]:     open01["index"],
                        JOINTS["middle_flexion"]: open01["middle"],
                        JOINTS["ring_ltl"]:       ring_pinky,
                    }
                    publish_ticks(pub, name_to_val01)

                # Dibujos
                if SHOW_PREVIEW:
                    h, w_img = frame.shape[:2]
                    for i, p in lm.items():
                        cv2.circle(frame, (int(p[0]*w_img), int(p[1]*h)), 3, (0,255,255), -1)
                    status = "Calibrado" if calib.ready else "Calibrando... abre/cierra y mueve muñeca"
                    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

            else:
                smoother.reset()
                # Si perdemos tracking mucho tiempo, mandar postura segura
                if (time.time() - last_seen) > SAFE_TIMEOUT_S and calib.ready:
                    name_to_val01 = {
                        # Muñeca a centro (~0.5)
                        JOINTS["w_rotation"]:   0.5,
                        JOINTS["w_adduction"]:  0.5,
                        JOINTS["w_flexion"]:    0.5,
                        # Pulgar y dedos semi-abiertos
                        JOINTS["th_adduction"]: 0.5,
                        JOINTS["th_flexion"]:   SAFE_POSE_FACTOR,
                        JOINTS["ix_flexion"]:   SAFE_POSE_FACTOR,
                        JOINTS["middle_flexion"]: SAFE_POSE_FACTOR,
                        JOINTS["ring_ltl"]:     SAFE_POSE_FACTOR,
                    }
                    publish_ticks(pub, name_to_val01)
                    sent_safe = True

            if SHOW_PREVIEW:
                if not res.multi_hand_landmarks:
                    txt = "Sin mano" if calib.ready else "Esperando mano / Calibrando"
                    cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,160,160), 2, cv2.LINE_AA)
                cv2.imshow("mediapipe_to_rh8d", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break

            # Mantener ritmo objetivo
            time.sleep(max(0.0, rate_dt))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
