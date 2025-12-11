#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import time

import rospy
from seed_robotics.msg import JointListSetSpeedPos, JointSetSpeedPos

# =================== CONFIG ===================
SIDE = "R"  # mano derecha
TOPIC_CMD = f"{SIDE}_speed_position"

UDP_BIND_IP   = "0.0.0.0"
UDP_BIND_PORT = 5005

# IMPORTANTE: este valor es "teórico".
# La frecuencia REAL será menor, porque se suma el tiempo de procesado + sleep,
# igual que en Fase 1 con mediapipe_to_rh8d.
PUBLISH_HZ = 10.0   # por ejemplo, y frequency=30 en el YAML de Seed

SAFE_TIMEOUT_S   = 0.8        # sin datos durante este tiempo => safe pose
SAFE_POSE_FACTOR = 0.1        # apertura dedos en postura segura

# Deadband por joint: si el cambio es menor que esto, no se modifica el valor
JOINT_DEADBAND = 0.01         # 1% de cambio en apertura

# Cuantización de valores [0..1] a "levels" pasos (solo al convertir a ticks)
QUANTIZATION_LEVELS = 128     # 1/128 ≈ 0.0078

DEFAULT_SPEED         = 300      # 0..1023 (ver README seed_robotics)

# ===== JOINTS / RANGOS (como en tu nodo de Fase 1) =====
JOINTS = {
    # Muñeca
    "w_rotation":  "r_w_rotation",
    "w_adduction": "r_w_adduction",
    "w_flexion":   "r_w_flexion",
    # Pulgar
    "th_adduction": "r_th_adduction",
    "th_flexion":   "r_th_flexion",
    # Dedos
    "ix_flexion":     "r_ix_flexion",
    "middle_flexion": "r_middle_flexion",
    "ring_ltl":       "r_ring_ltl_flexion",
}

JOINT_RANGE = {
    "r_w_rotation":       (800, 3300),
    "r_w_adduction":      (800, 3300),
    "r_w_flexion":        (800, 3300),
    "r_th_adduction":     (600, 3300),
    "r_th_flexion":       (400, 3500),
    "r_ix_flexion":       (400, 4000),
    "r_middle_flexion":   (400, 4000),
    "r_ring_ltl_flexion": (400, 4000),
}

# =================== ESTADO GLOBAL ===================
# Últimos valores 0..1 que REALMENTE se están mandando a la mano
LAST_CMD_VALS = {}           # joint_name -> val01


# =================== UTILIDADES ===================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clamp01_quantize(x, levels=QUANTIZATION_LEVELS):
    """
    Recorta x a [0..1] y lo cuantiza a 'levels' pasos.
    Se usa solo al convertir a ticks para reducir ruido.
    """
    x = clamp01(x)
    return round(x * levels) / float(levels)


def map01_to_ticks(joint_name, val01):
    lo, hi = JOINT_RANGE[joint_name]
    v = clamp01_quantize(val01)
    return int(round(lo + v * (hi - lo)))


def publish_ticks(pub, name_to_val01):
    """
    Publica comandos a la mano SIEMPRE que se llame.
    - name_to_val01: dict joint_name -> valor [0..1]
    - target_speed = -1 en todos los joints (como en tu config actual),
      para que el driver conserve la velocidad previa.
    """
    if not name_to_val01:
        return  # nada que mandar

    msg_list = []
    for jname, v01 in name_to_val01.items():
        ticks = map01_to_ticks(jname, v01)

        m = JointSetSpeedPos()
        m.name = jname
        m.target_pos = ticks

        # En esta fase estás usando siempre -1 (solo posición)
        m.target_speed = DEFAULT_SPEED

        msg_list.append(m)

    out = JointListSetSpeedPos()
    out.joints = msg_list
    pub.publish(out)


# =================== MAIN ===================
def main():
    global LAST_CMD_VALS

    rospy.init_node("udp_bridge", anonymous=False)
    pub = rospy.Publisher(TOPIC_CMD, JointListSetSpeedPos, queue_size=1)

    # Socket UDP no bloqueante
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_BIND_IP, UDP_BIND_PORT))
    sock.setblocking(False)

    rospy.loginfo(f"[udp_bridge] Escuchando UDP en {UDP_BIND_IP}:{UDP_BIND_PORT}")

    last_state = None        # dict joint_name -> val01 (último deseado recibido)
    last_recv_time = None    # timestamp del último paquete válido
    safe_pose_active = False # True si estamos en postura segura

    rate_dt = 1.0 / PUBLISH_HZ

    while not rospy.is_shutdown():
        # 1) Leer todos los paquetes disponibles, quedarnos con el último
        while True:
            try:
                data, addr = sock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception as e:
                rospy.logwarn(f"[udp_bridge] Error leyendo UDP: {e}")
                break

            # Formato esperado: seq (uint32) + 9 floats (little endian)
            if len(data) != (4 + 9 * 4):
                rospy.logwarn_throttle(
                    1.0,
                    f"[udp_bridge] Paquete con tamaño incorrecto: {len(data)} bytes"
                )
                continue

            try:
                unpacked = struct.unpack("<I9f", data)
            except struct.error as e:
                rospy.logwarn_throttle(
                    1.0,
                    f"[udp_bridge] Error unpack struct: {e}"
                )
                continue

            seq = unpacked[0]
            thumb, index, middle, ring, pinky, w_roll, w_add, w_flex, th_add = unpacked[1:]

            # RH8D tiene un actuador para anular+meñique => media
            ring_pinky = 0.5 * (ring + pinky)

            # Mapear a nombres de joint de Seed
            name_to_val01 = {
                # Muñeca
                JOINTS["w_rotation"]:   w_roll,
                JOINTS["w_adduction"]:  w_add,
                JOINTS["w_flexion"]:    w_flex,
                # Pulgar
                JOINTS["th_adduction"]: th_add,
                JOINTS["th_flexion"]:   thumb,
                # Dedos
                JOINTS["ix_flexion"]:     index,
                JOINTS["middle_flexion"]: middle,
                JOINTS["ring_ltl"]:       ring_pinky,
            }

            last_state = name_to_val01
            last_recv_time = time.time()
            safe_pose_active = False  # hemos recuperado comunicación

        now = time.time()

        # Si todavía no hemos recibido ni un solo paquete: no publicar nada
        if last_state is None or last_recv_time is None:
            time.sleep(max(0.0, rate_dt))
            continue

        # 2) Si llevamos demasiado tiempo sin datos: mandar postura segura
        if (now - last_recv_time) > SAFE_TIMEOUT_S:
            if not safe_pose_active:
                rospy.logwarn("[udp_bridge] Timeout de datos. Mandando postura segura.")

                safe_vals = {
                    JOINTS["w_rotation"]:     0.5,
                    JOINTS["w_adduction"]:    0.5,
                    JOINTS["w_flexion"]:      0.5,
                    JOINTS["th_adduction"]:   0.5,
                    JOINTS["th_flexion"]:     SAFE_POSE_FACTOR,
                    JOINTS["ix_flexion"]:     SAFE_POSE_FACTOR,
                    JOINTS["middle_flexion"]: SAFE_POSE_FACTOR,
                    JOINTS["ring_ltl"]:       SAFE_POSE_FACTOR,
                }

                # Actualizar LAST_CMD_VALS con la safe pose
                LAST_CMD_VALS = {jn: float(v) for jn, v in safe_vals.items()}
                safe_pose_active = True

            # Publicar safe pose en este ciclo
            publish_ticks(pub, LAST_CMD_VALS)
            time.sleep(max(0.0, rate_dt))
            continue

        # 3) Tenemos datos recientes -> actualizar comando deseado con deadband
        if not LAST_CMD_VALS:
            # Primera recepción: copiar directamente los valores recibidos
            LAST_CMD_VALS = {jn: float(v) for jn, v in last_state.items()}
        else:
            for jname, desired in last_state.items():
                desired = clamp01(desired)
                last = LAST_CMD_VALS.get(jname, desired)

                if abs(desired - last) > JOINT_DEADBAND:
                    # Cambio significativo -> actualizar comando
                    LAST_CMD_VALS[jname] = desired
                else:
                    # Cambio pequeño -> mantener el valor anterior
                    LAST_CMD_VALS[jname] = last

        # 4) Publicar SIEMPRE los valores actuales (periodicidad marcada por time.sleep)
        publish_ticks(pub, LAST_CMD_VALS)

        # === CONTROL DE "FRECUENCIA" ESTILO FASE 1 ===
        # Igual que en mediapipe_to_rh8d: no compensamos tiempo de procesado,
        # simplemente dormimos rate_dt. La frecuencia real será algo menor.
        time.sleep(max(0.0, rate_dt))


if __name__ == "__main__":
    main()
