#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sender.py
--------------------------
Nodo ROS que:
  - Se subscribe a R_AllSensors (sensor_pkg) para leer fuerzas fx, fy, fz
    de los 5 sensores táctiles de la mano RH8D derecha.
  - Construye un vector de 15 floats: [Fx0, Fy0, Fz0, ..., Fx4, Fy4, Fz4].
  - Normaliza (aprox.) a rango [-1, 1] dividiendo por FORCE_MAX.
  - Envía estos 15 floats a un PC Windows por UDP (Tailscale) para
    excitar los guantes TactGlove DK2 en tiempo real.

Ajusta:
  - WINDOWS_IP: IP Tailscale de tu portátil Windows.
  - UDP_PORT: puerto de escucha en Windows (debe coincidir con el script de Windows).
  - FORCE_MAX: escala aproximada de fuerzas para normalizar [-1, 1].
"""

import socket
import struct
import rospy

from sensor_pkg.msg import AllSensors  # tipo del topic R_AllSensors

# ================== CONFIG ==================
WINDOWS_IP   = "100.73.217.87"  # <-- pon aquí la IP Tailscale de tu Windows
UDP_PORT     = 50060              # Debe coincidir con el script de Windows
SENSOR_COUNT = 5                  # nº de sensores (id 0..4)
FORCE_MAX    = 10.0               # N aprox para normalizar [-1,1] (ajusta si hace falta)

TOPIC_SENSORS = "R_AllSensors"

# ============================================

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

class Sender(object):
    def __init__(self):
        self.seq = 0
        self.last_vec = [0.0] * 15

        # Socket UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Subscriber a los sensores
        self.sub = rospy.Subscriber(
            TOPIC_SENSORS,
            AllSensors,
            self.sensor_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Sender inicializado. Enviando a %s:%d",
            WINDOWS_IP, UDP_PORT
        )

    def sensor_callback(self, msg: AllSensors):
        """
        Callback de R_AllSensors.

        El mensaje que has mostrado es algo así:

          length: 5
          data:
            - id: 0
              fx: ...
              fy: ...
              fz: ...
            - id: 1
              fx: ...
              ...

        msg.data es una lista de 5 elementos (id 0..4) y cada uno tiene
        campos fx, fy, fz, abs, yaw, pitch, etc.

        Aquí extraemos:
          [Fx0, Fy0, Fz0, ..., Fx4, Fy4, Fz4]
        y los normalizamos a [-1,1].
        """
        vec = []

        for idx, s in enumerate(msg.data):
            if idx >= SENSOR_COUNT:
                break

            # Tomamos fx, fy, fz tal cual
            fx = float(s.fx)
            fy = float(s.fy)
            fz = float(s.fz)

            vec.extend([fx, fy, fz])

        # Rellena con ceros si por lo que sea hay <5 sensores
        while len(vec) < 15:
            vec.append(0.0)

        # Normaliza [-1, 1] para cada componente
        if FORCE_MAX > 1e-6:
            vec_norm = [clamp(v / FORCE_MAX, -1.0, 1.0) for v in vec]
        else:
            vec_norm = [0.0] * 15

        self.last_vec = vec_norm

        # Empaqueta y envía por UDP (seq + 15 floats, big endian)
        try:
            payload = struct.pack("!I15f", self.seq, *vec_norm)
            self.sock.sendto(payload, (WINDOWS_IP, UDP_PORT))
            self.seq = (self.seq + 1) & 0xFFFFFFFF
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Error enviando UDP táctil: %s", e)

def main():
    rospy.init_node("sender", anonymous=False)
    bridge = Sender()
    rospy.spin()

if __name__ == "__main__":
    main()
