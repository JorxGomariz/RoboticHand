# -*- coding: utf-8 -*-
"""
tactglove.py
--------------------------
Recibe por UDP (desde Ubuntu) las fuerzas táctiles de la mano RH8D:
  Paquete: uint32 seq + 15 floats normalizados [-1, 1]
           [Fx1, Fy1, Fz1, ..., Fx5, Fy5, Fz5]

Las convierte en intensidades (0..1) por dedo y las manda al guante
TactGlove DK2 derecho en tiempo real usando bhaptics-python.

Comportamiento:
  - Si NO llegan datos (> TIMEOUT_S): stop_all() y no vibra.
  - Para cada dedo:
      * Calcula magnitud cruda de fuerza (mag_raw).
      * Aprende un baseline de ruido mag_base cuando NO hay contacto.
      * Calcula mag_corr = max(0, mag_raw - (mag_base + NOISE_MARGIN)).
      * Filtra mag_corr con EMA.
      * Usa histéresis:
          - entra en contacto si ema_mag > ON_THRESHOLD
          - sale de contacto si ema_mag < OFF_THRESHOLD
      * Solo vibra si ese dedo está en contacto.
  - Así, si el ruido sube con el tiempo, el baseline se adapta y desaparece
    la vibración “fantasma” sin contacto real.
"""

import asyncio
import math
import socket
import struct
import time
import sys

import bhaptics_python as bh

# =========================
# FIX PARA WINDOWS (asyncio)
# =========================
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        print("[WARN] No se pudo establecer WindowsSelectorEventLoopPolicy:", e)

# =========================
# ======== CONFIG =========
# =========================
APP_ID  = "69122dc2f088f93372a8f3dc"
API_KEY = "a87vKdw8Q4aOTIXmMiLb"

RIGHT_GLOVE_POS = 9          # 9 = Right glove en SDK DK2
UDP_PORT        = 50060      # Debe coincidir con el script de Ubuntu

HZ              = 40         # 25–50 Hz es razonable
DT              = 1.0 / HZ
WINDOW_PLAYTIME = 4          # 4 -> ~20 ms
DEFAULT_SHAPE   = 0          # 0=const

# Detección de contacto (histeresis) SOBRE magnitud corregida
ON_THRESHOLD   = 0.7        # umbral para entrar en contacto
OFF_THRESHOLD  = 0.5        # umbral para salir de contacto
TIMEOUT_S      = 0.5         # sin datos recientes => apagar todo

# Filtro y baseline
EMA_ALPHA      = 0.3         # suavizado de magnitud corregida por dedo
NOISE_ALPHA    = 0.01        # qué rápido se adapta el baseline (ruido)
NOISE_MARGIN   = 0.03        # margen extra por encima del baseline para considerar algo como fuerza real

MOTOR_NAMES = ["Pulgar", "Índice", "Medio", "Anular", "Meñique", "Muñeca"]

# =========================
# ====== ESTADO GLOBAL ====
# =========================

latest_vec  = [0.0] * 15   # últimos 15 floats recibidos
last_update = 0.0          # timestamp del último paquete válido

# Estado por dedo (0..4)
ema_mag       = [0.0] * 5   # magnitud filtrada (EMA) de la fuerza corregida
baseline_noise = [0.0] * 5  # baseline de ruido (magnitud cruda) cuando NO hay contacto
in_contact    = [False] * 5 # True si el dedo está "en contacto"
was_active    = False       # si en el último ciclo vibraba algún dedo

# =========================
# ====== UTILIDADES =======
# =========================

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def map01_to_100(x: float) -> int:
    return int(round(100 * clamp01(x)))

async def play_frame_glove(motors01, shape=DEFAULT_SHAPE, playtime=WINDOW_PLAYTIME):
    """
    motors01: lista de 6 floats [0..1] (5 dedos + muñeca).
    """
    motors    = [map01_to_100(v) for v in motors01]
    playtimes = [playtime] * 6
    shapes    = [shape] * 6
    await bh.play_glove(RIGHT_GLOVE_POS, motors, playtimes, shapes, 0)

# =========================
# ===== UDP LISTENER ======
# =========================

async def udp_listener_task():
    """
    Tarea asíncrona que escucha paquetes UDP de Ubuntu y actualiza latest_vec.
    Usa recvfrom BLOQUEANTE dentro de run_in_executor, que funciona en Windows.
    """
    global latest_vec, last_update

    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(True)

    print(f"[UDP] Escuchando datos táctiles en 0.0.0.0:{UDP_PORT}")

    expected_len = 4 + 15 * 4  # seq (uint32) + 15 floats

    while True:
        try:
            data, addr = await loop.run_in_executor(
                None, sock.recvfrom, expected_len
            )
        except Exception as e:
            print(f"[UDP] Error en recvfrom: {e}")
            continue

        if len(data) < expected_len:
            print(f"[UDP] Paquete demasiado corto ({len(data)} bytes) desde {addr}")
            continue

        try:
            unpacked = struct.unpack("!I15f", data[:expected_len])
        except struct.error as e:
            print(f"[UDP] Error de unpack: {e}")
            continue

        seq = unpacked[0]
        vals = list(unpacked[1:])

        latest_vec  = vals
        last_update = time.time()
        # Debug opcional:
        # print(f"[UDP] seq={seq}, Fx1={vals[0]:.3f}")

# =========================
# ===== LOOP HÁPTICO ======
# =========================

async def haptics_task():
    """
    Tarea háptica principal:
      - Inicializa bHaptics
      - En bucle HZ Hz:
          * lee latest_vec
          * actualiza baseline de ruido (cuando no hay contacto)
          * calcula magnitud corregida y la filtra con EMA
          * usa histéresis por dedo para decidir contacto
          * vibra SOLO en dedos en contacto
    """
    global was_active, ema_mag, in_contact, baseline_noise

    ok = await bh.registry_and_initialize(APP_ID, API_KEY, "RH8D Tactile Realtime")
    if not ok:
        raise RuntimeError(
            "No se pudo inicializar bHaptics. "
            "Abre bHaptics Player y revisa AppID/API Key."
        )

    connected = await bh.is_bhaptics_device_connected(RIGHT_GLOVE_POS)
    print("Glove derecho conectado:", connected)
    if not connected:
        print("⚠️ No se detecta el guante derecho. "
              "El script corre igual pero no sentirás nada.")

    try:
        while True:
            now = time.time()
            age = now - last_update

            # Caso 1: no llegan datos recientes -> apagar todo y no mandar nada
            if age > TIMEOUT_S:
                if was_active:
                    await bh.stop_all()
                    was_active = False
                    # print("[HAPTICS] Timeout: stop_all()")
                await asyncio.sleep(DT)
                continue

            # Caso 2: tenemos datos recientes
            vec = latest_vec  # 15 floats [-1, 1]

            motors01 = [0.0] * 6
            any_active = False

            for finger in range(5):
                fx, fy, fz = vec[3*finger : 3*finger + 3]

                # magnitud cruda (no la recortamos a [0,1] todavía)
                mag_raw = math.sqrt(fx*fx + fy*fy + fz*fz)

                # Actualizar baseline de ruido SOLO cuando no hay contacto
                if not in_contact[finger]:
                    baseline_noise[finger] = (
                        (1.0 - NOISE_ALPHA) * baseline_noise[finger]
                        + NOISE_ALPHA * mag_raw
                    )

                # Magnitud corregida: restamos baseline + margen
                mag_corr = mag_raw - (baseline_noise[finger] + NOISE_MARGIN)
                if mag_corr < 0.0:
                    mag_corr = 0.0

                # Filtrado EMA de la magnitud corregida (0..algo)
                ema_mag[finger] = (
                    EMA_ALPHA * mag_corr + (1.0 - EMA_ALPHA) * ema_mag[finger]
                )

                m = ema_mag[finger]

                # Histéresis para estado de contacto
                if in_contact[finger]:
                    if m < OFF_THRESHOLD:
                        in_contact[finger] = False
                else:
                    if m > ON_THRESHOLD:
                        in_contact[finger] = True

                # Si está en contacto, escalamos la magnitud corregida a [0..1]
                if in_contact[finger]:
                    # mag_eff = (m - ON_THRESHOLD)/(max_m - ON_THRESHOLD)
                    # suponiendo max_m ~ 1.0
                    mag_eff = (m - ON_THRESHOLD) / max(1e-6, 1.0 - ON_THRESHOLD)
                    mag_eff = clamp01(mag_eff)
                    motors01[finger] = mag_eff
                    any_active = True
                else:
                    motors01[finger] = 0.0

            # muñeca (motor 5) la dejamos a 0 por ahora
            motors01[5] = 0.0

            if any_active:
                await play_frame_glove(motors01)
                was_active = True
            else:
                # No hay ningún dedo en contacto: guante apagado
                if was_active:
                    await bh.stop_all()
                    was_active = False
                    # print("[HAPTICS] Sin contactos: stop_all()")

            await asyncio.sleep(DT)

    finally:
        await bh.stop_all()
        await bh.close()
        print("\n[HAPTICS] Cerrado bHaptics.")

# =========================
# ========= MAIN ==========
# =========================

async def main():
    # Lanzamos en paralelo: escucha UDP + bucle háptico
    listener = asyncio.create_task(udp_listener_task())
    haptics  = asyncio.create_task(haptics_task())

    await asyncio.gather(listener, haptics)

if __name__ == "__main__":
    asyncio.run(main())
