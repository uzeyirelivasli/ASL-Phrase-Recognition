"""
features.py
Çəkiliş, təlim və canlı demo skriptlərinin ORTAQ istifadə etdiyi funksiyalar.
Nöqtə çıxarma məntiqi burada bir yerdə olmalıdır ki, hamısı eyni formatda işləsin.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Bədən modelindən yalnız bu nöqtələri saxlayırıq:
# 0 = burun, 11 = sol çiyin, 12 = sağ çiyin
REF_POINTS = (0, 11, 12)

# Bir kadrın vektor ölçüsü: 3 istinad nöqtəsi x 3 + 2 əl x 21 nöqtə x 3 = 135
FEATURE_SIZE = len(REF_POINTS) * 3 + 21 * 3 * 2


def detect(frame, holistic):
    """OpenCV BGR verir, MediaPipe RGB istəyir, ona görə çeviririk."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def extract_keypoints(results):
    """Bir kadrdan 135 rəqəmlik vektor çıxarır.

    Burun + iki çiyin : 3 nöqtə x (x, y, z) = 9
    Sol əl            : 21 nöqtə x (x, y, z) = 63
    Sağ əl            : 21 nöqtə x (x, y, z) = 63

    Nöqtə görünmürsə, sıfırlarla doldurulur ki, ölçü həmişə sabit qalsın.
    """
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ref = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in REF_POINTS]).flatten()
    else:
        ref = np.zeros(len(REF_POINTS) * 3)

    if results.left_hand_landmarks:
        lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([ref, lh, rh])


def draw_hands(frame, results):
    """Yalnız əlləri çəkir. Canlı demo üçün bu kifayətdir."""
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_reference(frame, results):
    """Burun və çiyinləri kiçik nöqtə kimi çəkir (yalnız çəkiliş zamanı yoxlamaq üçün)."""
    if not results.pose_landmarks:
        return
    h, w = frame.shape[:2]
    lm = results.pose_landmarks.landmark
    for i in REF_POINTS:
        cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 6, (0, 255, 255), -1)


def normalize_frame(v):
    """Bir kadrın vektorunu (135) normallaşdırır."""
    nose, ls, rs = v[0:3], v[3:6], v[6:9]
    scale = np.linalg.norm(ls[:2] - rs[:2])
    if scale < 1e-6:           # bədən tapılmayıb
        return np.zeros_like(v)
    pts = v.reshape(-1, 3).copy()          # 45 nöqtə x 3
    visible = np.any(pts != 0, axis=1)
    pts = (pts - nose) / scale
    pts[~visible] = 0
    return pts.flatten()


def normalize_sequence(seq):
    """(60, 135) ardıcıllığının hər kadrını normallaşdırır."""
    return np.array([normalize_frame(f) for f in seq])