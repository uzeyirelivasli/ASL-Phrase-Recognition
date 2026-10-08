"""
live_demo.py
Kamerada işarə edəndə ekranda cümləni göstərir.

İşə salmaq:
    python live_demo.py
Çıxmaq: Q
"""

import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

from features import (detect, draw_hands, extract_keypoints, mp_holistic,
                      normalize_sequence)
from train import SignNet

# ---------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------
MODEL_PATH = "model.pt"
SEQ_LEN = 60              # təlimdəki klip uzunluğu ilə eyni olmalıdır
CONF_THRESHOLD = 0.85     # bir kadrdakı təxmin üçün minimum əminlik
MIN_HAND_FRAMES = 20      # son 60 kadrın ən azı neçəsində əl görünməlidir
MIN_MOTION = 0.01         # əllərin minimum hərəkəti (dayanmış əlləri süzür)
STABLE_N = 10             # son neçə təxmin nəzərə alınsın
STABLE_VOTES = 7          # bunların ən azı neçəsi eyni cümlə olmalıdır
HOLD_SECONDS = 2.0        # tapılan cümlə ekranda neçə saniyə qalsın
DEBUG = True              # True: ekranda ehtimalları və hərəkət dəyərini göstərir

LABELS = {
    "how_are_you": "How are you?",
    "whats_your_name": "What's your name?",
    "nice_to_meet_you": "Nice to meet you",
    "thank_you": "Thank you",
    "have_a_good_day": "Have a good day",
}


def load_model():
    ckpt = torch.load(MODEL_PATH)
    classes = ckpt["classes"]
    model = SignNet(ckpt["input_size"], len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes


def hand_motion(seq_norm):
    """Əllərin orta hərəkət miqdarı. Dayanmış əllərdə ~0 olur.

    Yalnız iki ardıcıl kadrda da görünən nöqtələr hesaba alınır.
    """
    hands = seq_norm[:, 9:]
    a, b = hands[:-1], hands[1:]
    mask = (a != 0) & (b != 0)
    if not mask.any():
        return 0.0
    return float(np.abs(b - a)[mask].mean())


def put(frame, text, x, y, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def main():
    model, classes = load_model()
    print("Modelin siniflari:", classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilmadi.")
        return

    buffer = deque(maxlen=SEQ_LEN)       # son 60 kadrın nöqtələri
    votes = deque(maxlen=STABLE_N)       # son təxminlər
    shown_text, shown_conf, shown_until = "", 0.0, 0.0
    probs_now, motion_now, hand_frames = None, 0.0, 0

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = detect(frame, holistic)
            draw_hands(frame, results)

            buffer.append(extract_keypoints(results))

            if len(buffer) == SEQ_LEN:
                seq = np.array(buffer)
                hand_frames = int(np.sum(np.any(seq[:, 9:] != 0, axis=1)))
                seq_norm = normalize_sequence(seq)
                motion_now = hand_motion(seq_norm)

                if hand_frames >= MIN_HAND_FRAMES and motion_now >= MIN_MOTION:
                    x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        probs_now = torch.softmax(model(x), dim=1)[0].numpy()
                    idx = int(probs_now.argmax())
                    conf = float(probs_now[idx])
                    votes.append(classes[idx] if conf >= CONF_THRESHOLD else "none")
                else:
                    probs_now = None
                    votes.append("none")

                # Son STABLE_N təxminin çoxu eyni cümlədirsə, göstər
                name, count = Counter(votes).most_common(1)[0]
                if (name not in ("none", "nothing") and count >= STABLE_VOTES):
                    shown_text = LABELS.get(name, name)
                    shown_conf = float(probs_now[classes.index(name)]) if probs_now is not None else 0.0
                    shown_until = time.time() + HOLD_SECONDS

            h, w = frame.shape[:2]

            # Debug paneli
            if DEBUG:
                put(frame, f"el kadrlari: {hand_frames}/{SEQ_LEN}", 10, 22)
                put(frame, f"hereket: {motion_now:.4f} (min {MIN_MOTION})", 10, 44)
                if probs_now is not None:
                    for i, c in enumerate(classes):
                        put(frame, f"{c}: {probs_now[i]:.2f}", 10, 70 + i * 22,
                            (0, 255, 255) if probs_now[i] >= CONF_THRESHOLD else (200, 200, 200))
                else:
                    put(frame, "tanima yoxdur (el yoxdur / hereket yoxdur)", 10, 70, (0, 120, 255))

            # Nəticə
            if time.time() < shown_until:
                cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
                put(frame, shown_text, 15, h - 25, (0, 255, 0), 1.2, 3)
                put(frame, f"{shown_conf:.0%}", w - 110, h - 25, (200, 200, 200), 0.9, 2)

            cv2.imshow("ASL demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()