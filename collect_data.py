"""
collect_data.py
Kamera qarşısında işarə cümlələrini çəkib nöqtələri saxlayır.

İdarəetmə:
    1-5   -> cümləni seç
    SPACE -> çəkilişi başlat (geri sayım, sonra klip yazılır)
    Q     -> çıx
"""

import os
import time

import cv2
import numpy as np

from features import (FEATURE_SIZE, detect, draw_hands, draw_reference,
                      extract_keypoints, mp_holistic)

# ---------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------
SENTENCES = {
    "1": ("how_are_you", "How are you?"),
    "2": ("whats_your_name", "What's your name?"),
    "3": ("nice_to_meet_you", "Nice to meet you"),
    "4": ("thank_you", "Thank you"),
    "5": ("have_a_good_day", "Have a good day"),
    "6": ("nothing", "Nothing (random movement)"),

}

DATA_DIR = "data"      # data/how_are_you/0.npy, 1.npy ...
SEQ_LEN = 60           # hər klipdə kadr sayı (~2 saniyə)
COUNTDOWN = 3          # çəkilişdən əvvəl geri sayım (saniyə)
SHOW_REFERENCE = True  # True: burun/çiyin nöqtələri ekranda görünür (yoxlamaq üçün)


def draw_all(frame, results):
    draw_hands(frame, results)
    if SHOW_REFERENCE:
        draw_reference(frame, results)


def count_clips(slug):
    folder = os.path.join(DATA_DIR, slug)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def put_text(frame, text, y, color=(0, 255, 0), scale=0.8):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def record_clip(cap, holistic, slug, label):
    folder = os.path.join(DATA_DIR, slug)
    os.makedirs(folder, exist_ok=True)

    # Geri sayım
    start = time.time()
    while time.time() - start < COUNTDOWN:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        results = detect(frame, holistic)
        draw_all(frame, results)
        left = COUNTDOWN - int(time.time() - start)
        put_text(frame, f"Hazirlas: {left}", 40, (0, 200, 255), 1.2)
        put_text(frame, label, 80, (255, 255, 255))
        cv2.imshow("Data toplama", frame)
        cv2.waitKey(1)

    # Əsas çəkiliş
    sequence = []
    while len(sequence) < SEQ_LEN:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        results = detect(frame, holistic)
        draw_all(frame, results)

        sequence.append(extract_keypoints(results))

        put_text(frame, f"CEKILIR {len(sequence)}/{SEQ_LEN}", 40, (0, 0, 255), 1.0)
        put_text(frame, label, 80, (255, 255, 255))
        cv2.imshow("Data toplama", frame)
        cv2.waitKey(1)

    clip_id = count_clips(slug)
    path = os.path.join(folder, f"{clip_id}.npy")
    arr = np.array(sequence)
    assert arr.shape == (SEQ_LEN, FEATURE_SIZE), arr.shape
    np.save(path, arr)
    print(f"Saxlandi: {path}  forma={arr.shape}")
    return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilmadi.")
        return

    current = "1"

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = detect(frame, holistic)
            draw_all(frame, results)

            slug, label = SENTENCES[current]
            put_text(frame, f"Secilen: {label}", 35)
            put_text(frame, f"Klip sayi: {count_clips(slug)}", 70)
            put_text(frame, "1-5: secim | SPACE: cek | Q: cix",
                     frame.shape[0] - 15, (255, 255, 255), 0.6)
            cv2.imshow("Data toplama", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key != 255 and chr(key) in SENTENCES:
                current = chr(key)
            elif key == 32:
                record_clip(cap, holistic, slug, label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
