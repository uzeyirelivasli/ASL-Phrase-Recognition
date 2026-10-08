# ASL Phrase Recognition

Real-time recognition of a small set of American Sign Language (ASL) phrases from a webcam.
The model works on **hand and upper-body landmarks** instead of raw video, so it is lightweight and runs on a normal laptop.

[> **Demo video:** _add link or drag the video file here_](https://github.com/user-attachments/assets/17edf366-8f81-4f54-b6c1-56df0b599362)

## Recognized phrases

| # | Phrase |
|---|--------|
| 1 | How are you? |
| 2 | What's your name? |
| 3 | Nice to meet you |
| 4 | Thank you |
| 5 | Have a good day |
| 6 | _nothing_ (random movement, ignored by the demo) |

The extra `nothing` class teaches the model what is **not** a phrase (resting hands, scratching your head, waving...). Without it, the model is forced to pick one of the five phrases for any movement.

## How it works

```
Webcam frame -> MediaPipe Holistic -> landmarks -> 135 numbers per frame
                                                        |
               60 frames = one clip, shape (60, 135) <--+
                                                        |
                         normalize -> LSTM -> phrase probabilities
```

1. **Landmarks.** MediaPipe Holistic finds both hands (21 points each), the nose and both shoulders. Each frame becomes a vector of 135 numbers.
2. **Clips.** 60 consecutive frames (about 2 seconds) form one sample.
3. **Normalization.** Points are measured relative to the nose and divided by shoulder width, so the model does not depend on where you stand or how far you are from the camera.
4. **Model.** A 2-layer LSTM (64 units) followed by a linear layer classifies the clip.
5. **Live demo.** A sliding window of the last 60 frames is classified continuously. A phrase is shown only when the model is confident, the hands are moving, and most of the last 10 predictions agree.

Only the hands are drawn on screen in the demo. The nose and shoulders are used by the model internally but not displayed.

## Project structure

```
collect_data.py   record training clips from the webcam
features.py       shared landmark extraction, normalization and drawing helpers
train.py          train the LSTM and save model.pt
live_demo.py      real-time recognition from the webcam
camera_test.py    quick check that the camera and hand tracking work
```

`features.py` is shared by all scripts, so training and live recognition always use exactly the same feature extraction.

## Setup

Python 3.9-3.12 is recommended (MediaPipe 0.10.14).

```bash
pip install -r requirements.txt
```

## Usage

**1. Collect data**

```bash
python collect_data.py
```

| Key | Action |
|-----|--------|
| `1`-`6` | select phrase (6 = nothing) |
| `SPACE` | start recording (3-second countdown, then 60 frames) |
| `Q` | quit |

Clips are saved to `data/<phrase>/<n>.npy`. Tips for good data:
- Use one consistent variant of each sign.
- Vary distance, lighting and clothing between sessions.
- For the `nothing` class, record many different movements, especially ones that end near the face or chest.

**2. Train**

```bash
python train.py
```

Saves the best model to `model.pt`.

**3. Run the live demo**

```bash
python live_demo.py
```

Press `Q` to quit. Set `DEBUG = False` in `live_demo.py` to hide the probability panel.

Tunable settings at the top of `live_demo.py`: `CONF_THRESHOLD`, `MIN_MOTION`, `STABLE_N`, `STABLE_VOTES`, `HOLD_SECONDS`.

## Results

Dataset: 128 clips (20 per phrase, 28 for `nothing`), recorded by one person, with an 80/20 train/validation split per class.

The best validation accuracy was 1.00 on 24 clips (4 per class). This number is optimistic: the validation clips come from the same recording sessions as the training clips and the validation set is very small. Real-world accuracy, especially for other signers, is expected to be lower.

## Limitations

- **Hands and upper body only.** ASL grammar also uses facial expressions and head movement (for example, raised or lowered eyebrows for questions). These are not used here.
- **Single signer.** The data was recorded by one person. Other people may be recognized less reliably.
- **Tiny vocabulary.** Five phrases are recognized, not full ASL or fingerspelling.
- **Fixed-length window.** The model expects about 2 seconds per phrase, and a sliding window can be uncertain while a sign is only half finished.
- **Sign variants.** Many ASL signs have regional variants. This project uses one variant per phrase.

## Ideas for future work

- Record clips from several signers and in more environments
- Add face landmarks for non-manual markers
- Data augmentation (noise, small shifts, speed changes)
- More phrases and a held-out test set from a different day or person
- Replace the fixed 60-frame window with sign start/end detection

## Credits

- [MediaPipe](https://developers.google.com/mediapipe) for landmark detection
- [PyTorch](https://pytorch.org/) for the model
- Sign references: Lifeprint, Handspeak
