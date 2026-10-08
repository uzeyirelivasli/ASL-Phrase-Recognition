import glob
import os

import numpy as np
import torch
import torch.nn as nn

from features import FEATURE_SIZE, normalize_sequence

DATA_DIR = "data"
MODEL_PATH = "model.pt"
SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
VAL_RATIO = 0.2
LR = 1e-3


def load_data():
    classes = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    X, y = [], []
    for idx, name in enumerate(classes):
        files = sorted(glob.glob(os.path.join(DATA_DIR, name, "*.npy")))
        print(f"{name}: {len(files)} klip")
        for f in files:
            arr = np.load(f)
            if arr.shape[1] != FEATURE_SIZE:
                print(f"  KECILDI (yanlis forma {arr.shape}): {f}")
                continue
            X.append(normalize_sequence(arr))
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y), classes


def split_data(y):
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * VAL_RATIO))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


class SignNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
    return (pred == y).float().mean().item(), pred


def main():
    torch.manual_seed(SEED)
    X, y, classes = load_data()
    if len(X) == 0:
        print("Data tapilmadi. Evvelce collect_data.py ile klip cek.")
        return
    print(f"\nCemi {len(X)} klip, {len(classes)} sinif, forma {X.shape}")

    train_idx, val_idx = split_data(y)
    X_train = torch.tensor(X[train_idx]); y_train = torch.tensor(y[train_idx])
    X_val = torch.tensor(X[val_idx]);     y_val = torch.tensor(y[val_idx])
    print(f"Telim: {len(train_idx)} klip | Yoxlama: {len(val_idx)} klip\n")

    model = SignNet(FEATURE_SIZE, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[b]), y_train[b])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)

        train_acc, _ = accuracy(model, X_train, y_train)
        val_acc, _ = accuracy(model, X_val, y_val)

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "input_size": FEATURE_SIZE}, MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {total_loss/len(X_train):.3f} "
                  f"| telim {train_acc:.2f} | yoxlama {val_acc:.2f}")

    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt["model"])
    val_acc, pred = accuracy(model, X_val, y_val)
    print(f"\nEn yaxsi yoxlama deqiqliyi: {val_acc:.2f}")
    for c, name in enumerate(classes):
        mask = y_val == c
        if mask.sum() > 0:
            ok = (pred[mask] == c).float().mean().item()
            print(f"  {name}: {ok:.2f}  ({int(mask.sum())} klip)")
    print(f"\nModel saxlandi: {MODEL_PATH}")


if __name__ == "__main__":
    main()