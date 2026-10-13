"""Corre el experimento completo del TP y guarda metricas y figuras reproducibles.

Genera en experimentos/resultados/:
  - metricas.json          numeros que cita el informe
  - curvas.png             loss y accuracy por epoca
  - matriz_confusion.png   matriz 24x24 en test sintetico
  - cnn_vs_ks.csv          comparativa por audio de prueba real
"""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(SALIDA, exist_ok=True)
sns.set(style="whitegrid")


def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fijar_semillas(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SR = 16000
DUR = 4.0
N_SAMPLES = int(SR * DUR)
TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
T2I = {n: i for i, n in enumerate(TONICS)}
BASE_F = {"C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
          "F": 349.23, "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00,
          "A#": 466.16, "B": 493.88}
KEYS = [f"{t} maj" for t in TONICS] + [f"{t} min" for t in TONICS]
K2I = {k: i for i, k in enumerate(KEYS)}


def tono(freq, dur, sr, rng, gain=0.5):
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    x = np.zeros(n, dtype=np.float32)
    for h, a in enumerate([1.0, 0.4, 0.18], start=1):
        x += (a * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)
    a_n = min(int(0.02 * sr), n // 4)
    r_n = min(int(0.08 * sr), n // 4)
    env = np.ones(n, dtype=np.float32)
    env[:a_n] = np.linspace(0, 1, a_n)
    env[-r_n:] = np.linspace(1, 0, r_n)
    return (gain * x * env).astype(np.float32)


def acorde(root_hz, ivs, dur, sr, rng):
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    for iv in ivs:
        f = root_hz * (2.0 ** (iv / 12.0)) * (1.0 + rng.uniform(-0.004, 0.004))
        x += tono(f, dur, sr, rng, gain=0.33)
    return x


def prog_key(key_idx, rng, snr_db=None):
    tonica = TONICS[key_idx % 12]
    is_maj = key_idx < 12
    rf = BASE_F[tonica] / 2.0
    roots = [0, 5, 7, 0]
    y = np.zeros(int(SR * DUR), dtype=np.float32)
    seg = int(SR * 1.0)
    for j, r in enumerate(roots):
        r_hz = rf * (2.0 ** (r / 12.0))
        if is_maj:
            ivs = [0, 4, 7] if j in [0, 1, 3] else [0, 4, 7, 10]
        else:
            ivs = [0, 3, 7] if j in [0, 1, 3] else [0, 4, 7, 10]
        ch = acorde(r_hz, ivs, 1.0, sr=SR, rng=rng)
        y[j * seg:(j + 1) * seg] += ch[:seg]
    if snr_db is None:
        snr_db = float(rng.uniform(12, 25))
    p = np.mean(y ** 2) + 1e-9
    y += rng.normal(0, math.sqrt(p / (10 ** (snr_db / 10))), size=y.shape).astype(np.float32)
    return (0.9 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


class KeyDataset(Dataset):
    def __init__(self, labels, split="train"):
        self.labels = list(labels)
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        rng = np.random if self.split == "train" else np.random.RandomState(9000 + idx)
        return torch.from_numpy(prog_key(y, rng)).unsqueeze(0), torch.tensor(y, dtype=torch.long)


class KeyCNN1D(nn.Module):
    def __init__(self, n_classes=24, dropout=0.3):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=128, stride=8, padding=64),
                                nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4))
        self.b2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=16, padding=8),
                                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.b3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=7, padding=3),
                                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.b4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=5, padding=2),
                                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 96), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(96, n_classes))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.head(self.gap(x))


def mirex(t, p):
    if t == p:
        return 1.0
    tt, mt = t % 12, t < 12
    pt, mp = p % 12, p < 12
    if mt == mp and (pt - tt) % 12 == 7:
        return 0.5
    if (mt != mp) and ((tt == 0 and pt == 9) or (pt == 0 and tt == 9)
                       or ((pt - tt) % 12 == 9 and mt) or ((tt - pt) % 12 == 9 and mp)):
        return 0.3
    if tt == pt and mt != mp:
        return 0.3
    return 0.0


MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def baseline_ks_wave(y):
    import librosa
    ch = librosa.feature.chroma_cqt(y=y, sr=SR).mean(axis=1)
    best, bi = -2, (0, "maj")
    for t in range(12):
        for m, prof in [("maj", MAJ), ("min", MIN)]:
            r = np.corrcoef(ch, np.roll(prof, t))[0, 1]
            if r > best:
                best, bi = r, (t, m)
    return f"{TONICS[bi[0]]} {bi[1]}", float(best)


def cargar_audio_Xs(path, sr=SR, dur=DUR):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * dur)
    vs = []
    for i in range(0, max(len(y), 1), n):
        s = y[i:i + n]
        s = np.pad(s, (0, n - len(s))) if len(s) < n else s
        vs.append((0.95 * s / (np.max(np.abs(s)) + 1e-9)).astype(np.float32))
        if i + n >= len(y):
            break
    return vs


def predecir_cnn(model, path, maxv=6):
    model.eval()
    vs = cargar_audio_Xs(path)
    if len(vs) > maxv:
        vs = [vs[i] for i in np.linspace(0, len(vs) - 1, maxv).round().astype(int)]
    xb = torch.stack([torch.from_numpy(w).unsqueeze(0) for w in vs]).to(DEVICE)
    with torch.no_grad():
        v = torch.softmax(model(xb), 1).mean(0)
    i = int(v.argmax())
    top3 = sorted(range(24), key=lambda k: float(v[k]), reverse=True)[:3]
    return KEYS[i], float(v[i]), [(KEYS[k], round(float(v[k]), 4)) for k in top3]


def main():
    res = {"entorno": {"python": sys.version.split()[0], "torch": torch.__version__,
                       "device": DEVICE, "seed": 42}}

    NT, NV, NTs = 80, 20, 20
    tr = [i for i in range(24) for _ in range(NT)]
    va = [i for i in range(24) for _ in range(NV)]
    te = [i for i in range(24) for _ in range(NTs)]
    train_ds, val_ds, test_ds = KeyDataset(tr, "train"), KeyDataset(va, "val"), KeyDataset(te, "test")
    BATCH = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    res["dataset"] = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
                      "clases": 24, "sr": SR, "dur_s": DUR, "n_samples": N_SAMPLES,
                      "snr_db": [12, 25], "batch": BATCH}
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    model = KeyCNN1D(24).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    res["modelo"] = {"nombre": "KeyCNN1D", "params": n_par,
                     "kernels": [128, 16, 7, 5], "canales": [16, 32, 64, 128],
                     "dropout": 0.3, "entrada": [1, N_SAMPLES], "salida": 24}
    print(f"Params: {n_par:,}")

    EPOCHS, LR = 12, 1e-3
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def epoch(loader, train):
        model.train(train)
        tl, ok, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                lg = model(xb)
                loss = crit(lg, yb)
                if train:
                    loss.backward()
                    opt.step()
            tl += loss.item() * len(xb)
            ok += (lg.argmax(1) == yb).sum().item()
            n += len(xb)
        return tl / n, ok / n

    hist = {"tl": [], "vl": [], "ta": [], "va": []}
    best, bs = 0, None
    for ep in range(1, EPOCHS + 1):
        a, b = epoch(train_loader, True)
        c, d = epoch(val_loader, False)
        hist["tl"].append(a); hist["vl"].append(c)
        hist["ta"].append(b); hist["va"].append(d)
        if d > best:
            best = d
            bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Ep {ep:02d} train {a:.4f}/{b:.3f} || val {c:.4f}/{d:.3f}", flush=True)
    model.load_state_dict({k: v.to(DEVICE) for k, v in bs.items()})
    res["entrenamiento"] = {"epochs": EPOCHS, "lr": LR, "optimizador": "Adam",
                            "perdida": "CrossEntropyLoss", "historial": hist,
                            "best_val_acc": best}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["tl"], label="train", marker="o")
    ax[0].plot(hist["vl"], label="val", marker="s")
    ax[0].set_title("Perdida (CrossEntropy)"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(hist["ta"], label="train", marker="o")
    ax[1].plot(hist["va"], label="val", marker="s")
    ax[1].set_title("Accuracy (24 clases)"); ax[1].set_xlabel("epoca"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "curvas.png"), dpi=150)
    plt.close()

    model.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            P += model(xb.to(DEVICE)).argmax(1).cpu().tolist()
            T += yb.tolist()
    acc = accuracy_score(T, P)
    mx = float(np.mean([mirex(t, p) for t, p in zip(T, P)]))
    acc_tonica = float(np.mean([(t % 12) == (p % 12) for t, p in zip(T, P)]))
    acc_modo = float(np.mean([(t < 12) == (p < 12) for t, p in zip(T, P)]))
    res["test_sintetico"] = {"accuracy": acc, "mirex": mx,
                             "accuracy_tonica": acc_tonica, "accuracy_modo": acc_modo,
                             "n": len(T)}
    print(f"Test acc {acc:.3f} | MIREX {mx:.3f} | tonica {acc_tonica:.3f} | modo {acc_modo:.3f}")

    cm = confusion_matrix(T, P, labels=list(range(24)))
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=KEYS, yticklabels=KEYS)
    plt.title("Matriz de confusion 24x24 (test sintetico)")
    plt.xlabel("prediccion"); plt.ylabel("verdad")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "matriz_confusion.png"), dpi=150)
    plt.close()
    res["test_sintetico"]["matriz_confusion"] = cm.tolist()

    # Baseline K-S sobre el mismo test sintetico
    Pk = []
    for idx in range(len(test_ds)):
        w, _ = test_ds[idx]
        lab, _ = baseline_ks_wave(w.numpy()[0])
        Pk.append(K2I[lab])
    acc_ks = accuracy_score(T, Pk)
    mx_ks = float(np.mean([mirex(t, p) for t, p in zip(T, Pk)]))
    res["test_sintetico_ks"] = {"accuracy": acc_ks, "mirex": mx_ks}
    print(f"K-S sintetico acc {acc_ks:.3f} | MIREX {mx_ks:.3f}")

    # Audios reales versionados
    carpeta = os.path.join(RAIZ, "audios_prueba_final")
    verdad = {"test_C_maj_I-IV-V-I.wav": "C maj", "test_G_maj_I-IV-V-I.wav": "G maj",
              "test_E_maj_I-IV-V-I.wav": "E maj", "test_A_min_i-iv-V-i.wav": "A min",
              "demo_blues_A_12bar_key-A-maj.wav": "A maj"}
    filas = []
    if os.path.isdir(carpeta):
        for f in sorted(os.listdir(carpeta)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(carpeta, f)
            cnn, conf, top3 = predecir_cnn(model, path)
            ks, r = baseline_ks_wave(__import__("librosa").load(path, sr=SR, mono=True)[0])
            tru = verdad.get(f, "?")
            filas.append({"archivo": f, "verdad": tru, "cnn": cnn, "cnn_conf": round(conf, 4),
                          "cnn_top3": top3, "ks": ks, "ks_r": round(r, 4),
                          "cnn_ok": int(cnn == tru), "ks_ok": int(ks == tru),
                          "cnn_mirex": mirex(K2I[tru], K2I[cnn]) if tru in K2I else None,
                          "ks_mirex": mirex(K2I[tru], K2I[ks]) if tru in K2I else None})
            print(f"{f:40s} verdad {tru:7s} cnn {cnn:7s} ({conf:.2f}) ks {ks:7s} (r={r:.2f})")
    res["audios_reales"] = filas
    if filas:
        res["audios_reales_resumen"] = {
            "n": len(filas),
            "cnn_acc": float(np.mean([f["cnn_ok"] for f in filas])),
            "ks_acc": float(np.mean([f["ks_ok"] for f in filas])),
            "cnn_mirex": float(np.mean([f["cnn_mirex"] for f in filas])),
            "ks_mirex": float(np.mean([f["ks_mirex"] for f in filas])),
        }

    # Prueba sobre el MP3 real (Pappo) si esta disponible
    mp3 = [f for f in os.listdir(RAIZ) if f.lower().endswith(".mp3")]
    if mp3:
        path = os.path.join(RAIZ, mp3[0])
        try:
            cnn, conf, top3 = predecir_cnn(model, path)
            ks, r = baseline_ks_wave(__import__("librosa").load(path, sr=SR, mono=True)[0])
            res["mp3_real"] = {"archivo": mp3[0], "cnn": cnn, "cnn_conf": round(conf, 4),
                               "cnn_top3": top3, "ks": ks, "ks_r": round(r, 4)}
            print(f"{mp3[0]}: cnn {cnn} ({conf:.2f}) | ks {ks} (r={r:.2f})")
        except Exception as e:
            print("MP3 fallo:", e)

    # Regla cross-harp
    res["cross_harp"] = {t: TONICS[(T2I[t] + 5) % 12] for t in TONICS}

    with open(os.path.join(SALIDA, "metricas.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    if filas:
        import csv
        with open(os.path.join(SALIDA, "cnn_vs_ks.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["archivo", "verdad", "cnn", "cnn_conf", "ks", "ks_r", "cnn_ok", "ks_ok"])
            for f in filas:
                w.writerow([f["archivo"], f["verdad"], f["cnn"], f["cnn_conf"],
                            f["ks"], f["ks_r"], f["cnn_ok"], f["ks_ok"]])

    torch.save(model.state_dict(), os.path.join(SALIDA, "keycnn1d_best.pt"))
    print("\nListo ->", SALIDA)


if __name__ == "__main__":
    main()
