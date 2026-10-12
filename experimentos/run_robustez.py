"""Estres fuera de distribucion (OOD): mide si la CNN aprendio tonalidad o memorizo el generador.

Reusa el modelo entrenado en experimentos/resultados/keycnn1d_best.pt y lo evalua contra
variantes que NO estuvieron en entrenamiento: otra progresion, otro registro, otro timbre,
ruido fuerte y cambio de afinacion. Compara siempre contra el baseline K-S.

Salida: experimentos/resultados/robustez.json + robustez.png
"""
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from run_experimentos import (DEVICE, K2I, KEYS, N_SAMPLES, SR, TONICS, BASE_F, DUR,
                              KeyCNN1D, acorde, baseline_ks_wave, fijar_semillas, mirex)

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
sns.set(style="whitegrid")
fijar_semillas(123)


def prog_variante(key_idx, rng, roots=(0, 5, 7, 0), octava=0.5, armonicos=(1.0, 0.4, 0.18),
                  snr_db=None, afinacion_hz=440.0, septima_V=True):
    """Genera 4 s de una tonalidad permitiendo mover progresion, registro, timbre y afinacion."""
    tonica = TONICS[key_idx % 12]
    is_maj = key_idx < 12
    rf = BASE_F[tonica] * (afinacion_hz / 440.0) * octava
    y = np.zeros(int(SR * DUR), dtype=np.float32)
    seg = int(SR * DUR / len(roots))

    def tono_v(freq, dur, gain):
        n = int(SR * dur)
        t = np.arange(n, dtype=np.float32) / SR
        x = np.zeros(n, dtype=np.float32)
        for h, a in enumerate(armonicos, start=1):
            x += (a * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)
        a_n = min(int(0.02 * SR), n // 4)
        r_n = min(int(0.08 * SR), n // 4)
        env = np.ones(n, dtype=np.float32)
        env[:a_n] = np.linspace(0, 1, a_n)
        env[-r_n:] = np.linspace(1, 0, r_n)
        return (gain * x * env).astype(np.float32)

    for j, r in enumerate(roots):
        r_hz = rf * (2.0 ** (r / 12.0))
        if is_maj:
            base = [0, 4, 7]
        else:
            base = [0, 3, 7]
        # grado V (7 semitonos) va mayor con septima en blues/menor armonica
        if r == 7 and septima_V:
            ivs = [0, 4, 7, 10]
        else:
            ivs = base
        ch = np.zeros(seg, dtype=np.float32)
        for iv in ivs:
            f = r_hz * (2.0 ** (iv / 12.0)) * (1.0 + rng.uniform(-0.004, 0.004))
            ch += tono_v(f, seg / SR, 0.33)[:seg]
        y[j * seg:(j + 1) * seg] += ch
    if snr_db is None:
        snr_db = float(rng.uniform(12, 25))
    p = np.mean(y ** 2) + 1e-9
    y += rng.normal(0, math.sqrt(p / (10 ** (snr_db / 10))), size=y.shape).astype(np.float32)
    return (0.9 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


ESCENARIOS = {
    "base (igual que train)": dict(),
    "progresion I-V-vi-IV": dict(roots=(0, 7, 9, 5)),
    "progresion ii-V-I": dict(roots=(2, 7, 0, 0)),
    "registro +1 octava": dict(octava=1.0),
    "registro -1 octava": dict(octava=0.25),
    "timbre brillante (6 arm.)": dict(armonicos=(1.0, 0.8, 0.6, 0.45, 0.3, 0.2)),
    "timbre opaco (1 arm.)": dict(armonicos=(1.0,)),
    "ruido fuerte SNR 5 dB": dict(snr_db=5.0),
    "ruido fuerte SNR 0 dB": dict(snr_db=0.0),
    "afinacion 432 Hz": dict(afinacion_hz=432.0),
    "afinacion 445 Hz": dict(afinacion_hz=445.0),
}


def main():
    model = KeyCNN1D(24).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SALIDA, "keycnn1d_best.pt"), map_location=DEVICE))
    model.eval()

    N_POR_CLASE = 5
    res = {}
    for nombre, kw in ESCENARIOS.items():
        T, Pc, Pk = [], [], []
        for k in range(24):
            for j in range(N_POR_CLASE):
                rng = np.random.RandomState(50000 + k * 100 + j)
                w = prog_variante(k, rng, **kw)
                with torch.no_grad():
                    xb = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    pc = int(model(xb).argmax(1).item())
                lab, _ = baseline_ks_wave(w)
                T.append(k); Pc.append(pc); Pk.append(K2I[lab])
        acc_c = float(np.mean([t == p for t, p in zip(T, Pc)]))
        acc_k = float(np.mean([t == p for t, p in zip(T, Pk)]))
        mx_c = float(np.mean([mirex(t, p) for t, p in zip(T, Pc)]))
        mx_k = float(np.mean([mirex(t, p) for t, p in zip(T, Pk)]))
        ton_c = float(np.mean([(t % 12) == (p % 12) for t, p in zip(T, Pc)]))
        ton_k = float(np.mean([(t % 12) == (p % 12) for t, p in zip(T, Pk)]))
        res[nombre] = {"n": len(T), "cnn_acc": acc_c, "ks_acc": acc_k,
                       "cnn_mirex": mx_c, "ks_mirex": mx_k,
                       "cnn_acc_tonica": ton_c, "ks_acc_tonica": ton_k}
        print(f"{nombre:28s} CNN acc {acc_c:.3f} mirex {mx_c:.3f} | K-S acc {acc_k:.3f} mirex {mx_k:.3f}", flush=True)

    with open(os.path.join(SALIDA, "robustez.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    nombres = list(res.keys())
    x = np.arange(len(nombres))
    ancho = 0.38
    plt.figure(figsize=(12, 5.5))
    plt.bar(x - ancho / 2, [res[n]["cnn_acc"] for n in nombres], ancho, label="CNN 1D")
    plt.bar(x + ancho / 2, [res[n]["ks_acc"] for n in nombres], ancho, label="Baseline K-S")
    plt.xticks(x, nombres, rotation=38, ha="right")
    plt.ylabel("Accuracy (24 clases)")
    plt.title("Robustez fuera de distribucion: CNN 1D vs cromagrama + Krumhansl-Schmuckler")
    plt.axhline(1 / 24, ls="--", c="gray", lw=1, label="azar (1/24)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "robustez.png"), dpi=150)
    plt.close()
    print("\nListo ->", SALIDA)


if __name__ == "__main__":
    main()
