"""Diagnostico del fallo de octava: adonde se van las predicciones cuando cambia el registro.

Si la CNN hubiera aprendido tonalidad (relacion entre notas), cambiar la octava no deberia
afectarla: la tonalidad de C mayor es C mayor en cualquier registro. Este script mide el
patron de error para distinguir "aprendio tonalidad" de "memorizo frecuencia absoluta".

Salida: experimentos/resultados/diagnostico.json + diagnostico_octava.png
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from run_experimentos import DEVICE, K2I, KEYS, TONICS, KeyCNN1D, baseline_ks_wave, fijar_semillas
from run_robustez import prog_variante

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
sns.set(style="whitegrid")
fijar_semillas(123)


def main():
    model = KeyCNN1D(24).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SALIDA, "keycnn1d_best.pt"), map_location=DEVICE))
    model.eval()

    res = {}

    # 1) Barrido fino de desplazamiento en semitonos (no solo octavas)
    #    Si memorizo frecuencia absoluta, al subir k semitonos deberia predecir tonica+k.
    barrido = {}
    for st in [-12, -7, -5, -2, -1, 0, 1, 2, 5, 7, 12]:
        factor = 2.0 ** (st / 12.0)
        aciertos_reales, aciertos_desplazados, n = 0, 0, 0
        for k in range(24):
            for j in range(3):
                rng = np.random.RandomState(70000 + k * 100 + j)
                w = prog_variante(k, rng, octava=0.5 * factor)
                with torch.no_grad():
                    xb = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    p = int(model(xb).argmax(1).item())
                # etiqueta verdadera (la tonalidad no cambia al transponer el registro... pero si
                # transponemos todo el audio k semitonos, la tonalidad percibida SI se mueve k)
                esperado_transpuesto = ((k % 12) + st) % 12 + (0 if k < 12 else 12)
                aciertos_reales += int(p == k)
                aciertos_desplazados += int(p == esperado_transpuesto)
                n += 1
        barrido[st] = {"n": n,
                       "acc_misma_etiqueta": aciertos_reales / n,
                       "acc_etiqueta_transpuesta": aciertos_desplazados / n}
        print(f"shift {st:+3d} st | misma {aciertos_reales/n:.3f} | transpuesta {aciertos_desplazados/n:.3f}", flush=True)
    res["barrido_semitonos"] = barrido

    # 2) Distribucion de predicciones en +1 octava (colapso?)
    for nombre, oct_f in [("+1 octava", 1.0), ("-1 octava", 0.25)]:
        preds = []
        for k in range(24):
            for j in range(5):
                rng = np.random.RandomState(80000 + k * 100 + j)
                w = prog_variante(k, rng, octava=oct_f)
                with torch.no_grad():
                    xb = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    preds.append(int(model(xb).argmax(1).item()))
        c = Counter(preds)
        top = [(KEYS[k], v) for k, v in c.most_common(5)]
        res[f"colapso_{nombre.replace(' ', '_')}"] = {
            "clases_distintas_predichas": len(c),
            "top5": top,
            "frac_en_top1": c.most_common(1)[0][1] / len(preds),
        }
        print(f"{nombre}: {len(c)} clases distintas, top5 {top}", flush=True)

    # 3) Grafico del barrido
    sts = sorted(barrido.keys())
    plt.figure(figsize=(10, 5))
    plt.plot(sts, [barrido[s]["acc_misma_etiqueta"] for s in sts], marker="o",
             label="acierta la etiqueta original")
    plt.plot(sts, [barrido[s]["acc_etiqueta_transpuesta"] for s in sts], marker="s",
             label="acierta la etiqueta transpuesta (+k semitonos)")
    plt.axhline(1 / 24, ls="--", c="gray", lw=1, label="azar (1/24)")
    plt.xlabel("desplazamiento aplicado al audio (semitonos)")
    plt.ylabel("Accuracy")
    plt.title("Que sigue la CNN: la tonalidad o la frecuencia absoluta")
    plt.xticks(sts)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "diagnostico_octava.png"), dpi=150)
    plt.close()

    with open(os.path.join(SALIDA, "diagnostico.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\nListo ->", SALIDA)


if __name__ == "__main__":
    main()
