"""Agrega al notebook principal la seccion de hallazgos OOD, alineada con el informe."""
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(RAIZ, "TP_Final_Revisado_Profe_24Keys.ipynb")

MARCA = "PASO 11 — Validación fuera de distribución"

MD_HALLAZGO = """## PASO 11 — Validación fuera de distribución (resultado principal del TP)

> Los scripts `experimentos/run_robustez.py` y `experimentos/run_diagnostico.py` del repositorio
> ejecutan esta validación completa. Acá se resume la conclusión. Informe completo en `informe/informe.pdf`.

### El 100 % del test sintético no alcanza

| Método | Test sintético acc | MIREX |
|---|---|---|
| CNN 1D (80.968 params) | 1,000 | 1,000 |
| Baseline K-S (0 params) | 1,000 | 1,000 |

La línea de base **empata** con la red. El benchmark sintético está saturado: no discrimina
entre métodos, así que no sirve para afirmar que la CNN aprendió algo útil.

### Evaluación fuera de distribución (11 escenarios, la etiqueta no cambia)

| Escenario | CNN 1D | Baseline K-S |
|---|---|---|
| progresión nueva `I-V-vi-IV` | **0,925** | 0,500 |
| progresión nueva `ii-V-I` | **1,000** | 0,500 |
| timbre brillante (6 armónicos) | 0,908 | 1,000 |
| ruido fuerte SNR 0 dB | 0,808 | 1,000 |
| **transposición 1 octava** | **0,042** | **1,000** |
| audio real (5 temas conocidos) | 0,800 | **1,000** |

La CNN generaliza mejor a progresiones nuevas, pero **colapsa al nivel del azar (1/24 = 0,042)
ante un cambio de octava**, donde el método clásico no se degrada.

### Diagnóstico: la red mide altura absoluta, no tonalidad

Barrido fino de transposición:

| Desplazamiento | Acierta etiqueta original | Predice etiqueta transpuesta |
|---|---|---|
| −1 semitono | 0,083 | **0,917** |
| 0 | 1,000 | 1,000 |
| +1 semitono | 0,000 | **0,917** |

Al mover el audio un semitono, la red predice la etiqueta **transpuesta** el 91,7 % de las veces.
Está siguiendo la frecuencia absoluta con precisión de semitono.

**Causa:** el generador ubica cada tonalidad siempre en la misma octava (`BASE_F[tonica]/2`), por lo
que dentro del dataset "tonalidad" y "frecuencia absoluta" son predictores equivalentes. El gradiente
tomó el atajo más simple (*shortcut learning*) y el test, generado con el mismo sesgo, no podía detectarlo.
Además, una CNN 1D sobre onda cruda es equivariante en tiempo pero **no** en frecuencia; el cromagrama
sí es invariante a la octava por construcción.

**Conclusión honesta:** con este diseño la red **no superó** a la línea de base. El factor limitante fue
el diseño de los datos, no la arquitectura.

### Cómo se corrige

1. Aumentación por **pitch-shift** actualizando la etiqueta → rompe la correlación espuria.
2. **Randomizar la octava** en `prog_key` → elimina el sesgo en el origen (2 líneas).
3. Migrar a **CQT / log-mel + CNN 2D** → sobre CQT la transposición es una traslación y la conv2D ya es
   equivariante: da por arquitectura la invariancia que hoy falta.
4. Evaluación sistemática sobre los **604 temas de GiantSteps**.
"""


def main():
    with open(NB, encoding="utf-8") as fh:
        nb = json.load(fh)

    nb["cells"] = [c for c in nb["cells"] if MARCA not in "".join(c["source"])]

    celda = {"cell_type": "markdown", "metadata": {},
             "source": MD_HALLAZGO.splitlines(keepends=True)}

    # insertar antes de la celda de cierre
    idx = len(nb["cells"])
    for i, c in enumerate(nb["cells"]):
        if "Cierre — para planilla y repo" in "".join(c["source"]):
            idx = i
            break
    nb["cells"].insert(idx, celda)

    with open(NB, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"Notebook actualizado: {len(nb['cells'])} celdas, hallazgos en posicion {idx}")


if __name__ == "__main__":
    main()
