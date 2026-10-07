# Master of the Harmonica

TP Final — Clasificación multiclase con CNN 1D para evaluar la tonalidad de una canción y recomendar armónica + notas de acompañamiento.

Materia: Redes Neuronales y Deep Learning · Maestría en IA · Universidad de Palermo.

## Revisión de cátedra (aplicada)

> Falta definir qué entra y qué sale. Acotarlo: clasificar la tonalidad (24 clases) y derivar la armónica con una regla; en segunda posición es una cuarta justa arriba. Como los datos son sintéticos, sumar prueba con canciones reales de tonalidad conocida (GiantSteps Key). Línea de base sin redes: cromagrama + perfiles Krumhansl-Schmuckler.

- **Entra:** audio `WAV/MP3` → mono `16 kHz` → ventanas de 4 s normalizadas. No entra MIDI ni cromagrama a la red.
- **Sale:** tonalidad en **24 clases (12 tónicas x mayor/menor)** con confianza y Top-3, armónica blues en **2da posición = 4ta justa arriba `(+5 semitonos)`** y riff de celdas (soplar / aspirar / bending).
- **Baseline:** cromagrama `librosa` promediado + correlación de Pearson contra los 24 perfiles rotados de Krumhansl-Kessler. Si la CNN no lo supera, no agregó valor.

## Contenido

- `TP_Final_Revisado_Profe_24Keys.ipynb` — versión revisada (10 pasos, lista para Colab gratuito, CPU ok).
- `TP_Final_Revisado_Profe_24Keys.py` — mismo código en script plano (comentarios espejo del notebook).
- `audios_prueba_final/` — pruebas de tonalidad conocida versionadas (16 kHz mono) + `MANIFEST_keys.csv`:
  `test_C_maj`, `test_G_maj`, `test_E_maj` (I-IV-V-I), `test_A_min` (i-iv-V-i) y `demo_blues_A_12bar_key-A-maj.wav` (24 s, verdad `A maj` → armónica en `D`).
- `TP_Final_Armonica_Blues_CNN1D.ipynb` / `.py` — primera versión (12 notas, ventanas 2 s, se conserva como antecedente).
- `TP_final_idea.txt`, `trabajo-practico-consigna.txt` — idea original y consigna.

## Dataset

- **Sintético propio (entrenamiento):** progresiones `I-IV-V-I` / `i-iv-V-i` con `V mayor`, 4 s @16 kHz, 3 armónicos + envolvente + micro-desafinación + ruido SNR 12–25 dB, generación on-the-fly. `train 1920 (80/clase) / val 480 / test 480`, `val/test` determinísticos. Sin descargas.
- **Real (evaluación, no entrena):** `audios_prueba_final/` + descarga automática de 20 temas al azar de [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) (604 previews de 2 min EDM Beatport en 24 keys, Knees et al. ISMIR 2015; mirror JKU con verificación md5, `seed 7`, solo evaluación, ver Pasos 8 y 9b). Si no hay internet, el notebook sigue con demo + upload. Los MP3 de GiantSteps no se versionan.

## Cómo reproducir (Colab gratuito)

1. Subir `TP_Final_Revisado_Profe_24Keys.ipynb` a Google Colab.
2. Entorno GPU T4 (opcional, anda en CPU) → `Reiniciar y ejecutar todo`.
3. Verifica curvas + matriz 24×24 + score MIREX + comparativa `CNN vs K-S` + `As de la Armónica`.

## Entorno y reproducibilidad

- Python 3 + torch, torchaudio, numpy, matplotlib, seaborn, librosa, scipy, scikit-learn.
- Semilla fija `42` (random/numpy/torch, cudnn determinístico).
- Métricas: accuracy 24 clases + score estilo MIREX (correcta 1, quinta 0.5, relativa/paralela 0.3).

## Lógica Cross-Harp (simple)

Sin teoría: soplar (echar aire), aspirar (tomar aire), bending (aspirar frenando el aire, lloradito blues). La tabla sale de la fórmula, no se memoriza. El riff detallado está para `C`; el resto usa el riff genérico de 2da posición (celdas 2–4, casa en celda 2 aspirando).
