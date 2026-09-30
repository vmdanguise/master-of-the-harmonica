# Master of the Harmonica

TP Final — Clasificación de Notas Musicales (1D) + As de la Armónica Blues.

Materia: Redes Neuronales y Deep Learning · Maestría en IA · Universidad de Palermo.

## Contenido
- `TP_Final_Armonica_Blues_CNN1D.ipynb` — notebook principal (10 pasos, listo para Colab).
- `TP_Final_Armonica_Blues_CNN1D.py` — mismo código en script plano.
- `TP_final_idea.txt` — prompt/idea original del trabajo.
- `trabajo-practico-consigna.txt` — consigna de la materia.

## Cómo reproducir (Colab gratuito)
1. Subir el `.ipynb` a Google Colab.
2. Entorno: GPU T4 (opcional, anda en CPU).
3. `Entorno de ejecución → Reiniciar y ejecutar todo`.

## Entorno y reproducibilidad
- Python 3 + torch, torchaudio, numpy, matplotlib, seaborn, librosa, scipy, scikit-learn.
- Semillas fijas en 42 (random/numpy/torch, cudnn determinístico).
- Dataset sintético propio on-the-fly: 2 s @16 kHz, 12 notas cromáticas (C–B), 3 armónicos + ruido. Sin descargas externas.
- Ventanas de inferencia de 2 s con voto mayoritario.

## Lógica Cross-Harp (simple)
Sin teoría musical: soplar (echar aire), aspirar (tomar aire), bending (aspirar frenando el aire, sonido lloradito).
El modelo predice la nota de la canción y la tabla `CROSS_HARP` indica qué armónica agarrar + riff de celdas.
