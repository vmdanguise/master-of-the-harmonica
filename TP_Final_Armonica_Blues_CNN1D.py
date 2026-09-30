# TP Final - Master of the Harmonica - CNN 1D + Cross-Harp simple
# Generado desde TP_Final_Armonica_Blues_CNN1D_simpl.ipynb
# Materia: Redes Neuronales y Deep Learning - Maestria IA - Univ. Palermo

# ============================================================
# # 🎸 TP Final — Clasificación de Notas Musicales (1D) + As de la Armónica Blues
#
# **Materia:** Redes Neuronales y Deep Learning · Maestría en IA · Universidad de Palermo
#
# **Enfoque:** CNN 1D en PyTorch sobre audio crudo (waveform 16 kHz, ventanas de 2 s) + dataset sintético + lógica Cross-Harp (2ª posición).
#
# **Cómo usar en Colab:** `Archivo → Subir cuaderno` → `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)` → `Entorno de ejecución → Reiniciar y ejecutar todo`.
#
# **Estructura (10 pasos exigidos por la consigna):**
#
# | Paso | Contenido |
# |---|---|
# | 1 | Teoría CNN 1D |
# | 2 | Dependencias e importaciones |
# | 3 | Dataset sintético + Dataset/DataLoader |
# | 4 | Arquitectura `NoteClassifier1D` |
# | 5 | Training loop |
# | 6 | Curvas + matriz de confusión |
# | 7 | Lógica armónica blues (Cross-Harp) |
# | 8 | Carga de audio real (WAV/MP3) |
# | 9 | Inferencia |
# | 10 | Salida del "As de la Armónica" |
#
# > Criterios de cátedra aplicados: semillas fijas, celda de entorno (Parte C de la guía), ejecución lineal de arriba abajo, salidas visibles, sin descargas externas pesadas.
# ============================================================

# --- Celda 1 ---
# ============================================================
# Celda 0 — Entorno + reproducibilidad (Guía, Parte C y E)
# ============================================================
import sys, platform, random
import numpy as np

print("Python", sys.version.split()[0], "|", platform.system())
for nombre in ["numpy", "matplotlib", "pandas", "sklearn", "torch", "torchaudio", "librosa", "scipy", "seaborn"]:
    try:
        mod = __import__(nombre)
        print(f"{nombre:12s} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"{nombre:12s} NO INSTALADO")

def fijar_semillas(seed=42):
    """Fija semillas de random/numpy/torch (criterio de corrección de la materia)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

fijar_semillas(42)
print("\nSemilla fijada en 42 ✔")
try:
    import torch
    if torch.cuda.is_available():
        print("GPU NVIDIA:", torch.cuda.get_device_name(0))
    else:
        print("Sin GPU — se corre en CPU (alcanza para este TP)")
except ImportError:
    pass

# ============================================================
# ## PASO 1 — Explicación didáctica corta (CNN 1D, kernel, stride, pooling)
#
# **¿Qué es una CNN 1D?** Una red convolucional que opera sobre secuencias (eje tiempo `T`). Entrada: forma `(batch, canales=1, T=32000)`. Cada capa aprende *filtros* que detectan patrones locales (ataques, periodicidades).
#
# **Kernel 1D como filtro de frecuencia adaptativo.** Un kernel de tamaño $K$ es un vector de $K$ pesos que se desliza sobre la señal. La salida en $t$ es $y[t]=\sum_{k=0}^{K-1} w[k]\cdot x[t+k]$. Con $K$ grande (64–128 a 16 kHz) el filtro "ve" varios milisegundos → puede resonar con frecuencias bajas (ej. $f_0 \approx 261$–$494$ Hz, períodos de $2$–$4$ ms). El entrenamiento ajusta $w$ por backprop: el kernel *aprende* a ser un detector de periodicidad, sin FFT manual.
#
# **Convolución sobre el tiempo.** El mismo kernel se aplica en cada posición (pesos compartidos) → equivariancia temporal: una nota suena igual aunque empiece 200 ms después.
#
# **Stride.** Paso del deslizamiento. `stride=8` submuestrea ×8 la salida: menos cómputo y más invariancia, a costa de resolución temporal fina.
#
# **Pooling.** `MaxPool1d(4)` conserva el máximo local cada 4 muestras → invariancia a pequeños desplazamientos de fase y reducción ×4 de longitud. `AdaptiveAvgPool1d(1)` al final colapsa todo el tiempo en un vector fijo (promedio global) → el clasificador lineal ve un embedding de 128 dims invariante al *dónde* ocurrió la nota.
#
# **Pipeline del TP:** onda cruda 2 s @16 kHz → Conv ancha (captura $f_0$) → bloques Conv+BN+ReLU+Pool (jerarquía: periodo → timbre → nota) → promedio global → lineal 12 clases.
# ============================================================

# ============================================================
# ## PASO 2 — Instalación de dependencias e importaciones
# ============================================================

# --- Celda 4 ---
# En Colab la mayoría ya viene; esto garantiza el entorno gratuito sin romper nada.
# OMITIDO (solo Colab/Jupyter): !pip install -q librosa seaborn scikit-learn scipy 2>&1 | tail -n 2

import os, math, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio  # exigido por consigna (se usa para verificar sample rate)
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# OMITIDO (solo Colab/Jupyter): %matplotlib inline
sns.set(style="whitegrid")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE, "| torch:", torch.__version__)
print("librosa:", librosa.__version__, "| torchaudio:", torchaudio.__version__)

# ============================================================
# ## PASO 3 — Generación del dataset sintético de notas 1D
#
# **Idea:** 2 s @16 kHz → $T=32000$ muestras. Cada muestra: $x(t)=\sum_{h=1}^{3} A_h\sin(2\pi h f_0 t+\phi_h)\cdot env(t) + ruido$.
# Fundamental $f_0$ = nota cromática (octava 4), 2–3 armónicos con amplitud decreciente, fases aleatorias, envolvente ADSR simple, ruido blanco (SNR 10–25 dB) y micro-desafinación ±0.5 % → simula instrumentos reales sin descargar nada.
#
# **Memoria:** el `Dataset` genera la onda *on-the-fly* (no guarda 2000×32000 floats en RAM). Train = aleatorio cada época (aumentación natural); val/test = determinístico por índice.
# ============================================================

# --- Celda 6 ---
SR = 16000        # sample rate exigido
DUR = 2.0         # segundos por ventana
N_SAMPLES = int(SR * DUR)  # 32000

# 12 notas cromáticas, octava 4 (Hz, temperamento igual, A4=440)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FREQS = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}
NOTE_TO_IDX = {n: i for i, n in enumerate(NOTE_NAMES)}

def sintetizar_nota(freq_hz, sr=SR, dur=DUR, rng=None, snr_db=18.0):
    """Genera 2 s de tono con 3 armónicos + envolvente + ruido blanco. Devuelve float32 mono en [-1, 1]."""
    if rng is None:
        rng = np.random
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    # micro-desafinación ±0.5% (instrumento real)
    f0 = freq_hz * (1.0 + rng.uniform(-0.005, 0.005))
    # amplitudes: fundamental dominante + 2 armónicos que decaen
    amps = np.array([1.0, rng.uniform(0.25, 0.5), rng.uniform(0.1, 0.3)], dtype=np.float32)
    amps = amps / amps.sum()
    phases = rng.uniform(0, 2 * np.pi, size=3).astype(np.float32)
    x = np.zeros(n, dtype=np.float32)
    for h, (a, p) in enumerate(zip(amps, phases), start=1):
        x += a * np.sin(2 * np.pi * f0 * h * t + p).astype(np.float32)
    # envolvente ADSR simple (ataque 10 ms, release 200 ms)
    attack = int(0.01 * sr); release = int(0.20 * sr)
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    x *= env
    # ruido blanco según SNR deseada (10–25 dB aleatorio si no se fija)
    if snr_db is None:
        snr_db = float(rng.uniform(10, 25))
    sig_pow = np.mean(x ** 2) + 1e-9
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    x += rng.normal(0, math.sqrt(noise_pow), size=n).astype(np.float32)
    # normalización por pico a [-0.95, 0.95]
    peak = np.max(np.abs(x)) + 1e-9
    return (0.95 * x / peak).astype(np.float32)

class SyntheticNotesDataset(Dataset):
    """Dataset 1D: genera ondas on-the-fly. `split='train'` aleatorio; val/test determinísticos."""
    def __init__(self, labels, split="train", snr_db=None):
        self.labels = list(labels)
        self.split = split
        self.snr_db = snr_db
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        y = int(self.labels[idx])
        freq = NOTE_FREQS[NOTE_NAMES[y]]
        rng = np.random if self.split == "train" else np.random.RandomState(10_000 + idx)
        snr = None if (self.split == "train" and self.snr_db is None) else (self.snr_db or 18.0)
        if self.split == "train" and self.snr_db is None:
            snr = None  # SNR aleatoria 10–25 dB como aumentación
        wav = sintetizar_nota(freq, rng=rng, snr_db=snr)
        return torch.from_numpy(wav).unsqueeze(0), torch.tensor(y, dtype=torch.long)  # (1, 32000)

# --- Partición estratificada de índices ---
N_TRAIN_CLS, N_VAL_CLS, N_TEST_CLS = 150, 30, 30
all_labels = ([i for i in range(12) for _ in range(N_TRAIN_CLS)],
                [i for i in range(12) for _ in range(N_VAL_CLS)],
                [i for i in range(12) for _ in range(N_TEST_CLS)])
train_ds = SyntheticNotesDataset(all_labels[0], split="train")
val_ds   = SyntheticNotesDataset(all_labels[1], split="val")
test_ds  = SyntheticNotesDataset(all_labels[2], split="test")
BATCH = 32
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | batch={BATCH}")

# --- Sanity check visual: onda + espectro de un ejemplo ---
wav0, y0 = val_ds[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(wav0.numpy()[0, :2000]); ax[0].set_title(f"Onda 2 s (zoom 2000 pts) — {NOTE_NAMES[y0.item()]}")
ax[0].set_xlabel("muestra"); ax[0].set_ylabel("amplitud")
spec = np.abs(np.fft.rfft(wav0.numpy()[0]))
freqs = np.fft.rfftfreq(N_SAMPLES, 1 / SR)
ax[1].plot(freqs[:4000], spec[:4000]); ax[1].set_title("Espectro (picos en f0 + armónicos)")
ax[1].set_xlabel("Hz"); plt.tight_layout(); plt.show()

# ============================================================
# ## PASO 4 — Arquitectura CNN 1D (`NoteClassifier1D`)
#
# **Diseño:** kernel inicial ancho (128, stride 8) → cubre ~8 ms: varios períodos de $f_0$ (2–4 ms) → el filtro puede aprender la periodicidad grave. Luego kernels chicos (16→7→5) para timbre/ataque. `BatchNorm` estabiliza, `MaxPool` da invariancia de fase, `AdaptiveAvgPool1d(1)` fija el embedding a 128 dims.
#
# **Hiperparámetros:** `lr=1e-3` (Adam: seguro para este tamaño), `batch=32` (entra en CPU/T4 gratis), `Adam` (momento adaptativo, ideal con ruido), `CrossEntropyLoss` (12 clases mutuamente excluyentes). ~100 k parámetros → entrena en minutos en CPU.
# ============================================================

# --- Celda 8 ---
class NoteClassifier1D(nn.Module):
    """CNN 1D: (1, 32000) -> 12 notas. Kernel ancho inicial + bloques Conv-BN-ReLU-Pool."""
    def __init__(self, n_classes=12, dropout=0.25):
        super().__init__()
        # Bloque 1: kernel ancho capta f0 graves (128 pts ≈ 8 ms @16kHz), stride 8 comprime ×8
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=128, stride=8, padding=64),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
        )
        # Bloques 2-4: kernels chicos para armónicos/ataques, canales crecientes
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, padding=8),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B,128,T) -> (B,128,1): invariante temporal
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.gap(x)
        return self.head(x)

model = NoteClassifier1D().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\nParámetros entrenables: {n_params:,}")
# Verificación de formas con un batch dummy
with torch.no_grad():
    dummy = torch.randn(2, 1, N_SAMPLES).to(DEVICE)
    print("in:", tuple(dummy.shape), "-> out:", tuple(model(dummy).shape))

# ============================================================
# ## PASO 5 — Bucle de entrenamiento (10–15 épocas, CrossEntropy + Adam)
# ============================================================

# --- Celda 10 ---
EPOCHS = 12
LR = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(loader, train=True):
    model.train(train)
    tot_loss, tot_ok, tot_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward(); optimizer.step()
        tot_loss += loss.item() * len(xb)
        tot_ok += (logits.argmax(1) == yb).sum().item()
        tot_n += len(xb)
    return tot_loss / tot_n, tot_ok / tot_n

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc, best_state = 0.0, None
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"Época {ep:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.3f} || val loss {va_loss:.4f} acc {va_acc:.3f}")

model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
print(f"\n✔ Mejor val_acc={best_acc:.3f} — pesos restaurados para evaluar/inferir.")

# ============================================================
# ## PASO 6 — Evaluación: curvas + matriz de confusión 12×12
# ============================================================

# --- Celda 12 ---
# --- Curvas Loss / Accuracy ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
ax[0].set_title("Loss por época"); ax[0].set_xlabel("época"); ax[0].legend()
ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
ax[1].set_title("Accuracy por época"); ax[1].set_xlabel("época"); ax[1].legend()
plt.tight_layout(); plt.show()

# --- Matriz de confusión en TEST ---
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred += model(xb.to(DEVICE)).argmax(1).cpu().tolist()
        all_true += yb.tolist()
print(f"Test accuracy: {accuracy_score(all_true, all_pred):.3f}")
cm = confusion_matrix(all_true, all_pred, labels=list(range(12)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=NOTE_NAMES, yticklabels=NOTE_NAMES)
plt.title("Matriz de confusión 12×12 (test)"); plt.xlabel("predicha"); plt.ylabel("real")
plt.tight_layout(); plt.show()

# ============================================================
# ## PASO 7 - Logica armonica blues (Cross-Harp) en palabras simples
#
# No necesitas saber musica ni tecnica de armonica para usar esto.
#
# La armonica tiene 10 agujeritos arriba, numerados del 1 al 10. Solo vas a hacer 3 movimientos:
#
# 1. **Soplar:** echar aire por el agujerito (como soplar una vela despacio).
# 2. **Aspirar:** tomar aire por el agujerito (como tomar con bombilla).
# 3. **Bending:** es aspirar pero frenando un poco el aire con la lengua para atras, como cuando decis iii por dentro. La nota suena un poco mas grave y llorada, que es el sonido tipico del blues. No hay que estudiar nada mas: si te sale el lloradito, ya esta.
#
# **Que armonica agarro? Regla de oro:**
#
# El modelo te dice en que nota esta la cancion (ej: G). Vos no tocas en esa misma nota, agarras otra armonica que combina mejor para blues. No tenes que calcularlo, el programa ya trae la tabla:
#
# - Cancion en G -> agarra armonica en C
# - Cancion en A -> agarra armonica en D
# - Cancion en E -> agarra armonica en A
# - (y asi con las 12 notas, la tabla completa esta en el codigo)
#
# **Que toco?**
#
# El programa te da un riff: una listita de 6-7 pasos como celda 2 aspirando, celda 3 aspirando con bending, etc. La tocas en orden, sobre la cancion, y suena a blues. Siempre se empieza y se termina en la Celda 2 aspirando, que es tu casa.
# ============================================================

# --- Celda 14 ---
# PASO 7 - Que armonica agarrar y que tocar (explicado simple)
# Solo hay 3 acciones:
#   SOPLAR la celda N (echar aire)
#   ASPIRAR la celda N (tomar aire como con bombilla)
#   BENDING = aspirar frenando un poco el aire con la lengua para atras,
#   suena mas grave / lloradito, es el sonido blues.
# Ejemplo: celda 3 aspirando con bending.
# La tabla de abajo ya resuelve que armonica agarrar segun la cancion.
# Detalle tecnico para la catedra: es 2da posicion / Cross-Harp.

# Cancion -> armonica que hay que agarrar (ya calculado, no hay que pensar)
CROSS_HARP = {
    "C": "F", "C#": "F#", "D": "G", "D#": "G#", "E": "A", "F": "A#",
    "F#": "B", "G": "C", "G#": "C#", "A": "D", "A#": "D#", "B": "E",
}

# Riffs en idioma simple: cada paso dice celda + que hacer con el aire.
RIFFS_2DA = {
    "C": ["celda 2 aspirando", "celda 3 aspirando con bending", "celda 4 aspirando",
          "celda 4 soplando", "celda 4 aspirando", "celda 3 aspirando con bending", "celda 2 aspirando"],
    "G": ["celda 2 aspirando", "celda 3 aspirando con bending", "celda 4 aspirando",
          "celda 4 aspirando con bending suave", "celda 3 aspirando con bending", "celda 2 aspirando", "celda 2 soplando"],
    "A": ["celda 3 soplando", "celda 3 aspirando con bending", "celda 4 aspirando",
          "celda 5 soplando", "celda 4 soplando", "celda 4 aspirando", "celda 3 aspirando con bending"],
    "D": ["celda 1 aspirando", "celda 2 aspirando", "celda 3 aspirando con bending",
          "celda 4 aspirando", "celda 4 soplando", "celda 3 aspirando con bending", "celda 2 aspirando"],
    "E": ["celda 2 aspirando", "celda 2 aspirando con bending fuerte", "celda 3 aspirando con bending",
          "celda 4 aspirando", "celda 5 aspirando", "celda 4 aspirando", "celda 3 aspirando con bending"],
    "default": ["celda 2 aspirando (tu casa, empeza aca)",
                "celda 3 aspirando con bending (el lloradito blues)",
                "celda 4 aspirando",
                "celda 4 soplando",
                "celda 4 aspirando",
                "celda 3 aspirando con bending",
                "celda 2 aspirando (termina aca, resuelve siempre)"],
}

def recomendar_armonica(nota_cancion):
    # Recibe la nota de la cancion (ej. G) y devuelve que armonica agarrar y que tocar.
    nota = nota_cancion.strip().upper().replace("DB", "C#")
    assert nota in NOTE_NAMES, f"Nota invalida: {nota}"
    harm = CROSS_HARP[nota]
    riff = RIFFS_2DA.get(harm, RIFFS_2DA["default"])
    expl = f"La cancion esta en {nota}, asi que agarra la armonica en {harm}. Ya esta probado que esa combinacion suena bien para blues."
    return harm, riff, expl

# Demo rapida
for demo in ["G", "A", "E"]:
    h, r, e = recomendar_armonica(demo)
    print(f"Cancion en {demo} -> agarra armonica en {h}")
    print("  Tocar: " + " / ".join(r))
    print("")

# ============================================================
# ## PASO 8 — Carga y preprocesamiento de audio real (WAV/MP3)
#
# Al correr esta celda Colab abre el **botón de subida**: elegí tu MP3/WAV y el resto es automático (mono → 16 kHz → ventanas de 2 s). Si no subís nada, se usa la **demo incluida**: un 12-bar blues en A con bajo + acordes + lead (mucho más digno que un tono puro).
# ============================================================

# --- Celda 16 ---
def cargar_audio_2s(path, sr_objetivo=SR, dur=DUR):
    """Carga WAV/MP3 → mono @16kHz → lista de ventanas float32 de 2 s normalizadas."""
    y, sr = librosa.load(path, sr=sr_objetivo, mono=True)  # resample automático (MP3 OK en Colab)
    n = int(sr_objetivo * dur)
    ventanas = []
    for i in range(0, max(len(y), 1), n):
        seg = y[i:i + n]
        if len(seg) < n:  # pad con ceros si la cola es corta
            seg = np.pad(seg, (0, n - len(seg)))
        peak = np.max(np.abs(seg)) + 1e-9
        ventanas.append((0.95 * seg / peak).astype(np.float32))
        if i + n >= len(y):
            break
    print(f"{os.path.basename(path)}: {len(y)/sr_objetivo:.1f}s @ {sr_objetivo} Hz → {len(ventanas)} ventana(s) de 2 s")
    return ventanas

# ---------- Demo digna: 12-bar blues en A (bajo + séptimas + lead) ----------
def _f(midi):
    """MIDI -> Hz (A4=440)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def _nota(freq, dur, sr=SR, rng=None, vibrato=(0.0, 5.5), bright=0.35, gain=0.5):
    """Nota con 4 armónicos, vibrato leve y envolvente. Devuelve float32."""
    if rng is None:
        rng = np.random
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    depth, rate = vibrato
    vib = depth * np.sin(2 * np.pi * rate * t).astype(np.float32) if depth > 0 else 0.0
    x = np.zeros(n, dtype=np.float32)
    for h, a in enumerate([1.0, 0.45, 0.22, 0.1], start=1):
        ph = rng.uniform(0, 2 * np.pi)
        x += (a * np.sin(2 * np.pi * freq * h * t + h * vib + ph)).astype(np.float32)
    x *= bright  # timbre menos nasal que tono puro
    a_n = min(int(0.02 * sr), n // 4); r_n = min(int(0.08 * sr), n // 4)
    env = np.ones(n, dtype=np.float32)
    env[:a_n] = np.linspace(0, 1, a_n); env[-r_n:] = np.linspace(1, 0, r_n)
    return (gain * x * env).astype(np.float32)

def demo_blues_12bar(key_midi=45, sr=SR, seed=7):
    """12 compases (2 s c/u) de blues en A: walking bass + acordes 7 + lead pentatónica.
    Progresión: A7 A7 D7 A7 | E7 D7 A7 E7 | A7 D7 A7 E7 (versión compacta de 12)."""
    rng = np.random.RandomState(seed)
    BAR = 2.0
    # Raíces por compás (MIDI): A2=45, D3=50, E3=52
    roots = [45, 45, 50, 45, 52, 50, 45, 52, 45, 50, 45, 52]
    out = np.zeros(int(sr * BAR * 12), dtype=np.float32)
    penta_A = [57, 60, 62, 64, 67, 69, 72]  # A menor pentatónica (A C D E G...)
    for b, root in enumerate(roots):
        base = b * int(sr * BAR)
        # Walking bass: negra = 0.5 s (tónica, 5ª, 6ª, b7)
        for q, iv in enumerate([0, 7, 9, 10]):
            n = _nota(_f(root + iv - 12), 0.48, sr, rng, vibrato=(0.0, 5.5), bright=0.6, gain=0.55)
            s = base + q * int(sr * 0.5)
            out[s:s + len(n)] += n
        # Acorde 7 en off-beat (corcheas 1-y-3): tríada + 7ma, corto y percusivo
        for off in [0.25, 0.75, 1.25, 1.75]:
            for iv in [0, 4, 7, 10]:  # 7ma dominante
                n = _nota(_f(root + 12 + iv), 0.22, sr, rng, vibrato=(0.0, 5.5), bright=0.3, gain=0.20)
                s = base + int(sr * off)
                out[s:s + len(n)] += n
        # Lead pentatónica solo en los últimos 4 compases (pregunta-respuesta)
        if b >= 8:
            seq = rng.choice(penta_A, size=4)
            for j, m in enumerate(seq):
                n = _nota(_f(m), 0.42, sr, rng, vibrato=(0.6, 5.5), bright=0.5, gain=0.40)
                s = base + j * int(sr * 0.5)
                out[s:s + len(n)] += n
    # Eco simple (room) + saturación suave + normalización
    echo = np.zeros_like(out); d = int(sr * 0.28)
    echo[d:] = 0.25 * out[:-d]
    out = np.tanh(out + echo)
    out = (0.9 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)
    return out

os.makedirs("/tmp", exist_ok=True)
DEMO_WAV = "/tmp/demo_blues_A.wav"
demo_tema = demo_blues_12bar()
wavfile.write(DEMO_WAV, SR, (demo_tema * 32767).astype(np.int16))
print("Demo 12-bar blues en A guardada:", DEMO_WAV, f"({len(demo_tema)/SR:.0f} s)")
try:
    from IPython.display import Audio as _Audio
    display(_Audio(demo_tema, rate=SR))
except Exception:
    pass

# ---------- Subida de tu MP3/WAV (activa: abre el botón en Colab) ----------
MI_AUDIO = DEMO_WAV  # valor por defecto si no subís nada
try:
    from google.colab import files as _files
    print("📤 Elegí tu archivo MP3/WAV en el botón (o Cancelar para usar la demo):")
    _up = _files.upload()
    if _up:
        _name = sorted(_up.keys())[0]
        with open(f"/content/{_name}", "wb") as _fh:
            _fh.write(_up[_name])
        MI_AUDIO = f"/content/{_name}"
        print(f"✔ Audio recibido: {MI_AUDIO}")
    else:
        print("Sin subida → uso la demo:", DEMO_WAV)
except ImportError:
    print("No estás en Colab (corrida local) → uso la demo:", DEMO_WAV)

# ============================================================
# ## PASO 9 — Inferencia con el modelo 1D (voto mayoritario por ventanas)
# ============================================================

# --- Celda 18 ---
def predecir_nota_archivo(path, model=model, device=DEVICE, max_ventanas=12):
    """Predice la nota dominante: promedia probabilidades de las ventanas y vota."""
    model.eval()
    ventanas = cargar_audio_2s(path)
    if len(ventanas) > max_ventanas:  # audios largos: muestreo uniforme, no solo el inicio
        idx = np.linspace(0, len(ventanas) - 1, max_ventanas).round().astype(int)
        ventanas = [ventanas[i] for i in idx]
        print(f"(audio largo: voto con {max_ventanas} ventanas representativas)")
    xb = torch.stack([torch.from_numpy(w).unsqueeze(0) for w in ventanas]).to(device)
    with torch.no_grad():
        logits = model(xb)                      # (n_vent, 12)
        probs = torch.softmax(logits, dim=1)
        voto = probs.mean(dim=0)                # promedio de probabilidades
        idx = int(voto.argmax())
    top3 = sorted(range(12), key=lambda i: float(voto[i]), reverse=True)[:3]
    print("Top-3:", {NOTE_NAMES[i]: round(float(voto[i]), 3) for i in top3})
    return NOTE_NAMES[idx], float(voto[idx]), [NOTE_NAMES[int(p)] for p in probs.argmax(1)]

nota_pred, conf, por_ventana = predecir_nota_archivo(MI_AUDIO)
print(f"\n🎯 Nota predicha: {nota_pred} (confianza {conf:.2f}) | por ventana: {por_ventana}")

# ============================================================
# ## PASO 10 — Salida del "As de la Armónica" 🎩
# ============================================================

# --- Celda 20 ---
def mostrar_as_armonica(nota_cancion, confianza=None):
    harm, riff, expl = recomendar_armonica(nota_cancion)
    print("=" * 52)
    print("   EL AS DE LA ARMONICA - tu rutina blues")
    print("=" * 52)
    print(f"Cancion en           : {nota_cancion}")
    if confianza is not None:
        print(f"Seguridad del modelo : {confianza:.2f}")
    print(f"Agarra del estuche   : ARMONICA EN {harm}")
    print(f"Por que esa          : {expl}")
    print("-" * 52)
    print("Riff para improvisar (tocar en orden, sobre la pista):")
    print("  Soplar = echar aire | Aspirar = tomar aire")
    print("  Bending = aspirar frenando un poquito el aire para que suene lloradito.")
    for i, celda in enumerate(riff, 1):
        print(f"   {i}. {celda}")
    print("-" * 52)
    print("Tip: empeza y termina siempre en celda 2 aspirando.")
    print("Si te perdes, volve a esa celda y marca el pulso ahi.")
    print("=" * 52)

mostrar_as_armonica(nota_pred, conf)

# ============================================================
# ### Cierre — qué entregar y cómo reproducir
#
# 1. En Colab: `Reiniciar y ejecutar todo`, verificá curvas + matriz + As de la Armónica.
# 2. Descargá el `.ipynb` **con salidas visibles** y subilo al repo del grupo (ver `trabajo practico.txt`: README con librerías, semillas y link/licencia del dataset — aquí sintético propio).
# 3. Límites conocidos: tonos puros ≠ instrumentos reales; la micro-desafinación y el ruido ayudan pero un audio polifónico puede confundir al voto mayoritario. Propuesta futura: espectrograma log-mel + CNN 2D (idea ejemplo de la planilla: fallas en rodamientos) o fine-tuning desde YAMNet.
# ============================================================
