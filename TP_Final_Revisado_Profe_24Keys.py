# TP Final Revisado (profe) - Tonalidad 24 clases + CNN 1D + Baseline K-S + Armonica Blues
# Materia: Redes Neuronales y Deep Learning - Maestria IA - Univ. Palermo
# Revision de catedra: entrada audio WAV/MP3 mono 16kHz ventanas 4s; salida 24 clases
#   (12 tonicas x mayor/menor) + armonica 2da posicion (+5 semitonos, 4ta justa) + riff.
# Baseline sin redes: cromagrama + Krumhansl-Schmuckler. Prueba real: demo en A + 20 GiantSteps auto.
# Generado desde TP_Final_Revisado_Profe_24Keys.ipynb - ejecutar de arriba abajo en Colab (CPU ok).

# ==========================================================
# # TP Final Revisado (profe) — Tonalidad 24 clases + CNN 1D + Baseline K-S + Armónica Blues
#
# **Materia:** Redes Neuronales y Deep Learning · Maestría en IA · Universidad de Palermo
#
# **Revisión pedida por cátedra:**
# 1. **Entrada definida:** audio `WAV/MP3 -> mono 16 kHz -> ventanas de 4 s`. No entra MIDI ni cromagrama a la red.
# 2. **Salida definida:** tonalidad en **24 clases (12 tónicas x mayor/menor)** + derivación determinística de armónica en **2da posición = 4ta justa arriba (+5 semitonos)**.
# 3. **Datos sintéticos + prueba real:** entrena en sintético (progresiones mayor/menor) y evalúa en canciones reales de tonalidad conocida (**GiantSteps Key**, 604 previews 2 min, 24 keys, Knees et al. ISMIR 2015).
# 4. **Baseline sin redes:** cromagrama `librosa` correlacionado con perfiles **Krumhansl-Schmuckler** por Pearson.
#
# **Uso en Colab:** `Subir cuaderno` → `GPU T4` → `Reiniciar y ejecutar todo`. Corre en CPU free.
#
# ==========================================================

# Celda 0 — Entorno + reproducibilidad
import sys, platform, random  # importa datos del sistema y azar base
import numpy as np  # importa calculo con vectores y azar
print("Python", sys.version.split()[0], "|", platform.system())  # muestra version de Python y sistema operativo
for n in ["numpy","matplotlib","pandas","sklearn","torch","torchaudio","librosa","scipy","seaborn"]:  # recorre las librerias a verificar
    try:  # intenta importar la libreria de turno
        m=__import__(n); print(f"{n:12s} {getattr(m,'__version__','ok')}")  # importa por nombre y muestra su version
    except ImportError: print(f"{n:12s} NO INSTALADO")  # si falta avisa y sigue con la proxima
def fijar_semillas(seed=42):  # define como fijar semillas para repetir resultados
    random.seed(seed); np.random.seed(seed)  # fija el azar de Python y numpy
    try:  # intenta configurar las semillas de torch
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # fija el azar de torch en CPU y GPU
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False  # fuerza calculo deterministico en GPU
    except ImportError: pass  # si falta el modulo sigue sin cortar
fijar_semillas(42); print("\nSemilla 42 OK")  # aplica semilla 42 y lo confirma
try:  # intenta detectar la GPU disponible
    import torch  # importa torch para consultar el hardware
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (alcanza)")  # muestra GPU o avisa que alcanza con CPU
except ImportError: pass  # si falta el modulo sigue sin cortar


# ==========================================================
# ## PASO 1 — Teoría acotada (24 keys, 4ta justa, CNN 1D vs K-S)
#
# **Tonalidad (24 clases):** 12 tónicas x {mayor, menor}. No es la nota aislada: es el centro tonal + modo. Ej `A mayor` vs `A menor` comparten tónica pero distinta 3ra (C# vs C).
#
# **Regla 2da posición (Cross-Harp):** `harm = (tonica_cancion + 5) mod 12`. Una 4ta justa arriba. Ej `G(7)->(7+5)%12=0->C`. Vale para mayor y menor (depende solo de tónica).
#
# **CNN 1D:** kernel ancho 128 (~8 ms @16kHz) aprende periodicidad sin FFT; `stride/pool` dan invariancia temporal; `AdaptiveAvgPool1d(1)` colapsa tiempo. Entrada `(1, 64000)` 4 s → 24 logits.
#
# **Baseline K-S (sin redes):** cromagrama de 12 bins promediado en tiempo `c` se correlaciona (Pearson) con 24 perfiles rotados: `MAJ=[6.35,2.23,...]`, `MIN=[6.33,2.68,...]` (Krumhansl-Kessler). Gana el key con mayor `r`. Si la CNN no supera esto, no agregó valor.
#
# ==========================================================

# ==========================================================
# ## PASO 2 — Dependencias
# ==========================================================

# instala faltantes en Colab en silencio
# OMITIDO (solo Colab/Jupyter): pip install -q librosa seaborn scikit-learn scipy 2>&1 | tail -n 2
import os, math, random  # importa sistema, matematica y azar
import numpy as np, matplotlib.pyplot as plt, seaborn as sns  # importa numeros, graficos y estilo de graficos
from scipy.io import wavfile  # importa lectura y escritura de WAV
import torch, torch.nn as nn  # importa torch y sus capas de red
from torch.utils.data import Dataset, DataLoader  # importa base de dataset y cargador por lotes
import torchaudio, librosa  # importa audio de torch y analisis de audio
from sklearn.metrics import confusion_matrix, accuracy_score  # importa matriz de confusion y accuracy
# muestra los graficos dentro del notebook
# OMITIDO (solo Colab/Jupyter): %matplotlib inline
sns.set(style="whitegrid")  # usa estilo con grilla en los graficos
DEVICE="cuda" if torch.cuda.is_available() else "cpu"  # elige GPU si hay y si no CPU
print("DEVICE:",DEVICE,"| torch:",torch.__version__)  # muestra dispositivo y version de torch


# ==========================================================
# ## PASO 3 — Dataset sintético 24 clases (progresiones, no tonos puros)
#
# Cada sample = 4 s @16kHz = 64000. Contiene `I-IV-V-I` (mayor) o `i-iv-V-i` (menor, V mayor armónica). Cada acorde 1 s = triada/tetrada con 3 armónicos + envolvente + ruido + micro-desafinación. Generación on-the-fly (no guarda GB en RAM).
# ==========================================================

SR=16000; DUR=4.0; N_SAMPLES=int(SR*DUR)  # define 16kHz, 4 segundos y 64000 muestras
TONICS=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]  # lista las 12 tonicas con sostenidos
T2I={n:i for i,n in enumerate(TONICS)}  # diccionario de tonica a indice 0 a 11
BASE_F={"C":261.63,"C#":277.18,"D":293.66,"D#":311.13,"E":329.63,"F":349.23,"F#":369.99,"G":392.00,"G#":415.30,"A":440.00,"A#":466.16,"B":493.88}  # frecuencia base de cada tonica en octava 4
KEYS=[f"{t} maj" for t in TONICS]+[f"{t} min" for t in TONICS]  # arma las 24 clases, 12 mayor mas 12 menor
K2I={k:i for i,k in enumerate(KEYS)}  # diccionario de clase a indice 0 a 23
print("24 clases:",KEYS)  # muestra las 24 clases

def tono(freq,dur,sr,rng,gain=0.5):  # genera un tono con armonicos y envolvente
    n=int(sr*dur); t=np.arange(n,dtype=np.float32)/sr  # cantidad de muestras y eje de tiempo
    x=np.zeros(n,dtype=np.float32)  # vector de silencio donde se suma el sonido
    for h,a in enumerate([1.0,0.4,0.18],start=1):  # recorre fundamental y 2 armonicos con su amplitud
        x+=(a*np.sin(2*np.pi*freq*h*t+rng.uniform(0,2*np.pi))).astype(np.float32)  # suma el armonico h con fase al azar
    a_n=min(int(0.02*sr),n//4); r_n=min(int(0.08*sr),n//4)  # largo de ataque y decaimiento en muestras
    env=np.ones(n,dtype=np.float32); env[:a_n]=np.linspace(0,1,a_n); env[-r_n:]=np.linspace(1,0,r_n)  # envolvente que sube al inicio y baja al final
    return (gain*x*env).astype(np.float32)  # aplica volumen y devuelve el tono

def acorde(root_hz,ivs,dur,sr,rng):  # mezcla las notas de un acorde
    n=int(sr*dur); x=np.zeros(n,dtype=np.float32)  # muestras del acorde y vector vacio
    for iv in ivs:  # recorre los intervalos del acorde en semitonos
        f=root_hz*(2.0**(iv/12.0))*(1.0+rng.uniform(-0.004,0.004))  # frecuencia del intervalo con mini desafinacion
        x+=tono(f,dur,sr,rng,gain=0.33)  # suma el tono de esa nota al acorde
    return x  # devuelve el acorde mezclado

def prog_key(key_idx,rng,snr_db=None):  # sintetiza 4 segundos de progresion de una tonalidad
    tonica=TONICS[key_idx%12]; is_maj=key_idx<12  # tonica por resto 12 y modo por mitad 0 a 11 mayor
    rf=BASE_F[tonica]/2.0  # octava 3 para comping
    roots=[0,5,7,0]  # I-IV-V-I (mayor) / i-iv-V-i (menor, V mayor armonica)
    y=np.zeros(int(SR*DUR),dtype=np.float32); seg=int(SR*1.0)  # pista vacia de 4s y largo de 1 acorde
    for j,r in enumerate(roots):  # recorre los 4 acordes de la progresion
        r_hz=rf*(2.0**(r/12.0))  # fundamental del acorde segun su grado
        if is_maj: ivs=[0,4,7] if j in [0,1,3] else [0,4,7,10]  # triadas mayor y septima en el V
        else: ivs=[0,3,7] if j in [0,1,3] else [0,4,7,10]  # V mayor en menor
        ch=acorde(r_hz,ivs,1.0,sr=SR,rng=rng)  # sintetiza el acorde de 1 segundo
        y[j*seg:(j+1)*seg]+=ch[:seg]  # pega el acorde en su segundo de la pista
    if snr_db is None: snr_db=float(rng.uniform(12,25))  # elige nivel de ruido al azar si no se fijo
    p=np.mean(y**2)+1e-9; y+=rng.normal(0,math.sqrt(p/(10**(snr_db/10))),size=y.shape).astype(np.float32)  # agrega ruido blanco segun relacion senal ruido
    return (0.9*y/(np.max(np.abs(y))+1e-9)).astype(np.float32)  # normaliza por pico y devuelve la pista

class KeyDataset(Dataset):  # dataset PyTorch que genera audio al vuelo
    def __init__(self,labels,split="train"):  # guarda etiquetas y particion del dataset
        self.labels=list(labels); self.split=split  # copia etiquetas y particion
    def __len__(self): return len(self.labels)  # cantidad de ejemplos del dataset
    def __getitem__(self,idx):  # genera el ejemplo idx con onda y etiqueta
        y=int(self.labels[idx])  # etiqueta numerica del ejemplo
        rng=np.random if self.split=="train" else np.random.RandomState(9000+idx)  # azar libre en train y fijo en val y test
        return torch.from_numpy(prog_key(y,rng)).unsqueeze(0), torch.tensor(y,dtype=torch.long)  # devuelve onda de forma 1 por 64000 y etiqueta

NT,NV,NTs=80,20,20  # ejemplos por clase para train val y test
tr=[i for i in range(24) for _ in range(NT)]; va=[i for i in range(24) for _ in range(NV)]; te=[i for i in range(24) for _ in range(NTs)]  # listas de etiquetas balanceadas en 24 clases
train_ds, val_ds, test_ds = KeyDataset(tr,"train"), KeyDataset(va,"val"), KeyDataset(te,"test")  # crea los tres datasets
BATCH=32  # tamano de lote que entra en Colab gratis
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True)  # cargador mezclado para entrenar
val_loader=DataLoader(val_ds,batch_size=BATCH)  # cargador ordenado para validar
test_loader=DataLoader(test_ds,batch_size=BATCH)  # cargador ordenado para test final
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | T={N_SAMPLES}")  # muestra tamanos y largo temporal
w0,y0=val_ds[0]; print("ej:",KEYS[y0.item()],tuple(w0.shape))  # toma un ejemplo y muestra clase y forma
plt.figure(figsize=(10,2.5)); plt.plot(w0.numpy()[0,:8000]); plt.title(f"Progresion 4s (zoom) - {KEYS[y0.item()]}"); plt.show()  # grafica los primeros 8000 puntos de la onda


# ==========================================================
# ## PASO 4 — CNN 1D a 24 clases
# ==========================================================

class KeyCNN1D(nn.Module):  # red CNN 1D de 24 salidas
    def __init__(self,n_classes=24,dropout=0.3):  # guarda cantidad de clases y dropout
        super().__init__()  # inicializa la base de PyTorch
        self.b1=nn.Sequential(nn.Conv1d(1,16,kernel_size=128,stride=8,padding=64),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(4))  # bloque 1 con kernel ancho que capta graves y comprime
        self.b2=nn.Sequential(nn.Conv1d(16,32,kernel_size=16,padding=8),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4))  # bloque 2 con kernel 16 para armonicos y comprime
        self.b3=nn.Sequential(nn.Conv1d(32,64,kernel_size=7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(4))  # bloque 3 con kernel 7 que afina timbre y comprime
        self.b4=nn.Sequential(nn.Conv1d(64,128,kernel_size=5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(4))  # bloque 4 con kernel 5 hasta 128 canales y comprime
        self.gap=nn.AdaptiveAvgPool1d(1)  # promedia todo el tiempo en un vector fijo
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,96),nn.ReLU(),nn.Dropout(dropout),nn.Linear(96,n_classes))  # clasificador de 128 a 96 a 24 con dropout
    def forward(self,x):  # define la pasada hacia adelante de la red
        x=self.b1(x); x=self.b2(x); x=self.b3(x); x=self.b4(x); return self.head(self.gap(x))  # aplica bloques, promedia tiempo y clasifica
model=KeyCNN1D(24).to(DEVICE)  # crea el modelo en GPU o CPU
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")  # muestra cantidad de parametros
with torch.no_grad(): print("in",tuple(torch.randn(2,1,N_SAMPLES).to(DEVICE).shape),"-> out",tuple(model(torch.randn(2,1,N_SAMPLES).to(DEVICE)).shape))  # prueba formas entrada salida sin gradientes


# ==========================================================
# ## PASO 5 — Entrenamiento (CrossEntropy 24 + Adam)
# ==========================================================

EPOCHS=12; LR=1e-3  # 12 epocas y paso de aprendizaje chico y seguro
crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=LR)  # perdida multiclase y optimizador Adam
def epoch(loader,train):  # corre una epoca de train o de evaluacion
    model.train(train)  # activa modo entrenar o modo evaluar
    tl,ok,n=0.0,0,0  # acumuladores de perdida, aciertos y total
    for xb,yb in loader:  # recorre los lotes del cargador
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)  # manda el lote al dispositivo
        if train: opt.zero_grad()  # borra gradientes anteriores al entrenar
        with torch.set_grad_enabled(train):  # usa gradientes solo al entrenar
            lg=model(xb); loss=crit(lg,yb)  # predice el lote y calcula la perdida
            if train: loss.backward(); opt.step()  # retropropaga y actualiza pesos
        tl+=loss.item()*len(xb); ok+=(lg.argmax(1)==yb).sum().item(); n+=len(xb)  # acumula perdida y aciertos del lote
    return tl/n, ok/n  # devuelve perdida y accuracy promedio
hist={"tl":[],"vl":[],"ta":[],"va":[]}; best,bs=0,None  # historial de curvas y mejor modelo
for ep in range(1,EPOCHS+1):  # repite las 12 epocas
    a,b=epoch(train_loader,True); c,d=epoch(val_loader,False)  # entrena una epoca y valida
    hist["tl"].append(a); hist["vl"].append(c); hist["ta"].append(b); hist["va"].append(d)  # guarda metricas para graficar
    if d>best: best=d; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}  # guarda copia en CPU del mejor modelo
    print(f"Ep {ep:02d} train {a:.4f}/{b:.3f} || val {c:.4f}/{d:.3f}")  # muestra el progreso de la epoca
model.load_state_dict({k:v.to(DEVICE) for k,v in bs.items()}); print(f"Best val {best:.3f}")  # restaura el mejor y muestra su accuracy


# ==========================================================
# ## PASO 6 — Evaluación: curvas + matriz 24x24 + score MIREX
# ==========================================================

fig,ax=plt.subplots(1,2,figsize=(12,4))  # figura con dos graficos lado a lado
ax[0].plot(hist["tl"],label="train"); ax[0].plot(hist["vl"],label="val"); ax[0].set_title("Loss"); ax[0].legend()  # curva de perdida de train contra val
ax[1].plot(hist["ta"],label="train"); ax[1].plot(hist["va"],label="val"); ax[1].set_title("Acc"); ax[1].legend(); plt.show()  # curva de accuracy de train contra val
model.eval(); P,T=[],[]  # modo evaluacion y listas de prediccion y verdad
with torch.no_grad():  # evalua sin calcular gradientes
    for xb,yb in test_loader: P+=model(xb.to(DEVICE)).argmax(1).cpu().tolist(); T+=yb.tolist()  # junta predicciones y etiquetas de todo test
print(f"Test acc: {accuracy_score(T,P):.3f}")  # muestra accuracy final en test
cm=confusion_matrix(T,P,labels=list(range(24)))  # matriz de confusion de 24 por 24
plt.figure(figsize=(11,9)); sns.heatmap(cm,annot=False,cmap="Blues",xticklabels=KEYS,yticklabels=KEYS)  # dibuja la matriz como mapa de calor
plt.title("Confusion 24x24 (test sintetico)"); plt.xticks(rotation=90); plt.tight_layout(); plt.show()  # titula, rota etiquetas y muestra
# Score estilo MIREX: correcta=1, quinta=0.5, relativa/paralela=0.3, otro=0
def mirex(t,p):  # puntaje estilo MIREX con credito parcial
    if t==p: return 1.0  # acierto exacto vale 1 punto
    tt,mt=t%12,t<12; pt,mp=p%12,p<12  # separa tonica y modo de verdad y prediccion
    if mt==mp and (pt-tt)%12==7: return 0.5  # quinta del mismo modo vale 0 punto 5
    if (mt!=mp) and ((tt==0 and pt==9) or (pt==0 and tt==9) or ((pt-tt)%12==9 and mt) or ((tt-pt)%12==9 and mp)): return 0.3  # relativa o cambio de modo vale 0 punto 3
    if tt==pt and mt!=mp: return 0.3  # misma tonica en otro modo vale 0 punto 3
    return 0.0  # cualquier otro error vale 0
print(f"MIREX ponderado: {np.mean([mirex(t,p) for t,p in zip(T,P)]):.3f}")  # muestra promedio MIREX en test


# ==========================================================
# ## PASO 7 — Regla armónica (4ta justa, +5 semitonos)
#
# Solo 3 acciones: soplar / aspirar / bending (aspirar frenando el aire, suena lloradito). La tabla no se memoriza: sale de la fórmula.
# ==========================================================

def cuarta_arriba(tonica): return TONICS[(T2I[tonica]+5)%12]  # 2da posicion
CROSS={t:cuarta_arriba(t) for t in TONICS}  # tabla cancion a armonica con la regla mas 5
print("Cross-Harp (regla +5):",CROSS)  # muestra la tabla completa
# Riff detallado para C; el resto usa el riff generico de 2da posicion (celdas 2-4, casa en 2 aspirando).
RIFF_D={"C":["celda 2 aspirando","celda 3 aspirando con bending","celda 4 aspirando","celda 4 soplando","celda 4 aspirando","celda 3 aspirando con bending","celda 2 aspirando"],"default":["celda 2 aspirando (casa)","celda 3 aspirando con bending (lloradito)","celda 4 aspirando","celda 4 soplando","celda 4 aspirando","celda 3 aspirando con bending","celda 2 aspirando"]}
def recomendar(tonica):  # devuelve armonica y riff para una tonica
    h=CROSS[tonica]; return h, RIFF_D.get(h,RIFF_D["default"])  # busca armonica y riff o usa el generico
for d in ["G","A","E"]: print(d,"-> armonica",recomendar(d)[0])  # demo rapida con tres ejemplos


# ==========================================================
# ## PASO 8 — Audio real + GiantSteps Key (descarga automática 20 temas)
#
# **Entrada:** `WAV/MP3` -> mono 16kHz -> ventanas de 4 s. **Dataset real:** [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) — 604 previews 2 min EDM Beatport, 24 keys, ISMIR 2015 (Knees et al.). La celda clona las anotaciones, elige 20 temas al azar (seed 7), descarga los MP3 del mirror JKU con verificación md5 y los deja en `GS_ITEMS` **solo para evaluar** (no entrena). Si no hay internet o un link cayó, se sigue con la demo propia en `A` (tonalidad conocida) + upload manual.
#
# ==========================================================

def cargar_audio_Xs(path,sr=SR,dur=DUR):  # carga un audio y lo corta en ventanas
    y,_=librosa.load(path,sr=sr,mono=True); n=int(sr*dur); vs=[]  # lee mono a 16kHz y prepara lista de ventanas
    for i in range(0,max(len(y),1),n):  # recorre el audio de a ventanas
        s=y[i:i+n];  # toma el tramo actual del audio
        s=np.pad(s,(0,n-len(s))) if len(s)<n else s  # rellena con ceros si la cola es corta
        vs.append((0.95*s/(np.max(np.abs(s))+1e-9)).astype(np.float32))  # normaliza por pico y guarda la ventana
        if i+n>=len(y): break  # corta el bucle al llegar al final
    print(f"{os.path.basename(path)}: {len(y)/sr:.1f}s -> {len(vs)} ventanas {dur}s"); return vs  # informa ventanas y las devuelve
# Demo propia tonalidad conocida: 12-bar blues en A, SOLO walking bass (etiqueta: A maj para este TP)
def _f(m): return 440.0*(2.0**((m-69)/12.0))  # convierte nota MIDI a frecuencia en Hz
def _n(fr,d,sr=SR,rng=None,g=0.4):  # genera una nota corta con 3 armonicos
    rng=np.random if rng is None else rng; n=int(sr*d); t=np.arange(n,dtype=np.float32)/sr; x=np.zeros(n,dtype=np.float32)  # azar, muestras, tiempo y vector vacio
    for h,a in enumerate([1.0,0.45,0.22],start=1): x+=(a*np.sin(2*np.pi*fr*h*t+rng.uniform(0,6.28))).astype(np.float32)  # suma los 3 armonicos de la nota
    return (g*x).astype(np.float32)  # aplica volumen y devuelve la nota
def demo_blues(key="A",sr=SR,seed=7):  # arma el blues demo de 12 compases
    rng=np.random.RandomState(seed); BAR=2.0; roots=[45,45,50,45,52,50,45,52,45,50,45,52] if key=="A" else [43,43,48,43,50,48,43,50,43,48,43,50]  # azar fijo y raices de blues en A o en G
    out=np.zeros(int(sr*BAR*12),dtype=np.float32)  # pista vacia de 24 segundos
    for b,r in enumerate(roots):  # recorre los 12 compases
        base=b*int(sr*BAR)  # muestra inicial del compas actual
        bse=_f(r-12); out[base:base+int(sr*2.0)]+=np.concatenate([_n(bse,0.5,sr,rng,0.5) for _ in range(4)])[:int(sr*2.0)]  # suma bajo caminante de 4 negras en el compas
    out=np.tanh(out); return (0.9*out/(np.max(np.abs(out))+1e-9)).astype(np.float32)  # saturacion suave, normaliza y devuelve
os.makedirs("/tmp",exist_ok=True); DEMO="/tmp/demo_blues_A.wav"  # crea carpeta temporal y ruta de la demo
wavfile.write(DEMO,SR,(demo_blues()*32767).astype(np.int16)); print("Demo:",DEMO,"| key verdadera: A maj")  # guarda el WAV y muestra su clave verdadera
MI_AUDIO=DEMO  # usa la demo salvo que se suba otro audio
try:  # intenta abrir el boton de subida de Colab
    from google.colab import files as _F; print("Subi MP3/WAV o cancela para demo:"); _u=_F.upload()  # pide el archivo al usuario y lo recibe
    
    if _u:  # si se subio algun archivo
        _k=sorted(_u.keys())[0]; open(f"/content/{_k}","wb").write(_u[_k]); MI_AUDIO=f"/content/{_k}"  # guarda lo subido y lo usa como audio
except ImportError: print("Local -> demo")  # fuera de Colab sigue con la demo
# ---------- Descarga automatica GiantSteps: 20 temas al azar (SOLO evaluacion) ----------
import subprocess, hashlib, urllib.request  # importa comandos shell, md5 y descargas web
GS_REPO="https://github.com/GiantSteps/giantsteps-key-dataset.git"  # direccion del repo de anotaciones GiantSteps
GS_DIR="/tmp/gs_key"  # solo anotaciones (liviano); los audios NO se versionan en git
GS_MP3="https://www.cp.jku.at/datasets/giantsteps/backup/"  # mirror JKU (backup de Beatport)
FLAT={"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}  # anotaciones usan bemoles, el modelo sostenidos
def norm_key(txt):  # normaliza la clave GiantSteps al formato del TP
    """'Bb major' -> 'A# maj' (formato KEYS del TP)."""
    t,m=txt.strip().split(); return f"{FLAT.get(t,t)} {'maj' if m=='major' else 'min'}"  # separa tonica y modo y traduce bemol a sostenido
def descargar_giantsteps(n=20,seed=7):  # descarga n temas al azar solo para evaluar
    """Clona anotaciones, elige n temas al azar, descarga MP3 con md5. Devuelve [(path,key_verdadero)]."""
    if not os.path.isdir(os.path.join(GS_DIR,"annotations")):  # si todavia faltan las anotaciones
        print("Clonando anotaciones GiantSteps...")  # avisa que clona las anotaciones
        subprocess.run(["git","clone","-q","--depth","1",GS_REPO,GS_DIR],check=True)  # clona superficial solo las anotaciones
    ids=sorted(f[:-4] for f in os.listdir(os.path.join(GS_DIR,"annotations","key")) if f.endswith(".key"))  # lista los 604 ids con anotacion
    sel=random.Random(seed).sample(ids,min(n,len(ids)))  # elige n temas al azar de forma reproducible
    os.makedirs("/tmp/gs_audio",exist_ok=True); out=[]  # carpeta de mp3 y lista de resultados
    for i in sel:  # recorre los temas elegidos
        try:  # intenta un tema sin cortar los demas si falla
            key=open(os.path.join(GS_DIR,"annotations","key",i+".key")).read()  # lee la clave verdadera anotada
            mp3=f"/tmp/gs_audio/{i}.mp3"  # ruta local del mp3 del tema
            if not (os.path.exists(mp3) and os.path.getsize(mp3)>50000):  # si falta el archivo o esta cortado
                urllib.request.urlretrieve(GS_MP3+i+".mp3",mp3)  # descarga el preview desde el mirror JKU
            if os.path.getsize(mp3)<50000: print(f"x {i}: descarga incompleta"); continue  # descarta descarga incompleta y sigue
            ref=open(os.path.join(GS_DIR,"md5",i+".md5")).read().strip()  # lee el md5 esperado del repo
            if hashlib.md5(open(mp3,"rb").read()).hexdigest()!=ref: print(f"x {i}: md5 distinto"); continue  # descarta si el audio no coincide y sigue
            out.append((mp3,norm_key(key)))  # guarda path y clave normalizada
        except Exception as e: print(f"x {i}: {e}")  # informa el tema fallido y sigue
    print(f"GiantSteps: {len(out)}/{len(sel)} audios OK (seed {seed}, solo evaluacion, no entrena)")  # resume cuantos audios quedaron OK
    return out  # devuelve la lista de path y clave
try:  # intenta la descarga automatica de los 20 temas
    GS_ITEMS=descargar_giantsteps(20,seed=7)  # baja los 20 temas al ejecutar todo
except Exception as e:  # si no hay internet o fallan los links
    print("GiantSteps no disponible (sin internet o links caidos):",e,"-> demo + upload.")  # avisa y se sigue con demo y upload
    GS_ITEMS=[]  # lista vacia para omitir la tabla 9b


# ==========================================================
# ## PASO 9 — Inferencia CNN vs Baseline K-S
# ==========================================================

MAJ=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],dtype=float)  # perfil mayor de Krumhansl y Kessler
MIN=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17],dtype=float)  # perfil menor de Krumhansl y Kessler
def baseline_ks(path):  # estima la clave sin redes por cromagrama
    y,_=librosa.load(path,sr=SR,mono=True)  # lee el audio en mono a 16kHz
    ch=librosa.feature.chroma_cqt(y=y,sr=SR).mean(axis=1)  # cromagrama de 12 bandas promediado en tiempo
    best,bi=-2,0  # mejor correlacion y mejor clave hasta ahora
    for t in range(12):  # prueba las 12 tonicas
        for m,prof in [("maj",MAJ),("min",MIN)]:  # prueba modo mayor y modo menor
            pr=np.roll(prof,t)  # rota el perfil hasta la tonica t
            r=np.corrcoef(ch,pr)[0,1]  # correlacion de Pearson entre croma y perfil
            if r>best: best,bi=r,(t,m)  # guarda la clave con mayor correlacion
    lab=f"{TONICS[bi[0]]} {bi[1]}"; return lab,float(best)  # arma la etiqueta y devuelve con su r
def predecir_cnn(path,model=model,maxv=6,verbose=True):  # predice la clave con la CNN por voto
    model.eval(); vs=cargar_audio_Xs(path)  # pone modo evaluar y corta el audio en ventanas
    
    if len(vs)>maxv: vs=[vs[i] for i in np.linspace(0,len(vs)-1,maxv).round().astype(int)]  # en audios largos muestrea 6 ventanas parejas
    xb=torch.stack([torch.from_numpy(w).unsqueeze(0) for w in vs]).to(DEVICE)  # arma el lote de ventanas en el dispositivo
    with torch.no_grad(): v=torch.softmax(model(xb),1).mean(0)  # promedia probabilidades sin gradientes
    i=int(v.argmax()); top=sorted(range(24),key=lambda k: float(v[k]),reverse=True)[:3]  # toma la ganadora y el top 3
    if verbose: print("CNN Top-3:",{KEYS[k]:round(float(v[k]),3) for k in top})  # muestra el top 3 solo si se pide
    return KEYS[i],float(v[i])  # devuelve etiqueta y confianza
cnn_lab,cnn_c=predecir_cnn(MI_AUDIO); ks_lab,ks_r=baseline_ks(MI_AUDIO)  # predice el audio con ambos metodos
print(f"\nCNN: {cnn_lab} ({cnn_c:.2f}) | K-S: {ks_lab} (r={ks_r:.2f}) | verdad demo: A maj")  # muestra CNN contra K-S y la verdad de la demo
ton_cnn=cnn_lab.split()[0]  # extrae la tonica para la armonica


# ==========================================================
# ## PASO 9b — Tabla CNN vs K-S en GiantSteps (20 temas, solo evaluación)
#
# Cada tema se predice con la CNN (voto en 6 ventanas) y con el baseline K-S, sin entrenar. Se reporta accuracy y score MIREX de cada uno. Si `GS_ITEMS` está vacío (sin internet), la celda se omite.
#
# ==========================================================

if GS_ITEMS:  # si hay temas GiantSteps descargados
    print(f"{'tema':28s} {'verdad':8s} {'cnn':8s} {'ks':8s} cnn_ok ks_ok")  # imprime el encabezado de la tabla
    ok_c=ok_k=0; mc=mk=0.0  # contadores de aciertos y puntaje MIREX
    for path,true in GS_ITEMS:  # recorre los temas con su clave verdadera
        cnn,_=predecir_cnn(path,verbose=False); ks,_=baseline_ks(path)  # predice en silencio con ambos metodos
        cc,kk=cnn==true,ks==true; ok_c+=cc; ok_k+=kk  # compara con la verdad y acumula aciertos
        sc,sk=mirex(K2I[cnn],K2I[true]),mirex(K2I[ks],K2I[true]); mc+=sc; mk+=sk  # calcula puntaje parcial y lo acumula
        print(f"{os.path.basename(path):28s} {true:8s} {cnn:8s} {ks:8s} {int(cc):>6d} {int(kk):>5d}")  # imprime la fila del tema
    n=len(GS_ITEMS)  # cantidad de temas evaluados
    print(f"\nGiantSteps-{n} | CNN acc {ok_c/n:.2f} mirex {mc/n:.2f} || K-S acc {ok_k/n:.2f} mirex {mk/n:.2f}")  # muestra accuracy y MIREX de CNN contra K-S
else:  # si no hubo descarga de GiantSteps
    print("Sin GiantSteps (GS_ITEMS vacio) -> evaluacion solo sintetica + demo.")  # avisa que solo vale sintetica y demo


# ==========================================================
# ## PASO 10 — As de la Armónica
# ==========================================================

h,riff=recomendar(ton_cnn)  # pide armonica y riff para la tonica
print("="*52); print("   EL AS DE LA ARMONICA"); print("="*52)  # dibuja el marco del informe final
print(f"Tonalidad CNN : {cnn_lab} (conf {cnn_c:.2f})")  # muestra tonalidad y confianza del modelo
print(f"Baseline K-S  : {ks_lab} (r={ks_r:.2f})")  # muestra baseline y su correlacion
print(f"Agarra         : ARMONICA EN {h}  (cancion {ton_cnn} +5 semitonos)")  # indica la armonica con la regla mas 5
for i,p in enumerate(riff,1): print(f"   {i}. {p}")  # lista los pasos del riff en orden
print("Tip: casa = celda 2 aspirando."); print("="*52)  # consejo final y cierre del marco


# ==========================================================
# ### Cierre — para planilla y repo
#
# **Descripción breve (pegar en planilla):** Clasificación de tonalidad en 24 clases desde audio crudo mono 16kHz (ventanas 4 s) con CNN 1D desde cero; derivación determinística de armónica blues 2da posición (+5 semitonos) + riff. Baseline: cromagrama + Krumhansl-Schmuckler.
#
# **Red:** `KeyCNN1D` CNN 1D 24 salidas (CrossEntropy+Adam). Baseline no-neuronal K-S.
#
# **Dataset:** sintético propio progresiones I-IV-V / i-iv-V (train 1920/val 480/test 480, SNR 12-25dB) + evaluación en canciones reales tonalidad conocida: demo 12-bar A + subset GiantSteps Key (github GiantSteps/giantsteps-key-dataset, 604x2min, 24 keys, Knees et al. ISMIR2015). Sin descargas pesadas por defecto.
#
# **Límites:** 4 s no capta modulaciones; blues mezcla mayor/menor (A maj vs A min) → reportar Top-3 + score MIREX; futuro: log-mel + CNN 2D / YAMNet.
#
# ==========================================================
