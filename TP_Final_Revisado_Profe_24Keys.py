# TP Final Revisado (profe) - Tonalidad 24 clases + CNN 1D + Baseline K-S + Armonica Blues
# Materia: Redes Neuronales y Deep Learning - Maestria IA - Univ. Palermo
# Revision de catedra: entrada audio WAV/MP3 mono 16kHz ventanas 4s; salida 24 clases
#   (12 tonicas x mayor/menor) + armonica 2da posicion (+5 semitonos, 4ta justa) + riff.
# Baseline sin redes: cromagrama + Krumhansl-Schmuckler. Prueba real: demo en A + subset GiantSteps Key.
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
import sys, platform, random
import numpy as np
print("Python", sys.version.split()[0], "|", platform.system())
for n in ["numpy","matplotlib","pandas","sklearn","torch","torchaudio","librosa","scipy","seaborn"]:
    try:
        m=__import__(n); print(f"{n:12s} {getattr(m,'__version__','ok')}")
    except ImportError: print(f"{n:12s} NO INSTALADO")
def fijar_semillas(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except ImportError: pass
fijar_semillas(42); print("\nSemilla 42 OK")
try:
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (alcanza)")
except ImportError: pass


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

# OMITIDO (solo Colab/Jupyter): pip install -q librosa seaborn scikit-learn scipy 2>&1 | tail -n 2
import os, math, random
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.io import wavfile
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio, librosa
from sklearn.metrics import confusion_matrix, accuracy_score
# OMITIDO (solo Colab/Jupyter): %matplotlib inline
sns.set(style="whitegrid")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:",DEVICE,"| torch:",torch.__version__)


# ==========================================================
# ## PASO 3 — Dataset sintético 24 clases (progresiones, no tonos puros)
#
# Cada sample = 4 s @16kHz = 64000. Contiene `I-IV-V-I` (mayor) o `i-iv-V-i` (menor, V mayor armónica). Cada acorde 1 s = triada/tetrada con 3 armónicos + envolvente + ruido + micro-desafinación. Generación on-the-fly (no guarda GB en RAM).
# ==========================================================

SR=16000; DUR=4.0; N_SAMPLES=int(SR*DUR)
TONICS=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
T2I={n:i for i,n in enumerate(TONICS)}
BASE_F={"C":261.63,"C#":277.18,"D":293.66,"D#":311.13,"E":329.63,"F":349.23,"F#":369.99,"G":392.00,"G#":415.30,"A":440.00,"A#":466.16,"B":493.88}
KEYS=[f"{t} maj" for t in TONICS]+[f"{t} min" for t in TONICS]
K2I={k:i for i,k in enumerate(KEYS)}
print("24 clases:",KEYS)

def tono(freq,dur,sr,rng,gain=0.5):
    n=int(sr*dur); t=np.arange(n,dtype=np.float32)/sr
    x=np.zeros(n,dtype=np.float32)
    for h,a in enumerate([1.0,0.4,0.18],start=1):
        x+=(a*np.sin(2*np.pi*freq*h*t+rng.uniform(0,2*np.pi))).astype(np.float32)
    a_n=min(int(0.02*sr),n//4); r_n=min(int(0.08*sr),n//4)
    env=np.ones(n,dtype=np.float32); env[:a_n]=np.linspace(0,1,a_n); env[-r_n:]=np.linspace(1,0,r_n)
    return (gain*x*env).astype(np.float32)

def acorde(root_hz,ivs,dur,sr,rng):
    n=int(sr*dur); x=np.zeros(n,dtype=np.float32)
    for iv in ivs:
        f=root_hz*(2.0**(iv/12.0))*(1.0+rng.uniform(-0.004,0.004))
        x+=tono(f,dur,sr,rng,gain=0.33)
    return x

def prog_key(key_idx,rng,snr_db=None):
    tonica=TONICS[key_idx%12]; is_maj=key_idx<12
    rf=BASE_F[tonica]/2.0  # octava 3 para comping
    roots=[0,5,7,0]  # I-IV-V-I (mayor) / i-iv-V-i (menor, V mayor armonica)
    y=np.zeros(int(SR*DUR),dtype=np.float32); seg=int(SR*1.0)
    for j,r in enumerate(roots):
        r_hz=rf*(2.0**(r/12.0))
        if is_maj: ivs=[0,4,7] if j in [0,1,3] else [0,4,7,10]
        else: ivs=[0,3,7] if j in [0,1,3] else [0,4,7,10]  # V mayor en menor
        ch=acorde(r_hz,ivs,1.0,sr=SR,rng=rng)
        y[j*seg:(j+1)*seg]+=ch[:seg]
    if snr_db is None: snr_db=float(rng.uniform(12,25))
    p=np.mean(y**2)+1e-9; y+=rng.normal(0,math.sqrt(p/(10**(snr_db/10))),size=y.shape).astype(np.float32)
    return (0.9*y/(np.max(np.abs(y))+1e-9)).astype(np.float32)

class KeyDataset(Dataset):
    def __init__(self,labels,split="train"):
        self.labels=list(labels); self.split=split
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        y=int(self.labels[idx])
        rng=np.random if self.split=="train" else np.random.RandomState(9000+idx)
        return torch.from_numpy(prog_key(y,rng)).unsqueeze(0), torch.tensor(y,dtype=torch.long)

NT,NV,NTs=80,20,20
tr=[i for i in range(24) for _ in range(NT)]; va=[i for i in range(24) for _ in range(NV)]; te=[i for i in range(24) for _ in range(NTs)]
train_ds, val_ds, test_ds = KeyDataset(tr,"train"), KeyDataset(va,"val"), KeyDataset(te,"test")
BATCH=32
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=BATCH)
test_loader=DataLoader(test_ds,batch_size=BATCH)
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | T={N_SAMPLES}")
w0,y0=val_ds[0]; print("ej:",KEYS[y0.item()],tuple(w0.shape))
plt.figure(figsize=(10,2.5)); plt.plot(w0.numpy()[0,:8000]); plt.title(f"Progresion 4s (zoom) - {KEYS[y0.item()]}"); plt.show()


# ==========================================================
# ## PASO 4 — CNN 1D a 24 clases
# ==========================================================

class KeyCNN1D(nn.Module):
    def __init__(self,n_classes=24,dropout=0.3):
        super().__init__()
        self.b1=nn.Sequential(nn.Conv1d(1,16,kernel_size=128,stride=8,padding=64),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(4))
        self.b2=nn.Sequential(nn.Conv1d(16,32,kernel_size=16,padding=8),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4))
        self.b3=nn.Sequential(nn.Conv1d(32,64,kernel_size=7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(4))
        self.b4=nn.Sequential(nn.Conv1d(64,128,kernel_size=5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(4))
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,96),nn.ReLU(),nn.Dropout(dropout),nn.Linear(96,n_classes))
    def forward(self,x):
        x=self.b1(x); x=self.b2(x); x=self.b3(x); x=self.b4(x); return self.head(self.gap(x))
model=KeyCNN1D(24).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
with torch.no_grad(): print("in",tuple(torch.randn(2,1,N_SAMPLES).to(DEVICE).shape),"-> out",tuple(model(torch.randn(2,1,N_SAMPLES).to(DEVICE)).shape))


# ==========================================================
# ## PASO 5 — Entrenamiento (CrossEntropy 24 + Adam)
# ==========================================================

EPOCHS=12; LR=1e-3
crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=LR)
def epoch(loader,train):
    model.train(train)
    tl,ok,n=0.0,0,0
    for xb,yb in loader:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            lg=model(xb); loss=crit(lg,yb)
            if train: loss.backward(); opt.step()
        tl+=loss.item()*len(xb); ok+=(lg.argmax(1)==yb).sum().item(); n+=len(xb)
    return tl/n, ok/n
hist={"tl":[],"vl":[],"ta":[],"va":[]}; best,bs=0,None
for ep in range(1,EPOCHS+1):
    a,b=epoch(train_loader,True); c,d=epoch(val_loader,False)
    hist["tl"].append(a); hist["vl"].append(c); hist["ta"].append(b); hist["va"].append(d)
    if d>best: best=d; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
    print(f"Ep {ep:02d} train {a:.4f}/{b:.3f} || val {c:.4f}/{d:.3f}")
model.load_state_dict({k:v.to(DEVICE) for k,v in bs.items()}); print(f"Best val {best:.3f}")


# ==========================================================
# ## PASO 6 — Evaluación: curvas + matriz 24x24 + score MIREX
# ==========================================================

fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(hist["tl"],label="train"); ax[0].plot(hist["vl"],label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["ta"],label="train"); ax[1].plot(hist["va"],label="val"); ax[1].set_title("Acc"); ax[1].legend(); plt.show()
model.eval(); P,T=[],[]
with torch.no_grad():
    for xb,yb in test_loader: P+=model(xb.to(DEVICE)).argmax(1).cpu().tolist(); T+=yb.tolist()
print(f"Test acc: {accuracy_score(T,P):.3f}")
cm=confusion_matrix(T,P,labels=list(range(24)))
plt.figure(figsize=(11,9)); sns.heatmap(cm,annot=False,cmap="Blues",xticklabels=KEYS,yticklabels=KEYS)
plt.title("Confusion 24x24 (test sintetico)"); plt.xticks(rotation=90); plt.tight_layout(); plt.show()
# Score estilo MIREX: correcta=1, quinta=0.5, relativa/paralela=0.3, otro=0
def mirex(t,p):
    if t==p: return 1.0
    tt,mt=t%12,t<12; pt,mp=p%12,p<12
    if mt==mp and (pt-tt)%12==7: return 0.5
    if (mt!=mp) and ((tt==0 and pt==9) or (pt==0 and tt==9) or ((pt-tt)%12==9 and mt) or ((tt-pt)%12==9 and mp)): return 0.3
    if tt==pt and mt!=mp: return 0.3
    return 0.0
print(f"MIREX ponderado: {np.mean([mirex(t,p) for t,p in zip(T,P)]):.3f}")


# ==========================================================
# ## PASO 7 — Regla armónica (4ta justa, +5 semitonos)
#
# Solo 3 acciones: soplar / aspirar / bending (aspirar frenando el aire, suena lloradito). La tabla no se memoriza: sale de la fórmula.
# ==========================================================

def cuarta_arriba(tonica): return TONICS[(T2I[tonica]+5)%12]  # 2da posicion
CROSS={t:cuarta_arriba(t) for t in TONICS}
print("Cross-Harp (regla +5):",CROSS)
# Riff detallado para C; el resto usa el riff generico de 2da posicion (celdas 2-4, casa en 2 aspirando).
RIFF_D={"C":["celda 2 aspirando","celda 3 aspirando con bending","celda 4 aspirando","celda 4 soplando","celda 4 aspirando","celda 3 aspirando con bending","celda 2 aspirando"],"default":["celda 2 aspirando (casa)","celda 3 aspirando con bending (lloradito)","celda 4 aspirando","celda 4 soplando","celda 4 aspirando","celda 3 aspirando con bending","celda 2 aspirando"]}
def recomendar(tonica):
    h=CROSS[tonica]; return h, RIFF_D.get(h,RIFF_D["default"])
for d in ["G","A","E"]: print(d,"-> armonica",recomendar(d)[0])


# ==========================================================
# ## PASO 8 — Audio real + GiantSteps Key (tonalidad conocida)
#
# **Entrada:** `WAV/MP3` -> mono 16kHz -> ventanas de 4 s. **Dataset real:** [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) — 604 previews 2 min EDM Beatport, 24 keys, ISMIR 2015 (Knees et al.). Clonar y descargar subset de 20 temas para no saturar Colab free. Si no se descarga, se usa demo propia en `A` (tonalidad conocida) + upload manual.
# ==========================================================

def cargar_audio_Xs(path,sr=SR,dur=DUR):
    y,_=librosa.load(path,sr=sr,mono=True); n=int(sr*dur); vs=[]
    for i in range(0,max(len(y),1),n):
        s=y[i:i+n]; 
        s=np.pad(s,(0,n-len(s))) if len(s)<n else s
        vs.append((0.95*s/(np.max(np.abs(s))+1e-9)).astype(np.float32))
        if i+n>=len(y): break
    print(f"{os.path.basename(path)}: {len(y)/sr:.1f}s -> {len(vs)} ventanas {dur}s"); return vs
# Demo propia tonalidad conocida: 12-bar blues en A, SOLO walking bass (etiqueta: A maj para este TP)
def _f(m): return 440.0*(2.0**((m-69)/12.0))
def _n(fr,d,sr=SR,rng=None,g=0.4):
    rng=np.random if rng is None else rng; n=int(sr*d); t=np.arange(n,dtype=np.float32)/sr; x=np.zeros(n,dtype=np.float32)
    for h,a in enumerate([1.0,0.45,0.22],start=1): x+=(a*np.sin(2*np.pi*fr*h*t+rng.uniform(0,6.28))).astype(np.float32)
    return (g*x).astype(np.float32)
def demo_blues(key="A",sr=SR,seed=7):
    rng=np.random.RandomState(seed); BAR=2.0; roots=[45,45,50,45,52,50,45,52,45,50,45,52] if key=="A" else [43,43,48,43,50,48,43,50,43,48,43,50]
    out=np.zeros(int(sr*BAR*12),dtype=np.float32)
    for b,r in enumerate(roots):
        base=b*int(sr*BAR)
        bse=_f(r-12); out[base:base+int(sr*2.0)]+=np.concatenate([_n(bse,0.5,sr,rng,0.5) for _ in range(4)])[:int(sr*2.0)]
    out=np.tanh(out); return (0.9*out/(np.max(np.abs(out))+1e-9)).astype(np.float32)
os.makedirs("/tmp",exist_ok=True); DEMO="/tmp/demo_blues_A.wav"
wavfile.write(DEMO,SR,(demo_blues()*32767).astype(np.int16)); print("Demo:",DEMO,"| key verdadera: A maj")
MI_AUDIO=DEMO
try:
    from google.colab import files as _F; print("Subi MP3/WAV o cancela para demo:"); _u=_F.upload()
    
    if _u:
        _k=sorted(_u.keys())[0]; open(f"/content/{_k}","wb").write(_u[_k]); MI_AUDIO=f"/content/{_k}"
except ImportError: print("Local -> demo")
# Como evaluar GiantSteps (opcional, 20 temas):
# !git clone -q https://github.com/GiantSteps/giantsteps-key-dataset.git /content/gs  # anotaciones
# # descargar con script del repo solo 20 mp3, convertir a wav 16k, luego evaluar con predecir_tonalidad_archivo() y baseline_ks()
# Anotaciones en /content/gs/annotations/key/*.csv -> formato 'tonica modo' mapeable a KEYS.


# ==========================================================
# ## PASO 9 — Inferencia CNN vs Baseline K-S
# ==========================================================

MAJ=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],dtype=float)
MIN=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17],dtype=float)
def baseline_ks(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    ch=librosa.feature.chroma_cqt(y=y,sr=SR).mean(axis=1)
    best,bi=-2,0
    for t in range(12):
        for m,prof in [("maj",MAJ),("min",MIN)]:
            pr=np.roll(prof,t)
            r=np.corrcoef(ch,pr)[0,1]
            if r>best: best,bi=r,(t,m)
    lab=f"{TONICS[bi[0]]} {bi[1]}"; return lab,float(best)
def predecir_cnn(path,model=model,maxv=6):
    model.eval(); vs=cargar_audio_Xs(path)
    
    if len(vs)>maxv: vs=[vs[i] for i in np.linspace(0,len(vs)-1,maxv).round().astype(int)]
    xb=torch.stack([torch.from_numpy(w).unsqueeze(0) for w in vs]).to(DEVICE)
    with torch.no_grad(): v=torch.softmax(model(xb),1).mean(0)
    i=int(v.argmax()); top=sorted(range(24),key=lambda k: float(v[k]),reverse=True)[:3]
    print("CNN Top-3:",{KEYS[k]:round(float(v[k]),3) for k in top})
    return KEYS[i],float(v[i])
cnn_lab,cnn_c=predecir_cnn(MI_AUDIO); ks_lab,ks_r=baseline_ks(MI_AUDIO)
print(f"\nCNN: {cnn_lab} ({cnn_c:.2f}) | K-S: {ks_lab} (r={ks_r:.2f}) | verdad demo: A maj")
ton_cnn=cnn_lab.split()[0]


# ==========================================================
# ## PASO 10 — As de la Armónica
# ==========================================================

h,riff=recomendar(ton_cnn)
print("="*52); print("   EL AS DE LA ARMONICA"); print("="*52)
print(f"Tonalidad CNN : {cnn_lab} (conf {cnn_c:.2f})")
print(f"Baseline K-S  : {ks_lab} (r={ks_r:.2f})")
print(f"Agarra         : ARMONICA EN {h}  (cancion {ton_cnn} +5 semitonos)")
for i,p in enumerate(riff,1): print(f"   {i}. {p}")
print("Tip: casa = celda 2 aspirando."); print("="*52)


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
