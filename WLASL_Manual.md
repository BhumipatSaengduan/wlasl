# 🤟 WLASL Demo — คู่มือฉบับสมบูรณ์ตั้งแต่ต้นจนจบ

> ปลายทาง: อัปโหลดวิดีโอท่าภาษามือ ASL → ได้คำภาษาอังกฤษออกมาเลย

---

## 📋 ภาพรวม

```
STEP 1  — เตรียมเครื่อง (Python, Node.js, ffmpeg, Git)
STEP 2  — Clone repo
STEP 3  — Train โมเดลใน Google Colab
STEP 4  — Fix labels + Export weights
STEP 5  — Download weights มาที่เครื่อง
STEP 6  — ติดตั้ง dependencies บนเครื่อง
STEP 7  — รัน Backend API
STEP 8  — รัน Frontend (Web UI)
STEP 9  — ทดสอบและดูผล
```

---

## STEP 1 — เตรียมเครื่อง

### 1.1 ติดตั้ง Python 3.10+
- ดาวน์โหลด: https://www.python.org/downloads/
- ✅ ตรวจสอบ: `python --version` → เห็น `Python 3.10.x`

### 1.2 ติดตั้ง Node.js 18+
- ดาวน์โหลด: https://nodejs.org
- ✅ ตรวจสอบ: `node --version` → เห็น `v18.x.x`

### 1.3 ติดตั้ง Git
- ดาวน์โหลด: https://git-scm.com
- ✅ ตรวจสอบ: `git --version`

### 1.4 ติดตั้ง ffmpeg (Windows)

**วิธีที่ 1 — ผ่าน winget (ง่ายที่สุด):**
```powershell
winget install ffmpeg
```
ปิด terminal แล้วเปิดใหม่ แล้วรัน `ffmpeg -version`

**วิธีที่ 2 — ถ้า winget ใช้ไม่ได้:**
1. ดาวน์โหลด: https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
2. แตกไฟล์ → ย้ายโฟลเดอร์ไปไว้ที่ `C:\ffmpeg`
3. กด **Windows key** → พิมพ์ `environment` → **Edit the system environment variables**
4. คลิก **Environment Variables** → ใน **User variables** → คลิก **Path** → **Edit**
5. คลิก **New** → พิมพ์ `C:\ffmpeg\bin` → **OK** สามครั้ง
6. ปิด VSCode แล้วเปิดใหม่
7. ✅ ตรวจสอบ: `ffmpeg -version`

---

## STEP 2 — Clone repo

เปิด PowerShell ใน VSCode แล้วรัน:

```powershell
git clone https://github.com/BhumipatSaengduan/wlasl.git
cd wlasl
```

✅ ตรวจสอบ: เห็นโฟลเดอร์ `src/`, `scripts/`, `web/`, `colab/`

---

## STEP 3 — Train โมเดลใน Google Colab

### 3.1 เตรียม kaggle.json

1. ไปที่ https://www.kaggle.com/settings
2. เลื่อนหา **API** → คลิก **Create New Token**
3. ไฟล์ `kaggle.json` จะ download มาที่เครื่องอัตโนมัติ
4. เก็บไว้ — ใช้ upload เข้า Colab ใน STEP 3.4

> ⚠️ อย่า push kaggle.json ขึ้น GitHub เด็ดขาด

### 3.2 เปิด Notebook ใน Google Colab

1. ไปที่ https://colab.research.google.com
2. คลิก **File → Upload notebook**
3. Upload ไฟล์ `colab/Step7_Colab_Train_Export.ipynb` จาก repo

### 3.3 เปลี่ยน Runtime เป็น GPU

1. คลิก **Runtime** → **Change runtime type**
2. เลือก **T4 GPU** → **Save**

### 3.4 รัน Cell 1 — Clone repo

```python
!git clone https://github.com/BhumipatSaengduan/wlasl.git
%cd wlasl
```

✅ ตรวจสอบ: เห็น `Cloning into 'wlasl'...`

### 3.5 รัน Cell 2 — ติดตั้ง dependencies

```python
!pip -q install torch torchvision opencv-python tqdm
```

### 3.6 รัน Cell 3 — Upload kaggle.json และ download dataset

```python
!pip -q install kaggle
from google.colab import files
files.upload()  # popup ขึ้นมา → เลือกไฟล์ kaggle.json จากเครื่อง
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d risangbaskoro/wlasl-processed -p /content/data --unzip
!ls -la /content/data
```

✅ ตรวจสอบ: เห็นไฟล์เหล่านี้:
```
videos/                ← โฟลเดอร์วิดีโอ (~11,980 ไฟล์)
nslt_100.json
WLASL_v0.3.json
wlasl_class_list.txt
```

⏱️ ใช้เวลา: ~5-10 นาที

### 3.7 รัน Cell 4 — Train โมเดล

> ⚠️ ใช้ code นี้ทั้งหมด (แก้ไขแล้ว อ่าน dataset ได้ถูกต้อง)

เพิ่ม cell ใหม่ วาง code ทั้งหมดด้านล่าง แล้วรัน:

```python
import os, json, math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

os.chdir('/content/wlasl')  # สำคัญมาก อย่าลืม

# --- Load dataset จาก nslt_100.json ---
def load_nslt_dataset(data_root, json_path, max_classes=100, max_samples_per_class=50):
    with open(json_path) as f:
        data = json.load(f)
    video_dir = os.path.join(data_root, 'videos')
    class_to_samples = {}
    for video_id, info in data.items():
        class_idx = info['action'][0]
        video_file = os.path.join(video_dir, f"{int(video_id):05d}.mp4")
        if not os.path.exists(video_file):
            continue
        class_to_samples.setdefault(class_idx, []).append({
            'path': video_file, 'subset': info['subset']
        })
    selected_classes = sorted(class_to_samples.keys())[:max_classes]
    label_map = {orig: new for new, orig in enumerate(selected_classes)}
    train_samples, val_samples = [], []
    for orig_class in selected_classes:
        items = class_to_samples[orig_class][:max_samples_per_class]
        new_label = label_map[orig_class]
        for item in items:
            sample = (item['path'], new_label)
            if item['subset'] == 'val':
                val_samples.append(sample)
            else:
                train_samples.append(sample)
    labels = [str(c) for c in selected_classes]
    print(f"dataset: train={len(train_samples)} val={len(val_samples)} classes={len(selected_classes)}")
    return train_samples, val_samples, labels

# --- Model ---
class TinyFrameCNN(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.net(x).flatten(1)

class TinyVideoClassifier(nn.Module):
    def __init__(self, num_classes, frames=8, size=112, feat_dim=128):
        super().__init__()
        self.backbone = TinyFrameCNN(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        b, t, c, h, w = x.shape
        feat = self.backbone(x.view(b*t, c, h, w)).view(b, t, -1).mean(1)
        return self.classifier(feat)

# --- Dataset ---
FRAMES, SIZE = 8, 112

def sample_frames(path, num_frames=FRAMES, size=SIZE):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in np.linspace(0, total-1, num_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size))
                frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        frames = [np.zeros((size, size, 3), np.float32)]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    arr = np.stack(frames[:num_frames])
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).float()

class VideoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return sample_frames(path), label

# --- Train ---
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3
OUT_DIR = '/content/out'
os.makedirs(OUT_DIR, exist_ok=True)

train_samples, val_samples, labels = load_nslt_dataset(
    '/content/data', '/content/data/nslt_100.json',
    max_classes=100, max_samples_per_class=50
)

with open(f'{OUT_DIR}/labels.json', 'w') as f:
    json.dump(labels, f)

train_loader = DataLoader(VideoDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(VideoDataset(val_samples),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

model = TinyVideoClassifier(num_classes=len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_loss = math.inf
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"train {epoch}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"val {epoch}"):
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item() * x.size(0)
    train_loss /= max(1, len(train_samples))
    val_loss   /= max(1, len(val_samples))
    print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'meta': {'num_classes': len(labels), 'frames': FRAMES, 'size': SIZE}
        }, f'{OUT_DIR}/best.pt')
        print(f"  saved best checkpoint")
```

✅ ตรวจสอบ: เห็น `classes=100` และ loss มีค่าจริงๆ ไม่ใช่ 0
⏱️ ใช้เวลา: ~20-40 นาที (GPU T4)

---

## STEP 4 — Fix labels + Export weights (ยังอยู่ใน Colab)

### 4.1 แก้ labels.json ให้เป็นชื่อคำจริง

เพิ่ม cell ใหม่แล้วรัน:

```python
import json

# map index → ชื่อคำภาษาอังกฤษ
class_list = {}
with open('/content/data/wlasl_class_list.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            class_list[int(parts[0])] = parts[1]

# โหลด labels เดิม (ยังเป็นตัวเลข)
with open('/content/out/labels.json') as f:
    old_labels = json.load(f)

# แปลงเป็นชื่อคำ
new_labels = [class_list.get(int(l), f"unknown_{l}") for l in old_labels]
print("ตัวอย่าง:", new_labels[:10])
print(f"จำนวนทั้งหมด: {len(new_labels)} คำ")

# บันทึกทับ
with open('/content/out/labels.json', 'w') as f:
    json.dump(new_labels, f)
print("✅ labels.json อัปเดตแล้ว")
```

✅ ตรวจสอบ: เห็นชื่อคำภาษาอังกฤษ เช่น `['book', 'drink', 'computer', ...]`

### 4.2 Export เป็น TorchScript

เพิ่ม cell ใหม่แล้วรัน:

```python
import os
os.chdir('/content/wlasl')

!python3 -m src.export_torchscript \
    --ckpt /content/out/best.pt \
    --labels /content/out/labels.json \
    --out_ts /content/out/model.ts
```

✅ ตรวจสอบ: เห็น
```
TorchScript output shape: (1, 100)
Saved TorchScript: /content/out/model.ts
```

### 4.3 Download ไฟล์กลับมาที่เครื่อง

เพิ่ม cell ใหม่แล้วรัน:

```python
from google.colab import files
files.download("/content/out/model.ts")
files.download("/content/out/labels.json")
```

Browser จะ download 2 ไฟล์มาที่โฟลเดอร์ Downloads

---

## STEP 5 — วาง weights บนเครื่อง

เปิด PowerShell ใน VSCode:

```powershell
cd "C:\Users\bhumi\Desktop\CMU\4th Year\Chalarm\WLASL\wlasl_demo"

# สร้างโฟลเดอร์ weights
mkdir -Force weights

# วาง 2 ไฟล์ที่ download มา
copy "$env:USERPROFILE\Downloads\model.ts" weights\
copy "$env:USERPROFILE\Downloads\labels.json" weights\
```

✅ ตรวจสอบ:
```powershell
dir weights\
# ควรเห็น: labels.json  model.ts
```

---

## STEP 6 — ติดตั้ง Dependencies บนเครื่อง

```powershell
cd "C:\Users\bhumi\Desktop\CMU\4th Year\Chalarm\WLASL\wlasl_demo"
python -m venv .venv
.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install opencv-python numpy fastapi uvicorn python-multipart tqdm rich
```

✅ ตรวจสอบ:
```powershell
python scripts/healthcheck.py
```

ควรได้:
```
ffmpeg: OK
opencv: True
torch: True
OK: healthcheck passed
```

> ถ้า ffmpeg: MISSING → ดู STEP 1.4

---

## STEP 7 — รัน Backend API

เปิด **Terminal ที่ 1** (เปิดทิ้งไว้ตลอด ห้ามปิด):

```powershell
cd "C:\Users\bhumi\Desktop\CMU\4th Year\Chalarm\WLASL\wlasl_demo"
.venv\Scripts\activate
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

✅ ตรวจสอบ: เห็น
```
Uvicorn running on http://127.0.0.1:8000
```

---

## STEP 8 — รัน Frontend (Web UI)

เปิด **Terminal ที่ 2** (เปิดทิ้งไว้ตลอด ห้ามปิด):

```powershell
cd "C:\Users\bhumi\Desktop\CMU\4th Year\Chalarm\WLASL\wlasl_demo\web"
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

✅ ตรวจสอบ: เห็น
```
▲ Next.js 14.2.5
- Local: http://127.0.0.1:3000
✓ Ready in 2.1s
```

---

## STEP 9 — ทดสอบและดูผล

### 9.1 เปิด Browser

ไปที่: **http://127.0.0.1:3000**

ตรวจสอบ **API: OK** สีเขียว มุมบนขวา

### 9.2 ตั้งค่า

1. **ปลด checkbox Mock** ออก (untick)
2. คลิก **Advanced** ตรวจสอบ:
   - Weights path: `weights/model.ts`
   - Labels path: `weights/labels.json`

### 9.3 ทดสอบ Upload วิดีโอ

1. เตรียมวิดีโอภาษามือ ASL (.mp4 หรือ .webm ไม่เกิน 10 วินาที)
2. คลิก **Choose file** → เลือกวิดีโอ
3. คลิก **Upload & Infer**
4. ดูผลในตาราง — จะได้คำภาษาอังกฤษพร้อม % confidence:

```
| Rank | Label    | Score |
|------|----------|-------|
|  1   | hello    | 45.2% |
|  2   | book     | 23.1% |
|  3   | drink    | 12.8% |
|  4   | help     |  9.4% |
|  5   | computer |  6.1% |
```

### 9.4 ทดสอบผ่าน Webcam

1. กด **Start Recording**
2. ทำท่าภาษามือ 1 ท่า (1 คำ) ต่อกล้อง
3. กด **Stop Recording**
4. รอผลออกมาเป็นคำภาษาอังกฤษ

---

## ❗ Troubleshooting

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| Label เป็นตัวเลข (32, 12...) | ยังไม่ได้ fix labels | ทำ STEP 4.1 แล้ว download labels.json ใหม่ |
| `status=unknown low_confidence` | score ต่ำกว่า 50% | ปกติสำหรับโมเดลเล็ก ลอง train เพิ่ม epoch |
| `API: DOWN` | server ยังไม่ได้รัน | รัน uvicorn ใน Terminal 1 |
| `WEIGHTS_MISSING` | ไม่มีไฟล์ใน weights/ | ทำ STEP 5 |
| score ทุกอันเท่ากัน | Mock ยังติ๊กอยู่ | ปลด Mock ออก |
| `No module named 'src'` | ลืม cd เข้า repo ใน Colab | เพิ่ม `os.chdir('/content/wlasl')` |
| `classes=1` ตอน train | อ่าน dataset ผิด | ใช้ code ใน STEP 3.7 ที่แก้แล้ว |
| Webcam ไม่ทำงาน | browser block camera | กด Allow เมื่อ browser ถามสิทธิ์ |
| Colab session หมด | ทิ้งไว้นานเกิน | รัน Cell 4 ใหม่ตั้งแต่ต้น |
| ffmpeg: MISSING | ยังไม่ได้ติดตั้ง | ทำ STEP 1.4 |

---

## 📊 Checklist ตั้งแต่ต้นจนจบ

```
⬜ STEP 1  — ติดตั้ง Python, Node.js, Git, ffmpeg
⬜ STEP 2  — Clone repo
⬜ STEP 3  — Train ใน Colab (Cell 1-4)
⬜ STEP 4  — Fix labels + Export model.ts
⬜ STEP 5  — วาง weights/ บนเครื่อง
⬜ STEP 6  — ติดตั้ง Python packages
⬜ STEP 7  — รัน server (Terminal 1)
⬜ STEP 8  — รัน web (Terminal 2)
⬜ STEP 9  — เปิด http://127.0.0.1:3000 ทดสอบ Real mode
```
