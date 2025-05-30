import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt
import json

import sys
from pathlib import Path

# Agrega el directorio raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.model_utils import CNNBaseline

# ============================
# Configuración de entorno
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de cómputo: {device}")

# ============================
# Paso 1: Cargar el modelo entrenado
# ============================
model = CNNBaseline()
model.load_state_dict(torch.load("./checkpoints/cnn_baseline.pt", map_location=device))
model = model.to(device)
model.eval()
print("[INFO] Modelo cargado correctamente.")

# ============================
# Paso 2: Cargar el dataset original
# ============================
data_path = "./data/mel_fwod_dataset_variable.npz"
data = np.load(data_path, allow_pickle=True)
mel = data["mel"]
meta = data["meta"]
print(f"[INFO] Datos cargados. Total de muestras: {len(mel)}")

# ============================
# Paso 3: Preprocesar los datos
# ============================
# Añadir dimensión de canal para CNN (Bx1xHxW)
mel_tensor = [torch.tensor(m, dtype=torch.float32).unsqueeze(0) for m in mel]
X = torch.stack(mel_tensor)
X = X.to(device)

# ============================
# Paso 4: Predecir con el modelo
# ============================
batch_size = 64
dataloader = DataLoader(X, batch_size=batch_size)

all_preds = []
with torch.no_grad():
    for batch in dataloader:
        preds = model(batch)
        all_preds.append(preds.cpu())

y_pred = torch.cat(all_preds, dim=0).numpy()

# ============================
# Paso 5: Guardar el nuevo dataset
# ============================
output_file = "./data/mel_fwod_predicted.npz"
backup_dir = "./data/backups"
os.makedirs(backup_dir, exist_ok=True)

if os.path.exists(output_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"mel_fwod_predicted_backup_{timestamp}.zip")
    with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname="mel_fwod_predicted.npz")
    os.remove(output_file)
    print(f"[INFO] Archivo respaldado como {backup_file}")

np.savez(output_file, fwod=y_pred, meta=meta)
print(f"[INFO] Archivo nuevo generado y guardado en {output_file}")

# ============================
# Paso 6: Guardar métricas del modelo
# ============================
stats = {
    "total_muestras": len(mel),
    "shape_fwod_pred": list(y_pred.shape),
    "modelo_usado": "cnn_baseline.pt",
    "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device)
}

metrics_path = "./results/generacion_fwod_log.json"
os.makedirs("./results", exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(stats, f, indent=4)

print(f"[INFO] Métricas guardadas en {metrics_path}")

# ============================
# Paso 7: Graficar distribución de un componente
# ============================
plt.figure(figsize=(10, 4))
plt.hist(y_pred[:, 0], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribución de la componente y_0 del vector FWOD predicho")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("./results/histograma_y0.png")
print("[INFO] Histograma de la componente y_0 guardado como imagen.")

# Mostrar datos finales al usuario
import ace_tools as tools; tools.display_dataframe_to_user(name="Preview FWOD generado", dataframe=np.round(y_pred[:5], 2))