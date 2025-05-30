# utils/init_utils.py
# --------------------------------------------------
# Utilidades de inicialización para el proyecto FWOD

import os
import random
import numpy as np
import torch
import zipfile
from datetime import datetime
import shutil

# Ruta al archivo de dataset (.npz)
DATASET_PATH = './data/mel_fwod_dataset_variable.npz'

# Directorios de salida requeridos
OUTPUT_DIRS = ['./checkpoints', './estado', './images', './results']

def set_seed(seed: int = 42):
    """
    Configura la semilla global para garantizar reproducibilidad.
    Aplica a random, numpy y torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✅ Semilla global establecida: {seed}")

def get_device():
    """
    Detecta si CUDA está disponible y retorna el dispositivo correspondiente.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo seleccionado: {device}")
    return device

def prepare_directories(files_action: str = "delete"):
    """
    Verifica y crea los directorios necesarios para guardar modelos, imágenes y resultados.
    Si files_action="delete", borra archivos existentes.
    Si files_action="backup", genera un respaldo en un zip con timestamp y luego los borra.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_name = f"./backup_{timestamp}.zip"
    backup_created = False

    for folder in OUTPUT_DIRS:
        os.makedirs(folder, exist_ok=True)
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        if files_action == "backup" and files:
            with zipfile.ZipFile(backup_name, 'a', zipfile.ZIP_DEFLATED) as zipf:
                for file in files:
                    path = os.path.join(folder, file)
                    zipf.write(path, arcname=os.path.join(folder[2:], file))  # relative path in zip
                    os.remove(path)
            backup_created = True

        elif files_action == "delete":
            for file in files:
                os.remove(os.path.join(folder, file))

    print("✅ Carpetas de trabajo verificadas.")
    if backup_created:
        print(f"🗂️  Backup creado: {backup_name}")

def init_all(seed: int = 42, files_action: str = "backup"):
    """
    Inicializa el entorno completo: semilla, device, y estructura de carpetas.

    Parámetros:
        seed: Semilla para reproducibilidad.
        files_action: "delete" para borrar archivos, "backup" para respaldar y borrar.

    Retorna:
        device (torch.device): El dispositivo configurado (CPU o CUDA).
    """
    set_seed(seed)
    device = get_device()
    prepare_directories(files_action)
    return device
