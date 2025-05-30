# utils/viz_utils.py
# --------------------------------------------------
# Funciones para graficar métricas y registrar resultados

import os
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_y_graficar(model, train_loader, val_loader, train_losses, val_losses, model_name="Modelo", output_dir="./"):
    """
    Evalúa el modelo, genera gráficas y guarda resultados en JSON e imágenes PNG.
    """

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

    num_epochs = len(train_losses)

    # === Gráfica MAE ===
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train MAE')
    plt.plot(val_losses, label='Val MAE')
    plt.title('Evolución del MAE')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'images/{model_name}_mae.png'))
    plt.show()

    # === Gráfica MSE ===
    mse_train = []
    mse_val = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_epochs):
            # Entrenamiento
            preds_train = []
            targets_train = []
            for inputs, targets in train_loader:
                outputs = model(inputs)
                preds_train.append(outputs.cpu())
                targets_train.append(targets.cpu())
            preds_train = torch.cat(preds_train).numpy()
            targets_train = torch.cat(targets_train).numpy()
            mse_train.append(mean_squared_error(targets_train, preds_train))

            # Validación
            preds_val = []
            targets_val = []
            for inputs, targets in val_loader:
                outputs = model(inputs)
                preds_val.append(outputs.cpu())
                targets_val.append(targets.cpu())
            preds_val = torch.cat(preds_val).numpy()
            targets_val = torch.cat(targets_val).numpy()
            mse_val.append(mean_squared_error(targets_val, preds_val))

    plt.figure(figsize=(10, 4))
    plt.plot(mse_train, label='Train MSE')
    plt.plot(mse_val, label='Val MSE')
    plt.title('Evolución del MSE')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'images/{model_name}_mse.png'))
    plt.show()

    # === Cálculo final y guardado en JSON ===
    val_mae = val_losses[-1]
    train_mae = train_losses[-1]
    val_mse = mse_val[-1]
    train_mse = mse_train[-1]
    r2 = r2_score(targets_val, preds_val)

    results = {
        "model_name": model_name,
        "epochs": num_epochs,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "val_r2": r2,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    results_path = os.path.join(output_dir, 'results', 'model_results.json')

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Resultados registrados en {results_path}")
    print(f"✅ Gráficas guardadas en {os.path.join(output_dir, 'images/')}")

def comparar_modelos(results_path='./results/model_results.json', filtro_nombre=None, ordenar_por='val_mae'):
    """
    Lee el archivo JSON con resultados y grafica una comparación entre modelos.

    Parámetros:
    - results_path: ruta al archivo JSON con resultados.
    - filtro_nombre: lista de nombres de modelos a incluir (o None para incluir todos).
    - ordenar_por: métrica por la cual ordenar ('val_mae', 'val_mse', 'val_r2', etc.)
    """

    import matplotlib.pyplot as plt
    import json
    import os

    # Validar existencia del archivo
    if not os.path.exists(results_path):
        print("❌ Archivo de resultados no encontrado:", results_path)
        return

    # Cargar resultados
    with open(results_path, 'r') as f:
        resultados = json.load(f)

    # Filtro opcional por nombre
    if filtro_nombre:
        resultados = [r for r in resultados if r['model_name'] in filtro_nombre]

    # Filtrar solo registros con métricas necesarias
    required_keys = ['val_mae', 'val_mse', 'val_r2']
    resultados = [r for r in resultados if all(k in r for k in required_keys)]

    if len(resultados) == 0:
        print("⚠️ No hay modelos con métricas completas para comparar.")
        return

    # Ordenar
    resultados.sort(key=lambda r: r.get(ordenar_por, float('inf')))

    # Extraer valores
    nombres  = [r['model_name'] for r in resultados]
    val_mae  = [r['val_mae'] for r in resultados]
    val_mse  = [r['val_mse'] for r in resultados]
    val_r2   = [r['val_r2'] for r in resultados]

    # Gráfico comparativo
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(nombres, val_mae, color='teal')
    plt.title('Val MAE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.bar(nombres, val_mse, color='orange')
    plt.title('Val MSE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.bar(nombres, val_r2, color='purple')
    plt.title('Val R²')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.suptitle('📊 Comparación de modelos registrados')
    plt.tight_layout()
    plt.show()

def exportar_modelos_csv(results_path='./results/model_results.json', ordenar_por='val_mae'):
    """
    Exporta el contenido del archivo JSON de resultados a un archivo CSV ordenado.

    Parámetros:
    - results_path: Ruta al archivo JSON
    - ordenar_por: Campo por el cual ordenar ('val_mae', 'val_r2', etc.)
    """

    import pandas as pd

    if not os.path.exists(results_path):
        print("❌ No se encontró el archivo:", results_path)
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("⚠️ No hay resultados para exportar.")
        return

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Ordenar si la columna existe
    if ordenar_por in df.columns:
        df = df.sort_values(by=ordenar_por, ascending=True)

    # Exportar
    output_csv = './results/model_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"✅ Resultados exportados a CSV en: {output_csv}")

import json
import matplotlib.pyplot as plt
import numpy as np

def graficar_comparacion_modelos(resultados_path="./results/model_results.json"):
    """
    Carga resultados desde un JSON y grafica una comparación de MAE, MSE y R² entre modelos.
    """
    with open(resultados_path, "r") as f:
        resultados = json.load(f)

    modelos = list(resultados.keys())
    maes = [resultados[m]["mae"] for m in modelos]
    mses = [resultados[m]["mse"] for m in modelos]
    r2s  = [resultados[m]["r2"]  for m in modelos]

    x = np.arange(len(modelos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, maes, width, label="MAE")
    ax.bar(x, mses, width, label="MSE")
    ax.bar(x + width, r2s, width, label="R²")

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha="right")
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas por modelo")
    ax.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
