# utils/optuna_utils.py
# -----------------------------------------------------
# Funciones de utilidad para manejar resultados de Optuna

import pandas as pd
import os
import optuna
import torch
import torch.nn as nn
from datetime import datetime
from utils.viz_utils import evaluar_y_graficar
from utils.model_utils import build_dynamic_cnn_model
from utils.train_utils import entrenar_modelo

def crear_study_y_ejecutar(objective_func, direction='minimize', n_trials=30, early_stop_patience=None):
    """
    Ejecuta una búsqueda bayesiana con Optuna, incluyendo early stopping opcional.

    Parámetros:
    - objective_func: función objetivo a minimizar
    - direction: 'minimize' o 'maximize'
    - n_trials: número máximo de pruebas
    - early_stop_patience: si se especifica, aplica early stopping manual por número de pruebas sin mejora

    Retorna:
    - study: el objeto study resultante
    """
    study = optuna.create_study(direction=direction)

    if early_stop_patience is None:
        study.optimize(objective_func, n_trials=n_trials)
    else:
        best_score = float("inf") if direction == 'minimize' else float("-inf")
        no_improve = 0

        for trial in range(n_trials):
            try:
                study.optimize(objective_func, n_trials=1)
                current_best = study.best_value
                improved = (direction == 'minimize' and current_best < best_score) or                            (direction == 'maximize' and current_best > best_score)

                if improved:
                    best_score = current_best
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= early_stop_patience:
                    print(f"⛔ Early stopping: {no_improve} pruebas sin mejora.")
                    break
            except optuna.exceptions.TrialPruned:
                continue

    return study

def guardar_historial_optuna(study, model_name="modelo", output_dir="./results"):
    """
    Guarda el historial completo de todos los trials de un estudio Optuna en un archivo CSV.

    Parámetros:
    - study: instancia de optuna.Study después de ejecutarse.
    - model_name: nombre del modelo asociado (ej: 'cnn_deep')
    - output_dir: carpeta donde guardar el CSV

    El archivo se llamará: optuna_trials_<model_name>.csv
    """
    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Crear nombre de archivo
    output_path = os.path.join(output_dir, f"optuna_trials_{model_name}.csv")

    # Convertir trials a DataFrame
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(output_path, index=False)

    print(f"✅ Historial completo de trials guardado en '{output_path}'")
    
def crear_objective_optuna(modelo_id, train_loader, val_loader, device):
    """
    Crea una función objetivo compatible con Optuna para búsqueda de hiperparámetros.

    Parámetros:
    - modelo_id: identificador único del modelo para los nombres de archivos.
    - train_loader: DataLoader de entrenamiento.
    - val_loader: DataLoader de validación.
    - device: CPU o CUDA

    Retorna:
    - función objetivo parametrizada.
    """
    def objective(trial):
        posibles_filtros = [16, 32, 64, 128]
        max_capas = 5

        num_layers = trial.suggest_int("num_conv_layers", 3, max_capas)
        filters = [
            trial.suggest_categorical(f"filters_l{i}", posibles_filtros)
            for i in range(num_layers)
        ]
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        use_dropout   = trial.suggest_categorical("use_dropout", [True, False])
        dropout_rate  = trial.suggest_float("dropout_rate", 0.1, 0.5)
        activation    = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"])
        pool_type     = trial.suggest_categorical("pool_type", ["max", "avg"])
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        try:
            model = build_dynamic_cnn_model(
                num_layers=num_layers,
                filters=filters,
                activation=activation,
                use_batchnorm=use_batchnorm,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                pool_type=pool_type
            ).to(device)
            model.device = device

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.L1Loss()

            train_losses, val_losses = entrenar_modelo(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                num_epochs=10,
                model_name=f"trial_{trial.number}_{modelo_id}",
                guardar=False
            )
            return val_losses[-1]

        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"⚠️ Trial {trial.number} falló por error CUDA (omitido para estabilidad).")
                return float("inf")
            raise

    return objective