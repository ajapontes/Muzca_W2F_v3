
import torch
import json
import os

def guardar_estado(modelo, optimizer, model_name, train_losses, val_losses, ruta_base="./estado", extras=None):
    """
    Guarda el estado completo del entrenamiento para continuar más adelante.

    Parámetros:
    - modelo: modelo PyTorch
    - optimizer: optimizador PyTorch
    - model_name: identificador del modelo
    - train_losses / val_losses: listas de errores por época
    - ruta_base: carpeta base para guardar los archivos
    - extras: diccionario opcional con info adicional (ej: lr, batch_size)
    """
    os.makedirs(ruta_base, exist_ok=True)

    # Guardar pesos del modelo y estado del optimizador
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(ruta_base, f"{model_name}.pt"))

    # Guardar JSON con métricas y extras
    estado = {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "extras": extras or {}
    }

    with open(os.path.join(ruta_base, f"{model_name}_estado.json"), "w") as f:
        json.dump(estado, f, indent=4)

    print(f"✅ Estado guardado en '{ruta_base}/'")

def cargar_estado(modelo_clase, optimizer_clase, model_name, ruta_base="./estado", device="cpu", optimizer_args=None):
    """
    Restaura modelo y optimizador desde archivo, junto con métricas previas.

    Retorna:
    - modelo (con pesos)
    - optimizer (con estado)
    - train_losses
    - val_losses
    - extras (diccionario)
    """
    ruta_modelo = os.path.join(ruta_base, f"{model_name}.pt")
    ruta_json = os.path.join(ruta_base, f"{model_name}_estado.json")

    # Instanciar modelo y optimizador vacíos
    modelo = modelo_clase().to(device)
    optimizer = optimizer_clase(modelo.parameters(), **(optimizer_args or {}))

    # Cargar checkpoint
    checkpoint = torch.load(ruta_modelo, map_location=device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    modelo.eval()

    # Cargar estado json
    with open(ruta_json, "r") as f:
        estado = json.load(f)

    print(f"✅ Estado restaurado desde '{ruta_base}/'")

    return modelo, optimizer, estado["train_losses"], estado["val_losses"], estado.get("extras", {})
