import torch
import torch.nn.functional as F
from tqdm import tqdm

def entrenar_modelo(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, model_name="modelo", guardar=False):
    """
    Entrena un modelo PyTorch y registra métricas por época.

    Retorna:
    - train_losses: lista MAE por época en entrenamiento
    - val_losses: lista MAE por época en validación
    """

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_mae = 0.0
        running_train_mse = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # ✅ Validación de dispositivos
            assert outputs.device == targets.device, (
                f"Dispositivos incompatibles en entrenamiento: outputs={outputs.device}, targets={targets.device}"
            )

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_mae += F.l1_loss(outputs, targets, reduction='sum').item()
            running_train_mse += F.mse_loss(outputs, targets, reduction='sum').item()

        train_mae = running_train_mae / len(train_loader.dataset)
        train_mse = running_train_mse / len(train_loader.dataset)
        train_losses.append(train_mae)

        model.eval()
        running_val_mae = 0.0
        running_val_mse = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs)

                # ✅ Validación de dispositivos
                assert outputs.device == targets.device, (
                    f"Dispositivos incompatibles en validación: outputs={outputs.device}, targets={targets.device}"
                )

                running_val_mae += F.l1_loss(outputs, targets, reduction='sum').item()
                running_val_mse += F.mse_loss(outputs, targets, reduction='sum').item()

        val_mae = running_val_mae / len(val_loader.dataset)
        val_mse = running_val_mse / len(val_loader.dataset)
        val_losses.append(val_mae)

        print(f"\n📉 Epoch {epoch+1}/{num_epochs}")
        print(f"   🔹 Train MAE: {train_mae:.4f} | Train MSE: {train_mse:.4f}")
        print(f"   🔸 Val   MAE: {val_mae:.4f} | Val   MSE: {val_mse:.4f}")

    if guardar:
        torch.save(model.state_dict(), f"./checkpoints/{model_name}.pt")
        print(f"✅ Modelo guardado como './checkpoints/{model_name}.pt'")

    return train_losses, val_losses
