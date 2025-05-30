# utils/model_utils.py
# --------------------------------------------------
# Arquitecturas de redes neuronales convolucionales para predicción de FWOD

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    Red neuronal convolucional básica para predecir vectores FWOD (1x16)
    a partir de espectrogramas Mel (HxW).
    """
    def __init__(self, input_shape=(128, 16)):
        super(CNNBaseline, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # ajusta el tamaño final sin hardcode

        # Cálculo de tamaño final del flatten
        dummy_input = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            out = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))
            flatten_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNNDropout(nn.Module):
    """
    Variante de CNNBaseline con capas Dropout para regularización.
    Predice vectores FWOD (1x16) a partir de espectrogramas Mel.
    """
    def __init__(self, input_shape=(128, 16), dropout_prob=0.3):
        super(CNNDropout, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        # Cálculo automático del tamaño del flatten
        dummy_input = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            out = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))
            flatten_size = out.view(1, -1).shape[1]

        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class CNNBatchNorm(nn.Module):
    """
    Modelo CNN con Batch Normalization después de cada capa convolucional.
    Predice vectores FWOD (1x16) a partir de espectrogramas Mel.
    """
    def __init__(self, input_shape=(128, 16)):
        super(CNNBatchNorm, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # Fijo tamaño para salida FC

        # Calcular tamaño del flatten automáticamente
        dummy_input = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            out = self.pool3(self.bn3(self.conv3(self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(dummy_input)))))))))
            flatten_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNNDeep(nn.Module):
    """
    Modelo CNN profundo con BatchNorm para predicción de FWOD.
    """
    def __init__(self, input_shape=(128, 16)):
        super(CNNDeep, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool_final = nn.AdaptiveAvgPool2d((2, 2))

        dummy = torch.zeros(1, 1, *input_shape)
        out = self.pool_final(self.bn5(self.conv5(
            F.relu(self.bn4(self.conv4(
                self.pool3(self.bn3(self.conv3(
                    self.pool2(self.bn2(self.conv2(
                        self.pool1(self.bn1(self.conv1(dummy)))
                    )))
                )))
            )))
        )))
        self.flatten_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool_final(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

##################################################
import torch.nn as nn
import torch.nn.functional as F

class DynamicCNN(nn.Module):
    def __init__(self, num_layers, filters, activation, use_batchnorm,
                 use_dropout, dropout_rate, pool_type, input_shape=(128, 16)):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activ_name = activation
        self.use_dropout = use_dropout
        self.pool_type = pool_type

        in_channels = 1  # siempre parte de 1 canal (espectrograma Mel)
        for i in range(num_layers):
            out_channels = filters[i]
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.layers.append(conv)

            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))

            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "GELU":
                self.layers.append(nn.GELU())

            if pool_type == "max":
                self.layers.append(nn.MaxPool2d(2))
            elif pool_type == "avg":
                self.layers.append(nn.AvgPool2d(2))

            if use_dropout:
                self.layers.append(nn.Dropout2d(dropout_rate))

            in_channels = out_channels

        # Adaptive pooling final
        self.layers.append(nn.AdaptiveAvgPool2d((2, 2)))

        # Calcular tamaño para capa lineal
        dummy = torch.zeros(1, 1, *input_shape)
        x = dummy
        for layer in self.layers:
            x = layer(x)
        self.flatten_size = x.view(1, -1).shape[1]

        # Fully connected
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def build_dynamic_cnn_model(num_layers, filters, activation,
                            use_batchnorm, use_dropout, dropout_rate,
                            pool_type):
    """
    Construye una instancia del modelo CNN dinámico con los parámetros especificados.
    """
    return DynamicCNN(
        num_layers=num_layers,
        filters=filters,
        activation=activation,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        pool_type=pool_type
    )
    
# [SE MANTIENEN TODAS LAS DEFINICIONES ANTERIORES: CNNBaseline, CNNDropout, CNNBatchNorm, CNNDeep]
# OMITIDO POR BREVIDAD — SE DEBEN MANTENER COMO ESTÁN EN TU ARCHIVO ACTUAL

# -------------------------------------
# NUEVA VERSIÓN CORREGIDA DE DYNAMICCNN
# -------------------------------------
class DynamicCNN(nn.Module):
    def __init__(self, num_layers, filters, activation, use_batchnorm,
                 use_dropout, dropout_rate, pool_type, input_shape=(128, 16)):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activ_name = activation
        self.use_dropout = use_dropout
        self.pool_type = pool_type

        in_channels = 1  # entrada con 1 canal (Mel Spectrogram)
        for i in range(num_layers):
            out_channels = filters[i]
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))

            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "GELU":
                self.layers.append(nn.GELU())

            # Pooling solo en las primeras 3 capas
            if i < 3:
                if pool_type == "max":
                    self.layers.append(nn.MaxPool2d(2))
                elif pool_type == "avg":
                    self.layers.append(nn.AvgPool2d(2))

            if use_dropout:
                self.layers.append(nn.Dropout2d(dropout_rate))

            in_channels = out_channels

        # Adaptive pooling final
        self.layers.append(nn.AdaptiveAvgPool2d((2, 2)))

        # Cálculo del tamaño para la capa densa
        dummy = torch.zeros(1, 1, *input_shape)
        x = dummy
        for layer in self.layers:
            x = layer(x)
        self.flatten_size = x.view(1, -1).shape[1]

        # Capa totalmente conectada
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def build_dynamic_cnn_model(num_layers, filters, activation,
                            use_batchnorm, use_dropout, dropout_rate,
                            pool_type):
    return DynamicCNN(
        num_layers=num_layers,
        filters=filters,
        activation=activation,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        pool_type=pool_type
    )