# 📘 Proyecto Tesis IA Ritmo `mel → fwod`

## 🧠 Descripción

Este proyecto entrena y compara diferentes modelos de redes neuronales convolucionales (CNN) que predicen vectores FWOD a partir de espectrogramas Mel extraídos de archivos `.wav` del dataset de Magenta.

---

## ✅ Cómo cerrar el entorno de forma segura

1. **Guardar notebook**
   - Asegúrate de que el archivo `.ipynb` esté guardado (`Ctrl + S`)
   - Ejecuta la última celda para persistir resultados

2. **Verifica archivos clave**
   - Modelos entrenados: `./checkpoints/*.pt`
   - Resultados: `./results/model_results.json`
   - Imágenes: `./images/*.png`
   - Código modular: `./utils/` (`init_utils.py`, `model_utils.py`, `viz_utils.py`)
   - Librerías instaladas: `requirements.txt`

3. **(Opcional)** Limpiar entorno si usas Jupyter o VSCode:
   - Cierra el kernel / reinicia entorno
   - Cierra Visual Studio Code o JupyterLab

---

## 🚀 Cómo retomar el proyecto

1. **Activar entorno virtual**

```bash
cd C:\\Proyectos\\Tesis\\Ver 3
.venv\\Scripts\\activate
