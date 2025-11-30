# 🌧️ MLOps – Rain Prediction Inference (Docker + WSL)

Este proyecto incluye un contenedor Docker capaz de ejecutar inferencias utilizando
el pipeline entrenado para predecir **RainTomorrow**.

El contenedor está preparado para ejecutarse en **Windows con WSL2**.

---

# 🧩 1. Requisitos previos

### ✔ Tener WSL habilitado  
```powershell
wsl --update
wsl --shutdown
wsl
```

Para ver distribuciones disponibles
wsl --list --online

Instalar Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 🐳 2. Construir la imagen Docker 
### Desde la carpeta donde se encuentre Dockerfile
docker build -t rain-prediction-inference .

# ▶️ 3. Ejecutar el contenedor con inferencia
### Debe contar con un archivo JSON con los datos de entrada.
docker run -it --rm --name rain-container -v "${PWD}\files:/files" rain-prediction-inference

El script inferencia.py leerá automáticamente /files/input.json
y devolverá una salida similar a:
{
    "prediction": 1,
    "probability": 0.72
}

# 🔁 4. Reiniciar WSL (si es necesario, antes de correr el Docker)

wsl --shutdown
wsl --update

# 📁 5. Estructura del proyecto dentro de docker/
docker/
│── inferencia.py
│── pipeline.pkl
│── requirements.txt
│── Dockerfile
│── README.md






