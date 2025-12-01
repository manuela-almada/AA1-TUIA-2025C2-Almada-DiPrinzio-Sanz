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

### Para ver distribuciones disponibles
```powershell
wsl --list --online
```

### Instalar Ubuntu 22.04
```powershell
wsl --install -d Ubuntu-22.04
```

# 🐳 2. Construir la imagen Docker 
### Desde la carpeta donde se encuentre Dockerfile
```powershell
docker build -t rain-prediction-inference .
```

# ▶️ 3. Ejecutar el contenedor con inferencia
### Debe contar con un archivo CSV con los datos de entrada.
```powershell
docker run -it --rm --name rain-container -v "${PWD}\files:/files" rain-prediction-inference
```

El script inferencia.py leerá automáticamente /files/input.csv
y en archivo output.csv tendrá los resultados de la prediccion de lluvia (1 = lluvia, 0 = no lluvia)

# 🔁 4. Reiniciar WSL (si es necesario, antes de correr el Docker)
```powershell
wsl --shutdown
wsl --update
```

# 📁 5. Estructura del proyecto dentro de docker/
docker/

│── inferencia.py

│── pipeline.pkl

│── requirements.txt

│── Dockerfile






