# Usa una imagen base adecuada con Python.  Ajusta según tus necesidades.
FROM registry.hf.space/xfcxcxcdfdfd-hhhhvasasasasdsddsdsxxxxxxxxxxxxx:tpu-05109ee-xgpoa8pz

# Establece la variable de entorno para usar la GPU (necesaria si uses CUDA)
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# Instala las dependencias.  Reemplaza `pip install -r requirements.txt` con tus requisitos.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia tu aplicación
COPY app.py ./

# Define el puerto
EXPOSE 7860

# Comando para ejecutar tu aplicación
#CMD ["python", "app.py"]
