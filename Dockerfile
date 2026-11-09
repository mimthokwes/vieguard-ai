# Menggunakan base image Python resmi yang ringan dan mendukung ARM64 & AMD64
FROM python:3.11-slim

# Menghindari pembuatan file .pyc dan memastikan log Python langsung tercetak
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Direktori kerja di dalam kontainer
WORKDIR /app

# Salin dependensi terlebih dahulu agar memanfaatkan cache Docker
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source code proyek
COPY . .

# Pastikan folder knowledge dan index ada
RUN mkdir -p knowledge index

# Port default service AI
EXPOSE 5001

# Jalankan server
CMD ["python", "main.py"]
