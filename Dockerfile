# =============================
# 1. Base image (lightweight & compatible with PyTorch)
# =============================
FROM python:3.10-slim

# =============================
# 2. Set working directory
# =============================
WORKDIR /app

# =============================
# 3. System dependencies
# =============================
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# =============================
# 4. Copy project files
# =============================
COPY . /app

# =============================
# 5. Install Python dependencies
# =============================
# (We install torch first, then others to avoid version conflicts)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
RUN pip install --no-cache-dir segmentation-models-pytorch==0.3.3
RUN pip install --no-cache-dir gradio==4.36.1
RUN pip install --no-cache-dir numpy==1.26.4 rasterio==1.4.3 matplotlib==3.10.7 scikit-learn==1.7.2 pillow==12.0.0

# =============================
# 6. Expose the Gradio port
# =============================
EXPOSE 7860

# =============================
# 7. Run the Gradio app
# =============================
CMD ["python", "app.py"]
