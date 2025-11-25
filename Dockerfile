# GPU-enabled development image for ChromaDB + sentence-transformers
# Using official PyTorch runtime image (CUDA 12.1 + cuDNN9) for reliability
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace

# Copy requirements for layer caching
COPY requirements.txt ./

# Remove generic torch line (to avoid CPU wheel), install specific CUDA wheels, then remaining deps
RUN sed -i '/^torch$/d' requirements.txt && \
    pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch version:', torch.__version__)" && \
    python -c "import chromadb, sentence_transformers; print('Env ready')"

RUN mkdir -p /workspace/.cache/huggingface

CMD ["bash"]
