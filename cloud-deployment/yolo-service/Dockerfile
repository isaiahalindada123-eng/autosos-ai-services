# Use Ubuntu 20.04 with Python 3.9 for better OpenCV support
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    python3.9-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgfortran5 \
    libatlas3-base \
    libhdf5-103 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Install Python dependencies
COPY requirements.minimal.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir typing-extensions==4.5.0 && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models
RUN mkdir -p /app/models

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]