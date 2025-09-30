FROM ollama/ollama:latest

# Install additional dependencies for AutoSOS
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create directory for custom models
RUN mkdir -p /root/.ollama/models

# Copy model installation script
COPY install-models.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/install-models.sh

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# Start Ollama and install models
CMD ["sh", "-c", "/usr/local/bin/install-models.sh && /bin/ollama serve"]
