#!/bin/bash

# Start Ollama in the background
/bin/ollama serve &

# Wait for Ollama to start
sleep 10

# Install models
echo "Installing Ollama models..."

# Install Llama 3.2 3B (lightweight model for cloud)
/bin/ollama pull llama3.2:3b

# Install other useful models
/bin/ollama pull llama3.2:1b

echo "Models installed successfully!"

# Keep the container running
wait
