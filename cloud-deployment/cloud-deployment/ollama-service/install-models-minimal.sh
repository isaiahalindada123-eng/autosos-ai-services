#!/bin/bash

echo "🚀 Installing minimal Ollama models for free tier..."

# Start Ollama in background
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Install only the smallest model for free tier
echo "📦 Installing llama3.2:1b (fastest, smallest model)..."
ollama pull llama3.2:1b

# Create minimal motorcycle diagnostic model
echo "🔧 Creating minimal motorcycle diagnostic model..."
cat > /tmp/motorcycle-diagnostic-minimal << 'EOF'
FROM llama3.2:1b

SYSTEM """You are a motorcycle mechanic. Provide brief, helpful diagnostic advice in JSON format:

{
  "analysis": "Brief problem analysis",
  "diagnosis": {
    "issue": "Main issue",
    "severity": "Low|Medium|High|Critical",
    "recommendation": "What to do",
    "immediate_actions": ["Action 1", "Action 2"],
    "safety_warning": "Safety concern if any"
  }
}

Keep responses concise and practical."""
EOF

ollama create motorcycle-diagnostic-minimal -f /tmp/motorcycle-diagnostic-minimal

echo "✅ Minimal model installation completed!"
echo "📋 Available models:"
ollama list

# Keep Ollama running
wait $OLLAMA_PID
