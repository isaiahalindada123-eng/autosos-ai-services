#!/bin/bash

echo "🚂 Installing Ollama models for Railway deployment..."

# Start Ollama in background
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 15

# Install only the smallest model for Railway free tier
echo "📦 Installing llama3.2:1b (Railway optimized)..."
ollama pull llama3.2:1b

# Create Railway-optimized motorcycle diagnostic model
echo "🔧 Creating Railway motorcycle diagnostic model..."
cat > /tmp/motorcycle-diagnostic-railway << 'EOF'
FROM llama3.2:1b

SYSTEM """You are a motorcycle mechanic expert. Provide concise, helpful diagnostic advice in JSON format:

{
  "analysis": "Brief problem analysis",
  "diagnosis": {
    "issue": "Main issue identified",
    "severity": "Low|Medium|High|Critical",
    "recommendation": "Specific action to take",
    "immediate_actions": ["Action 1", "Action 2"],
    "long_term_solutions": ["Solution 1", "Solution 2"],
    "safety_warning": "Safety concern if applicable"
  },
  "follow_up_questions": ["Question 1", "Question 2"]
}

Keep responses practical and concise. Always respond in valid JSON format."""
EOF

ollama create motorcycle-diagnostic-railway -f /tmp/motorcycle-diagnostic-railway

echo "✅ Railway model installation completed!"
echo "📋 Available models:"
ollama list

# Keep Ollama running
wait $OLLAMA_PID
