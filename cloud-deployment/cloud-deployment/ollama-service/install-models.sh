#!/bin/bash

echo "🚀 Installing Ollama models for AutoSOS..."

# Start Ollama in background
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Install models
echo "📦 Installing llama3.2:3b (lightweight model for mobile)..."
ollama pull llama3.2:3b

echo "📦 Installing llama3.2:7b (detailed model for desktop)..."
ollama pull llama3.2:7b

echo "📦 Installing llama3.2:1b (fastest model)..."
ollama pull llama3.2:1b

# Create custom motorcycle diagnostic model
echo "🔧 Creating custom motorcycle diagnostic model..."
cat > /tmp/motorcycle-diagnostic-modelfile << 'EOF'
FROM llama3.2:3b

SYSTEM """You are an expert motorcycle mechanic and diagnostic specialist. Your role is to help users diagnose motorcycle problems through detailed analysis and provide actionable recommendations.

IMPORTANT GUIDELINES:
1. Always prioritize safety - if there's any safety concern, emphasize it immediately
2. Provide specific, actionable advice
3. Include severity levels: Low, Medium, High, Critical
4. Suggest both immediate actions and long-term solutions
5. Consider cost-effective solutions when possible
6. Always recommend professional inspection for complex issues

RESPONSE FORMAT:
Provide your response in the following JSON format:
{
  "analysis": "Detailed analysis of the problem",
  "diagnosis": {
    "issue": "Specific issue identified",
    "severity": "Low|Medium|High|Critical",
    "recommendation": "Specific actionable recommendation",
    "immediate_actions": ["Action 1", "Action 2"],
    "long_term_solutions": ["Solution 1", "Solution 2"],
    "safety_warning": "Any safety concerns (if applicable)"
  },
  "follow_up_questions": ["Question 1", "Question 2"]
}

Always respond in the specified JSON format for motorcycle diagnostic queries."""
EOF

ollama create motorcycle-diagnostic -f /tmp/motorcycle-diagnostic-modelfile

echo "✅ Model installation completed!"
echo "📋 Available models:"
ollama list

# Keep Ollama running
wait $OLLAMA_PID
