#!/bin/bash

# AutoSOS Free Tier Cloud Deployment Script
# Optimized for free cloud platforms

set -e

echo "🆓 AutoSOS Free Tier Cloud Deployment"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp env.example .env
    echo "📝 Please edit .env file with your configuration before continuing."
    echo "   Required: SUPABASE_URL, SUPABASE_KEY"
    read -p "Press Enter to continue after editing .env file..."
fi

# Load environment variables
source .env

# Validate required environment variables
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo "❌ Missing required environment variables:"
    echo "   - SUPABASE_URL"
    echo "   - SUPABASE_KEY"
    echo "   Please check your .env file."
    exit 1
fi

echo "✅ Environment variables validated"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p nginx/ssl

# Build and start services with free tier optimizations
echo "🔨 Building and starting services (free tier optimized)..."
docker-compose -f docker-compose.free.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 45

# Check service health
echo "🏥 Checking service health..."

# Check API Gateway
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Gateway is healthy"
else
    echo "❌ API Gateway is not responding"
fi

# Check FaceNet service
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ FaceNet service is healthy"
else
    echo "❌ FaceNet service is not responding"
fi

# Check YOLOv8 service
if curl -f http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ YOLOv8 service is healthy"
else
    echo "❌ YOLOv8 service is not responding"
fi

# Check Ollama service
if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is healthy"
else
    echo "❌ Ollama service is not responding"
fi

# Check Redis
if docker-compose -f docker-compose.free.yml exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Show service URLs
echo ""
echo "🌐 Service URLs:"
echo "   API Gateway: http://localhost:8000"
echo "   FaceNet Service: http://localhost:8001"
echo "   YOLOv8 Service: http://localhost:8002"
echo "   Ollama Service: http://localhost:11434"
echo "   Redis: redis://localhost:6379"

# Show resource usage
echo ""
echo "📊 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Show logs
echo ""
echo "📋 To view logs:"
echo "   docker-compose -f docker-compose.free.yml logs -f [service-name]"
echo ""
echo "📋 To stop services:"
echo "   docker-compose -f docker-compose.free.yml down"
echo ""
echo "📋 To restart services:"
echo "   docker-compose -f docker-compose.free.yml restart [service-name]"

echo ""
echo "🎉 Free tier deployment completed!"
echo "   Your AutoSOS AI services are now running with minimal resource usage."
echo "   Perfect for free cloud platforms like Railway, Render, or Oracle Cloud!"
