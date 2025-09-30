#!/bin/bash

# AutoSOS Railway Deployment Script
# Deploy your AI services to Railway cloud platform

set -e

echo "🚂 AutoSOS Railway Deployment"
echo "============================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway..."
    railway login
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

# Create Railway project
echo "🚂 Creating Railway project..."
if [ -z "$RAILWAY_PROJECT_ID" ]; then
    railway project new autosos-ai-services
    echo "📝 Please note your Railway project ID for future reference"
else
    railway project use $RAILWAY_PROJECT_ID
fi

# Add Redis service
echo "🗄️ Adding Redis service..."
railway add redis

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set SUPABASE_URL="$SUPABASE_URL"
railway variables set SUPABASE_KEY="$SUPABASE_KEY"
railway variables set RAILWAY_ENVIRONMENT="production"

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "⏳ Waiting for deployment to complete..."
sleep 30

# Get service URL
SERVICE_URL=$(railway domain)
echo ""
echo "🎉 Deployment completed!"
echo "🌐 Your AutoSOS AI services are now available at:"
echo "   $SERVICE_URL"
echo ""
echo "📋 Service endpoints:"
echo "   API Gateway: $SERVICE_URL/api/"
echo "   Health Check: $SERVICE_URL/health"
echo "   FaceNet: $SERVICE_URL/api/facenet/"
echo "   YOLOv8: $SERVICE_URL/api/yolo/"
echo "   Ollama: $SERVICE_URL/api/ollama/"
echo ""
echo "📊 Monitor your deployment:"
echo "   railway logs"
echo "   railway status"
echo ""
echo "🔧 Update your client app:"
echo "   Update CLOUD_BASE_URL to: $SERVICE_URL"
echo ""
echo "💰 Railway usage:"
echo "   Check your usage at: https://railway.app/dashboard"
echo "   Free tier includes $5 credit monthly"
