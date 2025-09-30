# Render Environment Variables Setup

## Required Environment Variables for AutoSOS Services

### FaceNet Service
Add these environment variables in your Render service settings:

```
SUPABASE_URL=https://atdibhoeaeqfgjswcqwx.supabase.co
SUPABASE_KEY=sb_publishable_8zWSuqsDoSKDiWkz3Yd_eg_E7N1X7oj
```

### YOLOv8 Service
Add these environment variables in your Render service settings:

```
SUPABASE_URL=https://atdibhoeaeqfgjswcqwx.supabase.co
SUPABASE_KEY=sb_publishable_8zWSuqsDoSKDiWkz3Yd_eg_E7N1X7oj
```

### API Gateway Service
Add these environment variables in your Render service settings:

```
REDIS_URL=redis://red-d3e380je5dus73fdhdp0:6379
FACENET_SERVICE_URL=https://your-facenet-service.onrender.com
YOLO_SERVICE_URL=https://your-yolo-service.onrender.com
OLLAMA_SERVICE_URL=https://your-ollama-service.onrender.com
```

## How to Add Environment Variables in Render

1. **Go to your service** in Render Dashboard
2. **Click on "Environment"** tab
3. **Click "Add Environment Variable"**
4. **Enter the key and value** as shown above
5. **Click "Save Changes"**
6. **Redeploy the service** (Render will auto-redeploy)

## Supabase Key Types

### Anon Key (Public)
- Starts with `sb_publishable_`
- Used for client-side operations
- Limited permissions

### Service Role Key (Private)
- Starts with `eyJ...` (JWT token)
- Used for server-side operations
- Full database access
- **Use this for your services**

## Getting Your Service Role Key

1. Go to **Supabase Dashboard**
2. Navigate to **Settings** → **API**
3. Copy the **service_role** key (not the anon key)
4. Use this key in your Render environment variables

## Troubleshooting

### "Invalid API key" Error
- Make sure you're using the **service_role** key, not the anon key
- Check that the key is copied correctly (no extra spaces)
- Verify the Supabase URL is correct

### "Row Level Security" Error
- Run the SQL script: `fix-supabase-rls-simple.sql`
- This disables RLS for service access

### Service Won't Start
- Check all environment variables are set
- Verify the service URLs are correct
- Check Render logs for specific errors
