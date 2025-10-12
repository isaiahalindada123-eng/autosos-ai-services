# Mapbox Navigation Service

A secure server-side service for handling Mapbox navigation requests using secret tokens.

## Features

- **Directions API**: Get driving directions between two points
- **Geocoding API**: Convert addresses to coordinates and vice versa
- **Distance Matrix API**: Calculate distances between multiple points
- **CORS Enabled**: Works with client-side applications
- **Secure**: Uses secret tokens server-side only

## Environment Variables

Create a `.env` file with:

```env
MAPBOX_SECRET_TOKEN=sk.your_secret_token_here
PORT=5000
FLASK_ENV=production
```

## API Endpoints

### Health Check
```
GET /health
```

### Get Directions
```
POST /api/directions
Content-Type: application/json

{
  "origin_lat": 14.5995,
  "origin_lng": 120.9842,
  "destination_lat": 14.6042,
  "destination_lng": 120.9822
}
```

### Geocoding
```
POST /api/geocoding
Content-Type: application/json

{
  "query": "Manila, Philippines"
}
```

### Distance Matrix
```
POST /api/matrix
Content-Type: application/json

{
  "coordinates": [
    {"lat": 14.5995, "lng": 120.9842},
    {"lat": 14.6042, "lng": 120.9822}
  ]
}
```

## Deployment to Render

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy as a Web Service
4. Use the provided URL in your client applications

## Security

- Secret tokens are never exposed to client-side
- CORS is properly configured
- Input validation on all endpoints
- Error handling for API failures
