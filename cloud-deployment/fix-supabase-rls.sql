-- Fix Supabase RLS policies for AutoSOS models
-- Run this in your Supabase SQL Editor

-- First, let's check if the models table exists and create it if needed
CREATE TABLE IF NOT EXISTS public.models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50),
    file_path TEXT NOT NULL,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ensure the sequence exists for models
CREATE SEQUENCE IF NOT EXISTS public.models_id_seq;
ALTER TABLE public.models ALTER COLUMN id SET DEFAULT nextval('public.models_id_seq');
ALTER SEQUENCE public.models_id_seq OWNED BY public.models.id;

-- Create face_embeddings table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.face_embeddings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    face_id VARCHAR(255) NOT NULL,
    embedding TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ensure the sequence exists for face_embeddings
CREATE SEQUENCE IF NOT EXISTS public.face_embeddings_id_seq;
ALTER TABLE public.face_embeddings ALTER COLUMN id SET DEFAULT nextval('public.face_embeddings_id_seq');
ALTER SEQUENCE public.face_embeddings_id_seq OWNED BY public.face_embeddings.id;

-- Disable RLS temporarily to allow service access
ALTER TABLE public.models DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_embeddings DISABLE ROW LEVEL SECURITY;

-- Or alternatively, create permissive policies for service access
-- Uncomment these if you prefer to keep RLS enabled:

-- DROP POLICY IF EXISTS "Allow service access to models" ON public.models;
-- CREATE POLICY "Allow service access to models" ON public.models
--     FOR ALL USING (true);

-- DROP POLICY IF EXISTS "Allow service access to face_embeddings" ON public.face_embeddings;
-- CREATE POLICY "Allow service access to face_embeddings" ON public.face_embeddings
--     FOR ALL USING (true);

-- Enable RLS (only if you're using the policies above)
-- ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.face_embeddings ENABLE ROW LEVEL SECURITY;

-- Grant necessary permissions
GRANT ALL ON public.models TO anon, authenticated;
GRANT ALL ON public.face_embeddings TO anon, authenticated;
GRANT USAGE ON SEQUENCE public.models_id_seq TO anon, authenticated;
GRANT USAGE ON SEQUENCE public.face_embeddings_id_seq TO anon, authenticated;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_models_type ON public.models(model_type);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON public.face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_face_id ON public.face_embeddings(face_id);
