-- Simple fix for Supabase RLS policies for AutoSOS
-- Run this in your Supabase SQL Editor

-- Drop and recreate tables to ensure clean state
DROP TABLE IF EXISTS public.face_embeddings CASCADE;
DROP TABLE IF EXISTS public.models CASCADE;

-- Create models table
CREATE TABLE public.models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50),
    file_path TEXT NOT NULL,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create face_embeddings table
CREATE TABLE public.face_embeddings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    face_id VARCHAR(255) NOT NULL,
    embedding TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Disable RLS for service access
ALTER TABLE public.models DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_embeddings DISABLE ROW LEVEL SECURITY;

-- Grant permissions
GRANT ALL ON public.models TO anon, authenticated;
GRANT ALL ON public.face_embeddings TO anon, authenticated;
GRANT USAGE ON SEQUENCE public.models_id_seq TO anon, authenticated;
GRANT USAGE ON SEQUENCE public.face_embeddings_id_seq TO anon, authenticated;

-- Create indexes
CREATE INDEX idx_models_type ON public.models(model_type);
CREATE INDEX idx_face_embeddings_user_id ON public.face_embeddings(user_id);
CREATE INDEX idx_face_embeddings_face_id ON public.face_embeddings(face_id);

-- Verify tables were created
SELECT 'Tables created successfully' as status;
