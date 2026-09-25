CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  collection TEXT NOT NULL,
  title TEXT,
  description TEXT,
  transcript TEXT,
  title_embedding VECTOR(768),
  description_embedding VECTOR(768),
  transcript_embedding VECTOR(768)
);

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS search_text TEXT
  GENERATED ALWAYS AS (
    COALESCE(title, '') || ' ' ||
    COALESCE(description, '') || ' ' ||
    COALESCE(transcript, '')
  ) STORED;

CREATE INDEX IF NOT EXISTS documents_title_embedding_hnsw
  ON documents USING hnsw (title_embedding vector_cosine_ops)
  WHERE title_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_description_embedding_hnsw
  ON documents USING hnsw (description_embedding vector_cosine_ops)
  WHERE description_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_transcript_embedding_hnsw
  ON documents USING hnsw (transcript_embedding vector_cosine_ops)
  WHERE transcript_embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_search_text_bm25
  ON documents USING bm25 (search_text)
  WITH (text_config = 'english');
