CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  collection TEXT NOT NULL,
  title TEXT,
  description TEXT,
  transcript TEXT,
  title_embedding VECTOR(768),
  description_embedding VECTOR(768),
  transcript_embedding VECTOR(768)
);
