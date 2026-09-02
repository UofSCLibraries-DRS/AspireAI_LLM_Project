CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  text_embedding VECTOR(1536),
  title TEXT,
  title_embedding VECTOR(1536),
  description TEXT,
  description_embedding VECTOR(1536),
  combined_embedding VECTOR(1536)
);