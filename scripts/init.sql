-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Sample table: product reviews
-- Structured columns (for SQL path) + a free-text column (for semantic path)
CREATE TABLE IF NOT EXISTS product_reviews (
    id SERIAL PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT NOT NULL,
    review_date DATE NOT NULL,
    region TEXT NOT NULL
);

-- Table to store embeddings for review_text (kept separate so ingestion is decoupled from raw data)
CREATE TABLE IF NOT EXISTS review_embeddings (
    review_id INTEGER PRIMARY KEY REFERENCES product_reviews(id) ON DELETE CASCADE,
    embedding VECTOR(384)  -- 384 dims matches all-MiniLM-L6-v2; change if you pick a different model
);

-- Index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS review_embeddings_idx
    ON review_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- A few seed rows so you have something to query immediately
INSERT INTO product_reviews (product_name, category, price, rating, review_text, review_date, region) VALUES
('Wireless Mouse', 'Electronics', 799.00, 4, 'Works well but the battery drains faster than expected.', '2025-01-15', 'West'),
('Wireless Mouse', 'Electronics', 799.00, 2, 'Stopped connecting after two weeks, very frustrating experience.', '2025-02-03', 'North'),
('Yoga Mat', 'Fitness', 1299.00, 5, 'Great grip and thickness, exactly what I needed for home workouts.', '2025-01-20', 'South'),
('Yoga Mat', 'Fitness', 1299.00, 1, 'Arrived damaged and customer service was unhelpful about the return.', '2025-03-01', 'East'),
('Bluetooth Speaker', 'Electronics', 2499.00, 3, 'Sound quality is decent but battery life is disappointing.', '2025-02-18', 'West'),
('Bluetooth Speaker', 'Electronics', 2499.00, 5, 'Amazing bass and the shipping was surprisingly fast.', '2025-01-05', 'North'),
('Running Shoes', 'Fitness', 3499.00, 4, 'Comfortable for long runs, slightly narrow fit though.', '2025-03-10', 'South'),
('Running Shoes', 'Fitness', 3499.00, 2, 'Sole started peeling off within a month of light use.', '2025-02-25', 'East');
