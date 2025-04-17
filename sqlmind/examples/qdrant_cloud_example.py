#!/usr/bin/env python
"""
Example script demonstrating how to use SQLMind with Qdrant Cloud vector storage.
"""

import os
from sqlmind.engine import SQLMindAzure
from sqlmind.vector_store import QdrantVectorStore

def main():
    # Qdrant Cloud configuration
    qdrant_config = {
        "qdrant_url": "https://d960d7c1-5c26-4a91-8e7a-fb70954d24c1.eu-west-1-0.aws.cloud.qdrant.io:6333", 
        "qdrant_api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.NQSukS5fheodhJDs5AxgxpOxHJCG9ROszaR2Jr6o1BU",
        "embedding_size": 1536,  # Size for OpenAI embeddings
        "questions_collection": "sqlmind_questions",
        "schema_collection": "sqlmind_schema",
        "docs_collection": "sqlmind_docs",
        "n_results": 5  # Number of similar results to return
    }
    
    # Initialize the vector store
    vector_store = QdrantVectorStore(qdrant_config)
    
    # Verify collections exist or create them
    print("Setting up Qdrant collections...")
    vector_store._setup_collections()
    print("Collections setup complete.")
    
    # Example schema information
    schema = """
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        signup_date DATE,
        last_login TIMESTAMP
    );

    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        price NUMERIC(10, 2) NOT NULL,
        category TEXT
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(customer_id),
        order_date TIMESTAMP NOT NULL,
        total_amount NUMERIC(10, 2) NOT NULL,
        status TEXT NOT NULL
    );

    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER REFERENCES orders(order_id),
        product_id INTEGER REFERENCES products(product_id),
        quantity INTEGER NOT NULL,
        price NUMERIC(10, 2) NOT NULL
    );
    """
    
    # Add the schema to the vector store
    print("Adding schema to vector store...")
    schema_id = vector_store.add_schema(schema)
    print(f"Schema added with ID: {schema_id}")
    
    # Example training data (question-SQL pairs)
    training_data = [
        {
            "question": "How many customers do we have?",
            "sql": "SELECT COUNT(*) FROM customers;"
        },
        {
            "question": "What is the total revenue from all orders?",
            "sql": "SELECT SUM(total_amount) FROM orders;"
        },
        {
            "question": "List the top 5 most expensive products",
            "sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"
        },
        {
            "question": "Find customers who haven't placed any orders",
            "sql": "SELECT c.customer_id, c.name FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;"
        }
    ]
    
    # Add the training data to the vector store
    print("Adding training data to vector store...")
    for item in training_data:
        question_id = vector_store.add_question_sql(item["question"], item["sql"])
        print(f"Added question with ID: {question_id}")
    
    # Example: Retrieve similar questions
    test_question = "What are our top 5 highest priced products?"
    print(f"\nFinding similar questions to: '{test_question}'")
    similar_questions = vector_store.get_similar_questions(test_question)
    
    print("\nSimilar questions found:")
    for i, item in enumerate(similar_questions, 1):
        print(f"{i}. Question: {item['question']}")
        print(f"   SQL: {item['sql']}\n")
    
    # Example: Display all training data
    print("All training data:")
    training_df = vector_store.get_all_training_data()
    print(training_df)
    
    print("\nQdrant Cloud integration example complete.")

if __name__ == "__main__":
    main() 