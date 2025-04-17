"""
Talk2SQL Azure OpenAI Example

This example demonstrates using Talk2SQL with Azure OpenAI, including
the XML tag parsing for SQL queries.
"""

import os
import pandas as pd
from talk2sql.engine import Talk2SQLAzure

# --- Configuration ---
config = {
    # Azure OpenAI settings
    "azure_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    "azure_api_version": os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
    "azure_deployment": os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini"),
    "azure_embedding_deployment": "text-embedding-ada-002",
    "temperature": 0.0,
    
    # Qdrant settings
    "location": ":memory:",  # In-memory for demo purposes
    
    # Retry settings
    "max_retry_attempts": 3,
    "save_query_history": True,
    
    # General settings
    "debug_mode": True,
}

# --- Initialize Talk2SQL Azure ---
Talk2SQL = Talk2SQLAzure(config)

# --- Connect to a SQLite database ---
Talk2SQL.connect_to_sqlite(":memory:")  # In-memory database for demo

# --- Create a simple example database ---
Talk2SQL.run_sql("""
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price NUMERIC,
    stock INTEGER
)
""")

Talk2SQL.run_sql("""
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    sale_date DATE,
    quantity INTEGER,
    total_amount NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
""")

# Insert sample data
Talk2SQL.run_sql("""
INSERT INTO products (product_id, name, category, price, stock)
VALUES 
    (1, 'Laptop', 'Electronics', 1200.00, 45),
    (2, 'Smartphone', 'Electronics', 800.00, 120),
    (3, 'Headphones', 'Electronics', 150.00, 75),
    (4, 'Desk Chair', 'Furniture', 250.00, 30),
    (5, 'Desk', 'Furniture', 350.00, 20),
    (6, 'Bookshelf', 'Furniture', 150.00, 15),
    (7, 'Coffee Maker', 'Appliances', 80.00, 25),
    (8, 'Blender', 'Appliances', 60.00, 30)
""")

Talk2SQL.run_sql("""
INSERT INTO sales (sale_id, product_id, sale_date, quantity, total_amount)
VALUES 
    (101, 1, '2023-01-10', 2, 2400.00),
    (102, 2, '2023-01-15', 5, 4000.00),
    (103, 3, '2023-01-20', 3, 450.00),
    (104, 4, '2023-01-25', 1, 250.00),
    (105, 5, '2023-02-05', 2, 700.00),
    (106, 2, '2023-02-10', 8, 6400.00),
    (107, 7, '2023-02-15', 4, 320.00),
    (108, 8, '2023-02-20', 3, 180.00),
    (109, 1, '2023-02-25', 1, 1200.00),
    (110, 3, '2023-03-01', 6, 900.00)
""")

# --- Add schema information to Talk2SQL ---
schema = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price NUMERIC,
    stock INTEGER
);

CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    sale_date DATE,
    quantity INTEGER,
    total_amount NUMERIC,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

Talk2SQL.add_schema(schema)

# --- Add documentation ---
documentation = """
The product sales database tracks inventory and sales information.

The 'products' table contains product information including name, category, price, and current stock level.

The 'sales' table records individual sales transactions with the quantity sold and total amount for each sale.
Products and sales are related through the product_id field.
"""

Talk2SQL.add_documentation(documentation)

# --- Add example question-SQL pairs for training (with XML tags) ---
example_pairs = [
    {
        "question": "What is the total revenue from all sales?",
        "sql": "<sql>SELECT SUM(total_amount) FROM sales;</sql>"
    },
    {
        "question": "Which product category has the highest total sales?",
        "sql": """<sql>
SELECT p.category, SUM(s.total_amount) as category_sales
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY category_sales DESC
LIMIT 1;
</sql>"""
    },
    {
        "question": "What's the average price of products in each category?",
        "sql": "<sql>SELECT category, AVG(price) as avg_price FROM products GROUP BY category;</sql>"
    }
]

from talk2sql.utils import extract_content_from_xml_tags

for example in example_pairs:
    # Extract SQL from XML tags
    sql = example["sql"]
    if "<sql>" in sql:
        sql = extract_content_from_xml_tags(sql, "sql")
    
    Talk2SQL.add_question_sql(example["question"], sql)

# --- Run a query ---
print("\n----- Example 1: Basic Query -----")
question = "What are the top 3 best-selling products by quantity?"
result = Talk2SQL.smart_query(question)

# --- Run a query that requires joining tables ---
print("\n----- Example 2: Join Query -----")
question = "What is the total revenue for each product category?"
result = Talk2SQL.smart_query(question)

# --- Example 3: Analyze the data comprehensively ---
print("\n----- Example 3: Comprehensive Analysis -----")
question = "Which product has the highest revenue per unit sold?"
analysis = Talk2SQL.analyze_data(question, visualize=True, explain=True, suggest_followups=True)

if analysis["success"]:
    print("\nExplanation:")
    print(analysis["explanation"])
    
    print("\nFollow-up questions:")
    for i, q in enumerate(analysis["followup_questions"], 1):
        print(f"{i}. {q}")
else:
    print(f"Analysis failed: {analysis['error']}")

# --- Example 4: Run a demo with a more complex question ---
print("\n----- Example 4: Interactive Demo -----")
Talk2SQL.demo("What are the monthly sales trends for Electronics products in 2023?")