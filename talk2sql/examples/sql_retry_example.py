"""
Talk2SQL Retry Mechanism Example

This example demonstrates the SQL error handling and retry capabilities of Talk2SQL,
as well as the new XML tag parsing for SQL queries using Azure OpenAI.
"""

import os
import pandas as pd
from talk2sql.engine import Talk2SQLAzure
from talk2sql.utils import format_sql_with_xml_tags, extract_content_from_xml_tags

# --- Configuration ---
config = {
    # Azure OpenAI settings
    "azure_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    "azure_api_version": os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
    "azure_deployment": os.environ.get("AZURE_DEPLOYMENT", "gpt-4o"),
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

# --- Initialize Talk2SQLAzure ---
Talk2SQL = Talk2SQLAzure(config)

# --- Connect to a SQLite database ---
Talk2SQL.connect_to_sqlite(":memory:")  # In-memory database for demo

# --- Create a sample database with a deliberate schema issue ---
# 1. Create tables with a naming inconsistency that will cause errors
Talk2SQL.run_sql("""
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date DATE,
    country TEXT
)
""")

Talk2SQL.run_sql("""
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    client_id INTEGER,  -- Note: This is intentionally named client_id instead of customer_id
    order_date DATE,
    total_amount NUMERIC,
    status TEXT
)
""")

Talk2SQL.run_sql("""
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_name TEXT,
    quantity INTEGER,
    price NUMERIC,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
)
""")

# 2. Insert sample data
Talk2SQL.run_sql("""
INSERT INTO customers (customer_id, name, email, signup_date, country)
VALUES 
    (1, 'John Smith', 'john@example.com', '2022-01-10', 'USA'),
    (2, 'Maria Garcia', 'maria@example.com', '2022-02-15', 'Mexico'),
    (3, 'Hiroshi Tanaka', 'hiroshi@example.com', '2022-03-20', 'Japan'),
    (4, 'Sarah Johnson', 'sarah@example.com', '2022-04-25', 'USA'),
    (5, 'Ahmed Hassan', 'ahmed@example.com', '2022-05-30', 'Egypt')
""")

Talk2SQL.run_sql("""
INSERT INTO orders (order_id, client_id, order_date, total_amount, status)
VALUES 
    (101, 1, '2023-01-15', 150.50, 'Completed'),
    (102, 2, '2023-02-20', 89.99, 'Completed'),
    (103, 3, '2023-03-25', 249.99, 'Completed'),
    (104, 4, '2023-04-30', 75.25, 'Processing'),
    (105, 5, '2023-05-05', 189.75, 'Completed')
""")

Talk2SQL.run_sql("""
INSERT INTO order_items (item_id, order_id, product_name, quantity, price)
VALUES 
    (1001, 101, 'Laptop', 1, 120.50),
    (1002, 101, 'Mouse', 1, 30.00),
    (1003, 102, 'Headphones', 1, 89.99),
    (1004, 103, 'Smartphone', 1, 249.99),
    (1005, 104, 'Keyboard', 1, 75.25)
""")

# --- Add schema information to Talk2SQL ---
schema = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date DATE,
    country TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    client_id INTEGER,  -- Inconsistent naming with customer_id
    order_date DATE,
    total_amount NUMERIC,
    status TEXT
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_name TEXT,
    quantity INTEGER,
    price NUMERIC,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
"""

Talk2SQL.add_schema(schema)

# --- Add documentation for context ---
documentation = """
The e-commerce database contains information about customers, their orders, and items within those orders.

The 'customers' table holds customer information including their name, email, signup date, and country.

The 'orders' table contains order details like order date, total amount, and status (which can be 'Completed', 'Processing', or 'Cancelled'). Note that the foreign key to customers is named 'client_id' in the orders table.

The 'order_items' table lists the individual items within each order, including the product name, quantity, and price.
"""

Talk2SQL.add_documentation(documentation)

# --- Add example question-SQL pairs for training ---
# Note: We now use XML tags for SQL
example_pairs = [
    {
        "question": "How many customers do we have?",
        "sql": "<sql>SELECT COUNT(*) FROM customers;</sql>"
    },
    {
        "question": "What is the total revenue from all completed orders?",
        "sql": "<sql>SELECT SUM(total_amount) FROM orders WHERE status = 'Completed';</sql>"
    },
    {
        "question": "Which country has the most customers?",
        "sql": "<sql>SELECT country, COUNT(*) as customer_count FROM customers GROUP BY country ORDER BY customer_count DESC LIMIT 1;</sql>"
    }
]

for example in example_pairs:
    # Extract SQL from XML tags if needed
    sql = example["sql"]
    if "<sql>" in sql:
        sql = extract_content_from_xml_tags(sql, "sql")
    
    Talk2SQL.add_question_sql(example["question"], sql)

# --- Example 1: Query that will need to be corrected ---
print("\n----- Example 1: Query With Error That Needs Correction -----")
print("This query will initially fail because it tries to join customers and orders using inconsistent field names")

question = "List all customers with their order totals"
result = Talk2SQL.smart_query(question)

# --- Example 2: Another query type that will need correction ---
print("\n----- Example 2: Another Query Type With Error -----")
print("This query will initially fail because it references a non-existent column")

question = "What's the average order value per country?"
result = Talk2SQL.smart_query(question)

# --- Example 3: Query analysis after several attempts ---
print("\n----- Example 3: Query History Analysis -----")
error_analysis = Talk2SQL.analyze_error_patterns()
print("\nError Analysis:")
print(f"Total queries: {error_analysis['total_queries']}")
print(f"Error rate: {error_analysis['error_rate']:.2%}")
print(f"Retry success rate: {error_analysis['retry_success_rate']:.2%}")
print("\nCommon error types:")
for error_type, count in error_analysis['common_error_types']:
    print(f"- {error_type}: {count}")

# --- Example 4: Generate training recommendations based on errors ---
print("\n----- Example 4: Training Recommendations From Errors -----")
recommendations = Talk2SQL.train_from_errors()
print("\nRecommendations for improving training data:")
print(recommendations['recommendations'])

# --- Example 5: Demonstrating XML tag parsing ---
print("\n----- Example 5: XML Tag Parsing Example -----")
# Create a fake LLM response with XML tags
fake_llm_response = """
To answer this question, I'll need to find the top-selling product by quantity.

<sql>
SELECT product_name, SUM(quantity) as total_quantity
FROM order_items
GROUP BY product_name
ORDER BY total_quantity DESC
LIMIT 1;
</sql>

This query will return the product name and its total quantity sold.
"""

# Extract the SQL using our utility function
extracted_sql = extract_content_from_xml_tags(fake_llm_response, "sql")
print("\nExample LLM Response with XML Tags:")
print(fake_llm_response)
print("\nExtracted SQL:")
print(extracted_sql)

# Format SQL with XML tags
print("\nFormatting SQL with XML tags:")
print(format_sql_with_xml_tags("""SELECT * FROM customers WHERE country = 'USA';"""))