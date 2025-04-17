import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_database(db_path="./databases/sample.sqlite"):
    """Create a sample SQLite database for testing the Voice SQL Agent."""
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create a connection to the database
    conn = sqlite3.connect(db_path)
    
    # Create sales table
    sales_data = pd.DataFrame({
        'id': range(1, 1001),
        'date': pd.date_range(start='2023-01-01', periods=1000),
        'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], 1000),
        'category': np.random.choice(['Electronics', 'Accessories', 'Software'], 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'price': np.random.uniform(50, 2000, 1000).round(2),
        'customer_id': np.random.randint(1, 101, 1000)
    })
    
    # Create customers table
    customers_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f"Customer {i}" for i in range(1, 101)],
        'email': [f"customer{i}@example.com" for i in range(1, 101)],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], 100),
        'state': np.random.choice(['CA', 'NY', 'TX', 'IL', 'AZ', 'PA', 'FL', 'OH', 'GA', 'NC'], 100),
        'age': np.random.randint(18, 80, 100),
        'account_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 100)
    })
    
    # Create products table
    products_data = pd.DataFrame({
        'name': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories'],
        'base_price': [1200, 800, 500, 300, 80],
        'stock': np.random.randint(50, 500, 5)
    })
    
    # Create employee data
    employees_data = pd.DataFrame({
        'id': range(1, 21),
        'name': [f"Employee {i}" for i in range(1, 21)],
        'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], 20),
        'salary': np.random.uniform(40000, 120000, 20).round(2),
        'hire_date': pd.date_range(start='2020-01-01', periods=20),
        'manager_id': [None] + list(np.random.randint(1, 5, 19))
    })
    
    # Write tables to SQLite database
    sales_data.to_sql('sales', conn, index=False, if_exists='replace')
    customers_data.to_sql('customers', conn, index=False, if_exists='replace')
    products_data.to_sql('products', conn, index=False, if_exists='replace')
    employees_data.to_sql('employees', conn, index=False, if_exists='replace')
    
    # Create some views
    conn.execute('''
    CREATE VIEW IF NOT EXISTS sales_by_product AS
    SELECT 
        product,
        SUM(quantity) as total_quantity,
        SUM(price * quantity) as total_revenue
    FROM sales
    GROUP BY product
    ''')
    
    conn.execute('''
    CREATE VIEW IF NOT EXISTS customer_spending AS
    SELECT 
        c.id,
        c.name,
        c.city,
        c.state,
        SUM(s.price * s.quantity) as total_spent
    FROM customers c
    JOIN sales s ON c.id = s.customer_id
    GROUP BY c.id
    ''')
    
    # Close the connection
    conn.close()
    
    print(f"Sample database created at {db_path}")
    print("Tables created: sales, customers, products, employees")
    print("Views created: sales_by_product, customer_spending")

if __name__ == "__main__":
    create_sample_database() 