"""
Example demonstrating the streaming and multithreading capabilities of Talk2SQL.
"""

import os
import sys
import time
from pprint import pprint

# Add parent directory to path to import Talk2SQL
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talk2sql import Talk2SQLAzure, EventType

# Configure Talk2SQL with streaming enabled
config = {
    "azure_openai_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_openai_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "azure_openai_api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    "azure_openai_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "enable_streaming": True,
    "enable_threading": True,
}

# Initialize Talk2SQL
Talk2SQL = Talk2SQLAzure(config)

# Example database connection function - replace with your actual connection
def connect_to_db():
    import pandas as pd
    
    # Mock function that returns sample data
    def run_sql(sql):
        # This is just a mock implementation for demonstration
        print(f"Running SQL: {sql}")
        
        # Return different sample data based on the SQL
        if "customers" in sql.lower():
            return pd.DataFrame({
                "customer_id": range(1, 11),
                "name": [f"Customer {i}" for i in range(1, 11)],
                "revenue": [i * 1000 for i in range(1, 11)]
            })
        elif "products" in sql.lower():
            return pd.DataFrame({
                "product_id": range(1, 6),
                "name": [f"Product {i}" for i in range(1, 6)],
                "price": [i * 10 for i in range(1, 6)]
            })
        else:
            return pd.DataFrame({
                "id": range(1, 5),
                "value": [10, 20, 30, 40]
            })
    
    Talk2SQL.run_sql = run_sql

# Connect to the database
connect_to_db()

# Add some example schema information
Talk2SQL.add_schema("""
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(255),
    revenue DECIMAL(10, 2),
    created_at TIMESTAMP
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10, 2),
    inventory INT
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    quantity INT,
    order_date TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
""")

# Event handler for streaming events
def handle_streaming_event(event):
    event_data = event.to_dict()
    event_type = event_data["event"]
    
    # Handle different event types
    if event_type == EventType.SQL_GENERATION:
        if event_data["metadata"]["status"] == "completed":
            print("\n🔍 GENERATED SQL QUERY:")
            print(event_data["data"])
    
    elif event_type == EventType.SQL_EXECUTION:
        if event_data["metadata"]["status"] == "started":
            print("\n⏳ EXECUTING SQL...")
        elif event_data["metadata"]["status"] == "completed":
            print("✅ SQL EXECUTED SUCCESSFULLY")
    
    elif event_type == EventType.DATAFRAME_READY:
        print("\n📊 DATA READY:")
        if isinstance(event_data["data"], dict) and event_data["data"]["type"] == "dataframe":
            # Print the first few rows of the dataframe
            rows = event_data["data"]["data"][:3]
            for row in rows:
                print(row)
            if len(event_data["data"]["data"]) > 3:
                print(f"... and {len(event_data['data']['data']) - 3} more rows")
    
    elif event_type == EventType.VISUALIZATION_READY:
        if event_data["metadata"]["status"] == "completed":
            print("\n📈 VISUALIZATION GENERATED (would be displayed in a notebook/UI)")
    
    elif event_type == EventType.LLM_SUMMARY:
        if event_data["metadata"]["status"] == "completed":
            print("\n🔎 DATA ANALYSIS:")
            print(event_data["data"])
    
    elif event_type == EventType.ERROR:
        print(f"\n❌ ERROR: {event_data['data']}")

# Register the event handler
Talk2SQL._streaming_pipeline.executor.register_event_handler("*", handle_streaming_event)

def demo_streaming():
    """Run a demo of streaming capabilities."""
    print("\n==== STREAMING DEMO ====\n")
    print("Processing question with streaming events...\n")
    
    # Process a question with streaming
    task_id = Talk2SQL.ask(
        "What are the top 3 customers by revenue?",
        streaming=True
    )
    
    print(f"Task started with ID: {task_id}")
    
    # Wait for events to be processed (in a real app, you would use an event loop)
    # For this demo, we'll just sleep for a few seconds
    time.sleep(10)
    
    print("\n==== STREAMING DEMO COMPLETED ====\n")

def demo_threading():
    """Run a demo of threading capabilities by running multiple queries concurrently."""
    print("\n==== THREADING DEMO ====\n")
    print("Processing multiple questions concurrently...\n")
    
    # Questions to process
    questions = [
        "Show me all customers with revenue over 5000",
        "What products do we have in stock?",
        "How many orders do we have per customer?"
    ]
    
    # Process multiple questions concurrently
    task_ids = []
    for q in questions:
        task_id = Talk2SQL.ask(q, streaming=True)
        task_ids.append(task_id)
        print(f"Started task {task_id} for question: {q}")
    
    # Wait for all tasks to complete (in a real app, you would use an event loop)
    time.sleep(15)
    
    print("\n==== THREADING DEMO COMPLETED ====\n")

if __name__ == "__main__":
    # Run the streaming demo
    demo_streaming()
    
    # Run the threading demo
    demo_threading() 