"""
Talk2SQL Quick Start Example

This example demonstrates the basic usage of Talk2SQL with Azure OpenAI, 
including XML tag parsing for SQL queries.
"""

import os
import pandas as pd
from talk2sql.engine import Talk2SQLAzure
from talk2sql.utils import format_sql_with_xml_tags

# --- Configuration ---
config = {
    # Azure OpenAI settings
    "azure_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    "azure_api_version": os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
    "azure_deployment": os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini"),
    "azure_embedding_deployment": "text-embedding-ada-002",
    "temperature": 0.3,
    
    # Vector store settings
    "location": ":memory:",  # In-memory for demo purposes
    
    # General settings
    "debug_mode": True,
}

# --- Initialize Talk2SQLAzure ---
Talk2SQL = Talk2SQLAzure(config)

# --- Connect to a SQLite database ---
Talk2SQL.connect_to_sqlite(":memory:")  # In-memory database for demo

# --- Create a sample database ---
Talk2SQL.run_sql("""
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    position TEXT,
    hire_date DATE,
    salary NUMERIC
)
""")

Talk2SQL.run_sql("""
INSERT INTO employees (employee_id, name, department, position, hire_date, salary)
VALUES 
    (1, 'John Doe', 'Engineering', 'Senior Engineer', '2020-01-15', 120000),
    (2, 'Jane Smith', 'Marketing', 'Marketing Manager', '2019-03-20', 110000),
    (3, 'Bob Johnson', 'Engineering', 'Engineer', '2021-05-10', 95000),
    (4, 'Alice Williams', 'HR', 'HR Manager', '2018-11-01', 105000),
    (5, 'Charlie Brown', 'Engineering', 'Lead Engineer', '2017-08-15', 135000),
    (6, 'Diana Prince', 'Sales', 'Sales Director', '2016-04-20', 150000),
    (7, 'Edward Jones', 'Marketing', 'Marketing Specialist', '2022-01-10', 85000),
    (8, 'Fiona Green', 'Engineering', 'QA Engineer', '2021-09-15', 90000)
""")

# --- Add schema to Talk2SQL ---
schema = """
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    position TEXT,
    hire_date DATE,
    salary NUMERIC
);
"""

Talk2SQL.add_schema(schema)

# --- Add documentation ---
documentation = """
The employees table contains information about company employees including their 
name, department, position, hire date, and salary.

Departments include: Engineering, Marketing, HR, and Sales.

Salaries are annual amounts in USD.
"""

Talk2SQL.add_documentation(documentation)

# --- Add example question-SQL pairs with XML tags ---
example_pairs = [
    {
        "question": "What is the average salary by department?",
        "sql": "<sql>SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department;</sql>"
    },
    {
        "question": "Who are the employees in Engineering?",
        "sql": "<sql>SELECT name, position FROM employees WHERE department = 'Engineering';</sql>"
    }
]

# Add examples to the vector store
for example in example_pairs:
    # Extract the SQL from the XML tags using the utility function
    from talk2sql.utils import extract_content_from_xml_tags
    sql = extract_content_from_xml_tags(example["sql"], "sql")
    Talk2SQL.add_question_sql(example["question"], sql)

# --- Basic Query Example ---
print("\n----- Basic Query Example -----")
question = "What is the total salary for each department?"
result = Talk2SQL.smart_query(question)

# --- More Complex Query Example ---
print("\n----- Complex Query Example -----")
question = "Who is the highest paid employee in each department?"
result = Talk2SQL.smart_query(question)

# --- Direct SQL Generation Example ---
print("\n----- Direct SQL Generation Example -----")
question = "List employees hired after 2020 sorted by salary"
sql = Talk2SQL.generate_sql(question)
print(f"Generated SQL for '{question}':")
print(sql)

# --- Query with Explanation ---
print("\n----- Query with Explanation Example -----")
question = "What is the average tenure of employees by department?"
sql, df, _ = Talk2SQL.ask(question, auto_train=True)
if df is not None:
    explanation = Talk2SQL.explain_results(question, df)
    print("\nExplanation:")
    print(explanation)

print("\n✅ Quick Start Example Complete")
