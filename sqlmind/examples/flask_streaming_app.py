"""
Example Flask application that demonstrates SQLMind's streaming capabilities.
"""

import os
import sys
import json

# Add parent directory to path to import sqlmind
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmind import SQLMindAzure, create_flask_streaming_endpoint
from flask import Flask, request, jsonify, Response, render_template_string

# Configure SQLMind with streaming enabled
config = {
    "azure_openai_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_openai_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "azure_openai_api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    "azure_openai_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "enable_streaming": True,
    "enable_threading": True,
}

# Initialize SQLMind
sqlmind = SQLMindAzure(config)

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
    
    sqlmind.run_sql = run_sql

# Connect to the database
connect_to_db()

# Add some example schema information
sqlmind.add_schema("""
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

# Create the Flask app
app = Flask(__name__)

# Set up the streaming endpoint
create_flask_streaming_endpoint(app, sqlmind._streaming_pipeline)

# Define a simple HTML page with JavaScript for consuming the SSE stream
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SQLMind Streaming Demo</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .input-panel {
            flex: 1;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-panel {
            flex: 2;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .section {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .hidden {
            display: none;
        }
        pre {
            background-color: #f1f1f1;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        #status {
            color: #3498db;
            font-style: italic;
        }
        .event-indicator {
            margin-right: 10px;
            font-weight: bold;
        }
        #visualization-container {
            width: 100%;
            height: 400px;
            max-width: 100%;
            margin-top: 20px;
        }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>SQLMind Streaming Demo</h1>
    
    <div class="container">
        <div class="input-panel">
            <h2>Ask a Question</h2>
            <textarea id="question" rows="4" placeholder="Enter your SQL question here...">What are the top 5 customers by revenue?</textarea>
            <label>
                <input type="checkbox" id="allow-introspection"> Allow Introspection
            </label>
            <button id="submit">Submit Question</button>
            <div id="status" class="hidden">Processing...</div>
        </div>
        
        <div class="results-panel">
            <div id="sql-section" class="section hidden">
                <h2>SQL Query</h2>
                <pre id="sql-code"></pre>
            </div>
            
            <div id="data-section" class="section hidden">
                <h2>Results</h2>
                <div id="data-container"></div>
            </div>
            
            <div id="visualization-section" class="section hidden">
                <h2>Visualization</h2>
                <div id="visualization-container"></div>
            </div>
            
            <div id="summary-section" class="section hidden">
                <h2>Summary</h2>
                <div id="summary-content"></div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const questionInput = document.getElementById('question');
            const allowIntrospection = document.getElementById('allow-introspection');
            const submitButton = document.getElementById('submit');
            const statusElement = document.getElementById('status');
            const sqlSection = document.getElementById('sql-section');
            const sqlCode = document.getElementById('sql-code');
            const dataSection = document.getElementById('data-section');
            const dataContainer = document.getElementById('data-container');
            const visualizationSection = document.getElementById('visualization-section');
            const visualizationContainer = document.getElementById('visualization-container');
            const summarySection = document.getElementById('summary-section');
            const summaryContent = document.getElementById('summary-content');
            
            let eventSource = null;
            
            // Function to create a table from data
            function createTable(data, columns) {
                let table = document.createElement('table');
                
                // Create header row
                let thead = document.createElement('thead');
                let headerRow = document.createElement('tr');
                
                columns.forEach(column => {
                    let th = document.createElement('th');
                    th.textContent = column;
                    headerRow.appendChild(th);
                });
                
                thead.appendChild(headerRow);
                table.appendChild(thead);
                
                // Create body
                let tbody = document.createElement('tbody');
                
                data.forEach(row => {
                    let tr = document.createElement('tr');
                    
                    columns.forEach(column => {
                        let td = document.createElement('td');
                        td.textContent = row[column];
                        tr.appendChild(td);
                    });
                    
                    tbody.appendChild(tr);
                });
                
                table.appendChild(tbody);
                return table;
            }
            
            // Handle form submission
            submitButton.addEventListener('click', function() {
                // Reset UI
                statusElement.classList.remove('hidden');
                sqlSection.classList.add('hidden');
                dataSection.classList.add('hidden');
                visualizationSection.classList.add('hidden');
                summarySection.classList.add('hidden');
                
                // Close existing event source if any
                if (eventSource) {
                    eventSource.close();
                }
                
                // Submit the question
                fetch('/api/stream/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: questionInput.value,
                        allow_introspection: allowIntrospection.checked
                    })
                })
                .then(response => response.json())
                .then(data => {
                    const taskId = data.task_id;
                    console.log('Task ID:', taskId);
                    
                    // Set up SSE connection
                    eventSource = new EventSource(`/api/stream/events/${taskId}`);
                    
                    // Listen for specific event types
                    eventSource.addEventListener('sql_generation', function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.metadata.status === 'completed') {
                            sqlCode.textContent = data.data;
                            sqlSection.classList.remove('hidden');
                            statusElement.textContent = 'SQL generated, executing...';
                        }
                    });
                    
                    eventSource.addEventListener('sql_execution', function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.metadata.status === 'completed') {
                            statusElement.textContent = 'SQL executed, processing results...';
                        }
                    });
                    
                    eventSource.addEventListener('dataframe_ready', function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.data.type === 'dataframe') {
                            // Create and display table
                            const tableData = data.data.data;
                            const columns = data.data.columns;
                            
                            dataContainer.innerHTML = '';
                            dataContainer.appendChild(createTable(tableData, columns));
                            dataSection.classList.remove('hidden');
                            
                            statusElement.textContent = 'Data loaded, generating visualization...';
                        }
                    });
                    
                    eventSource.addEventListener('visualization_ready', function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.metadata.status === 'completed') {
                            try {
                                // Parse the Plotly figure JSON
                                const figure = JSON.parse(data.data.data);
                                
                                // Render the figure
                                Plotly.newPlot('visualization-container', figure.data, figure.layout);
                                visualizationSection.classList.remove('hidden');
                                
                                statusElement.textContent = 'Visualization generated, generating summary...';
                            } catch (error) {
                                console.error('Error rendering visualization:', error);
                            }
                        }
                    });
                    
                    eventSource.addEventListener('llm_summary', function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.metadata.status === 'completed') {
                            summaryContent.innerHTML = data.data.replace(/\\n/g, '<br>');
                            summarySection.classList.remove('hidden');
                            
                            statusElement.textContent = 'Complete!';
                            eventSource.close();
                        }
                    });
                    
                    eventSource.addEventListener('error', function(event) {
                        const data = JSON.parse(event.data);
                        statusElement.textContent = `Error: ${data.data}`;
                        eventSource.close();
                    });
                })
                .catch(error => {
                    console.error('Error submitting question:', error);
                    statusElement.textContent = `Error: ${error.message}`;
                });
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    # This is just for the example - in production, use a proper WSGI server
    app.run(debug=True, threaded=True, port=5000) 