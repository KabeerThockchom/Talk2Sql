from typing import Dict, List, Union, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import datetime
import sqlite3
from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams
import os

from talk2sql.vector_store.qdrant import QdrantVectorStore
from talk2sql.llm.anthropic import AnthropicLLM
from talk2sql.llm.azure_openai import AzureOpenAILLM

class Talk2SQLAnthropic(QdrantVectorStore, AnthropicLLM):
    """
    Main Talk2SQL engine that combines Qdrant vector storage, Anthropic LLM capabilities,
    and Azure OpenAI embeddings.
    """
    
    def __init__(self, config=None):
        """
        Initialize Talk2SQL with Anthropic and Qdrant, using Azure OpenAI for embeddings.
        
        Args:
            config: Configuration dictionary with options:
              - max_retry_attempts: Maximum number of retries for failed SQL queries (default: 3)
              - save_query_history: Whether to save query history with errors and retries (default: True)
              - anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
              - claude_model: Claude model name (default: "claude-3-7-sonnet-20250219")
              - azure_api_key: Azure OpenAI API key for embeddings (or use AZURE_OPENAI_API_KEY env var)
              - azure_endpoint: Azure OpenAI endpoint for embeddings (or use AZURE_ENDPOINT env var)
              - azure_api_version: Azure API version (or use AZURE_API_VERSION env var)
              - azure_embedding_deployment: Embedding model name (default: "text-embedding-ada-002")
              - history_db_path: Path to SQLite database for storing query history (default: "query_history.db")
        """
        config = config or {}
        
        # Initialize QdrantVectorStore first
        QdrantVectorStore.__init__(self, config)
        
        # Save reference to Qdrant client before it gets overwritten
        self.qdrant_client = self.client
        
        # Now initialize AnthropicLLM (which might overwrite self.client)
        AnthropicLLM.__init__(self, config)
        
        # Set up Azure OpenAI client for embeddings
        self._setup_azure_embedding_client(config)
        
        # Now that both clients are initialized, set up the collections
        self._setup_collections()
        
        # Additional configuration
        self.debug_mode = config.get("debug_mode", False)
        self.auto_visualization = config.get("auto_visualization", True)
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        self.save_query_history = config.get("save_query_history", True)
        self.history_db_path = config.get("history_db_path", "query_history.db")
        
        # Initialize query history DB
        self._init_history_db()
        
        # SQLite connection
        self.conn = None
    
    def _init_history_db(self):
        """
        Initialize the SQLite database for query history.
        """
        if not self.save_query_history:
            return
            
        try:
            import sqlite3
            
            # Connect to the history database
            history_conn = sqlite3.connect(self.history_db_path)
            cursor = history_conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    question TEXT,
                    sql TEXT,
                    success INTEGER,
                    error_message TEXT,
                    retry_count INTEGER,
                    data BLOB,
                    columns TEXT,
                    visualization BLOB,
                    summary TEXT,
                    total_time_ms REAL,
                    sql_generation_time_ms REAL,
                    sql_execution_time_ms REAL,
                    visualization_time_ms REAL,
                    explanation_time_ms REAL,
                    timing_details TEXT
                )
            ''')
            
            history_conn.commit()
            history_conn.close()
            
            if self.debug_mode:
                print(f"Initialized query history database at {self.history_db_path}")
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error initializing query history database: {e}")
                
            # Fallback to in-memory storage
            self.query_history = []
            self.save_query_history = False
    
    def _setup_azure_embedding_client(self, config):
        """Set up Azure OpenAI client for embeddings."""
        # Get Azure OpenAI API credentials
        self.azure_api_key = config.get("azure_api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        self.azure_endpoint = config.get("azure_endpoint", os.getenv("AZURE_ENDPOINT"))
        self.azure_api_version = config.get("azure_api_version", os.getenv("AZURE_API_VERSION", "2024-02-15-preview"))
        self.azure_embedding_deployment = config.get("azure_embedding_deployment", "text-embedding-ada-002")
        
        # Validate required parameters
        if not self.azure_api_key:
            raise ValueError("Azure OpenAI API key is required for embeddings. Provide in config or set AZURE_OPENAI_API_KEY env var.")
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required for embeddings. Provide in config or set AZURE_ENDPOINT env var.")
        
        # Initialize Azure OpenAI client for embeddings
        from openai import AzureOpenAI
        self.azure_client = AzureOpenAI(
            api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.azure_api_version
        )
    
    def connect_to_sqlite(self, db_path: str):
        """
        Connect to a SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        try:
            # Verify database file exists
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")
            
            # Get file size to verify it's not empty
            file_size = os.path.getsize(db_path)
            if file_size == 0:
                raise ValueError(f"Database file is empty (0 bytes): {db_path}")
            
            if self.debug_mode:
                print(f"Found database file: {db_path} ({file_size} bytes)")
            
            # Connect to database
            self.conn = sqlite3.connect(db_path)
            
            # Test connection with a simple query
            test_cursor = self.conn.cursor()
            test_cursor.execute("PRAGMA database_list")
            db_info = test_cursor.fetchall()
            if self.debug_mode:
                print(f"Database info: {db_info}")
            
            # Define the run_sql function to use this connection
            def run_sql(sql_query):
                return pd.read_sql_query(sql_query, self.conn)
            
            # Set the run_sql function
            self.run_sql = run_sql
            self.run_sql_is_set = True
            
            # Check for tables
            tables_cursor = self.conn.cursor()
            tables_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = tables_cursor.fetchall()
            
            if self.debug_mode:
                if tables:
                    table_names = [t[0] for t in tables]
                    print(f"Connected to SQLite database: {db_path}")
                    print(f"Available tables: {', '.join(table_names)}")
                else:
                    print(f"Connected to SQLite database: {db_path}, but no tables found")
                
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"Error connecting to SQLite database: {e}")
                import traceback
                traceback.print_exc()
            raise e
    
    # Override generate_embedding to use Azure OpenAI
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Azure OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Create an embedding with Azure OpenAI
            response = self.azure_client.embeddings.create(
                model=self.azure_embedding_deployment,
                input=text
            )
            
            # Return the embedding
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating Azure OpenAI embedding: {e}")
            # Fallback to simple embedding
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Simple fallback embedding function.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple hash-based embedding (not for production use)
        """
        import hashlib
        import numpy as np
        
        # Hash the text to get deterministic vector
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.sha256(text_bytes)
        hash_bytes = hash_obj.digest()
        
        # Convert hash to list of floats
        np.random.seed(int.from_bytes(hash_bytes[:4], byteorder='little'))
        return np.random.normal(0, 1, 1536).tolist()
    
    # Override vector store methods to ensure they work correctly
    def add_question_sql(self, question: str, sql: str) -> str:
        """
        Add question-SQL pair to vector store.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            
        Returns:
            ID of the stored entry
        """
        # Create a composite representation
        content = f"Question: {question}\nSQL: {sql}"
        
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(content)
        
        # Get embedding
        embedding = self.generate_embedding(question)
        
        # Insert into questions collection
        self.qdrant_client.upsert(
            collection_name=self.questions_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question": question,
                        "sql": sql
                    }
                )
            ]
        )
        
        return f"{point_id}-q"
    
    def add_schema(self, schema: str) -> str:
        """
        Add database schema to vector store.
        
        Args:
            schema: Database schema (DDL)
            
        Returns:
            ID of the stored entry
        """
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(schema)
        
        # Get embedding
        embedding = self.generate_embedding(schema)
        
        # Insert into schema collection
        self.qdrant_client.upsert(
            collection_name=self.schema_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "schema": schema
                    }
                )
            ]
        )
        
        return f"{point_id}-s"
    
    def add_documentation(self, documentation: str) -> str:
        """
        Add documentation to vector store.
        
        Args:
            documentation: Documentation text
            
        Returns:
            ID of the stored entry
        """
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(documentation)
        
        # Get embedding
        embedding = self.generate_embedding(documentation)
        
        # Insert into documentation collection
        self.qdrant_client.upsert(
            collection_name=self.docs_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "documentation": documentation
                    }
                )
            ]
        )
        
        return f"{point_id}-d"
    
    def record_query_attempt(self, question: str, sql: str, success: bool, error_message: str = None, 
                           retry_count: int = 0, data: pd.DataFrame = None, columns: List[str] = None,
                           visualization = None, summary: str = None, total_time_ms: float = None,
                           sql_generation_time_ms: float = None, sql_execution_time_ms: float = None,
                           visualization_time_ms: float = None, explanation_time_ms: float = None,
                           timing_details: Dict = None):
        """
        Record a query attempt in the history.
        
        Args:
            question: The natural language question
            sql: The SQL query
            success: Whether the query succeeded
            error_message: Error message if the query failed
            retry_count: Number of retries performed
            data: DataFrame result of the query (for successful queries)
            columns: DataFrame columns (for successful queries)
            visualization: Plotly visualization (for successful queries)
            summary: Natural language summary of the results (for successful queries)
            total_time_ms: Total query time in milliseconds
            sql_generation_time_ms: Time to generate SQL in milliseconds
            sql_execution_time_ms: Time to execute SQL in milliseconds
            visualization_time_ms: Time to generate visualization in milliseconds
            explanation_time_ms: Time to generate explanation in milliseconds
            timing_details: Detailed timing information as a dictionary
        """
        if not self.save_query_history:
            return
            
        try:
            import sqlite3
            import pickle
            import json
            
            # Generate a unique ID
            entry_id = str(hash(f"{question}_{datetime.datetime.now().isoformat()}"))
            timestamp = datetime.datetime.now().isoformat()
            
            # Connect to the history database
            history_conn = sqlite3.connect(self.history_db_path)
            cursor = history_conn.cursor()
            
            # Convert DataFrame and visualization to binary if they exist
            data_blob = None
            if success and data is not None:
                data_blob = pickle.dumps(data)
                
            columns_json = None
            if columns:
                columns_json = json.dumps(columns)
                
            vis_blob = None
            if success and visualization is not None:
                vis_blob = pickle.dumps(visualization)
                
            # Convert timing details to JSON if it exists
            timing_details_json = None
            if timing_details:
                timing_details_json = json.dumps(timing_details)
                
            # Insert the record
            cursor.execute(
                '''
                INSERT INTO query_history 
                (id, timestamp, question, sql, success, error_message, retry_count, data, columns, 
                visualization, summary, total_time_ms, sql_generation_time_ms, sql_execution_time_ms,
                visualization_time_ms, explanation_time_ms, timing_details) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (entry_id, timestamp, question, sql, 1 if success else 0, error_message, retry_count,
                 data_blob, columns_json, vis_blob, summary, total_time_ms, sql_generation_time_ms,
                 sql_execution_time_ms, visualization_time_ms, explanation_time_ms, timing_details_json)
            )
            
            history_conn.commit()
            history_conn.close()
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error recording query history to database: {e}")
                import traceback
                traceback.print_exc()
    
    def get_query_history(self, successful_only: bool = False, with_errors_only: bool = False, limit: int = None):
        """
        Get the query history.
        
        Args:
            successful_only: Only return successful queries
            with_errors_only: Only return queries that had errors
            limit: Maximum number of queries to return
            
        Returns:
            List of query history entries
        """
        if not self.save_query_history:
            return []
            
        try:
            import sqlite3
            import pickle
            import json
            
            # Connect to the history database
            history_conn = sqlite3.connect(self.history_db_path)
            cursor = history_conn.cursor()
            
            # Build the query
            query = "SELECT * FROM query_history"
            conditions = []
            
            if successful_only:
                conditions.append("success = 1")
                
            if with_errors_only:
                conditions.append("error_message IS NOT NULL")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                
            # Execute the query
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Convert rows to dictionaries
            result = []
            for row in rows:
                # Create a dict with all columns
                entry = dict(zip(column_names, row))
                
                # Convert success to boolean
                entry["success"] = bool(entry.get("success", 0))
                
                # Deserialize data if it exists
                if entry.get("data"):
                    try:
                        entry["data"] = pickle.loads(entry["data"])
                    except:
                        entry["data"] = None
                
                # Deserialize columns if they exist
                if entry.get("columns"):
                    try:
                        entry["columns"] = json.loads(entry["columns"])
                    except:
                        entry["columns"] = None
                
                # Deserialize visualization if it exists
                if entry.get("visualization"):
                    try:
                        entry["visualization"] = pickle.loads(entry["visualization"])
                    except:
                        entry["visualization"] = None
                
                # Deserialize timing details if they exist
                if entry.get("timing_details"):
                    try:
                        entry["timing_details"] = json.loads(entry["timing_details"])
                    except:
                        entry["timing_details"] = None
                
                result.append(entry)
            
            history_conn.close()
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error retrieving query history from database: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def analyze_error_patterns(self):
        """
        Analyze error patterns in query history.
        
        Returns:
            Dictionary with error analysis
        """
        history = self.get_query_history()
        
        if not history:
            return {"message": "No query history available for analysis"}
        
        # Count total queries and errors
        total_queries = len(history)
        error_queries = len([q for q in history if not q["success"]])
        retried_queries = len([q for q in history if q["retry_count"] > 0])
        successful_retries = len([q for q in history if q["retry_count"] > 0 and q["success"]])
        
        # Group errors by type
        error_types = {}
        for query in history:
            if query["error_message"]:
                # Extract error type (first line or up to first colon)
                error_type = query["error_message"].split('\n')[0]
                if ':' in error_type:
                    error_type = error_type.split(':', 1)[0]
                
                if error_type in error_types:
                    error_types[error_type] += 1
                else:
                    error_types[error_type] = 1
        
        # Calculate retry effectiveness
        retry_success_rate = (successful_retries / retried_queries) if retried_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "error_queries": error_queries,
            "error_rate": error_queries / total_queries if total_queries > 0 else 0,
            "retried_queries": retried_queries,
            "successful_retries": successful_retries,
            "retry_success_rate": retry_success_rate,
            "common_error_types": sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        }
    
    def explain_sql(self, sql: str) -> str:
        """
        Generate a detailed explanation of a SQL query with additional context.
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Natural language explanation
        """
        # Get related schema information if available
        schema_info = ""
        try:
            # Extract table names from the SQL query to find relevant schema info
            import re
            tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
            tables = [t[0] if t[0] else t[1] for t in tables]
            
            if tables and self.conn:
                for table in tables:
                    try:
                        # Get schema for this table
                        schema_df = self.run_sql(f"PRAGMA table_info({table})")
                        if not schema_df.empty:
                            schema_info += f"\nTable '{table}' columns: {', '.join(schema_df['name'].tolist())}\n"
                    except:
                        pass
        except:
            pass

        # Create a more detailed prompt with schema info if available
        if schema_info:
            prompt = [
                self.system_message(
                    "You are an expert SQL educator. Explain SQL queries in clear, simple terms. "
                    "Include how the query works and what it's trying to accomplish."
                ),
                self.user_message(
                    f"Please explain this SQL query in simple terms:\n\n{sql}\n\n"
                    f"Additional schema information:{schema_info}"
                )
            ]
        else:
            # Fall back to parent class behavior if no schema info
            prompt = [
                self.system_message("You are an expert SQL educator. Explain SQL queries in clear, simple terms."),
                self.user_message(f"Please explain this SQL query in simple terms:\n\n{sql}")
            ]
        
        return self.submit_prompt(prompt)
    
    def generate_follow_up_questions(self, question: str, sql: str, result_info: str, n=3) -> List[str]:
        """
        Generate follow-up questions based on previous query with enhanced context.
        
        Args:
            question: Original question
            sql: SQL query used
            result_info: Information about query results
            n: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        # Get database schema info to enhance follow-up relevance
        schema_info = ""
        try:
            if self.conn:
                # Get list of all tables
                tables_df = self.run_sql("""
                    SELECT name 
                    FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                
                if not tables_df.empty:
                    table_list = tables_df['name'].tolist()
                    schema_info = f"Available tables: {', '.join(table_list)}\n\n"
                    
                    # Also include tables extracted from the current query
                    import re
                    current_tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
                    current_tables = [t[0] if t[0] else t[1] for t in current_tables]
                    
                    for table in current_tables:
                        if table in table_list:
                            try:
                                # Get column info for this table
                                columns_df = self.run_sql(f"PRAGMA table_info({table})")
                                if not columns_df.empty:
                                    schema_info += f"Table '{table}' columns: {', '.join(columns_df['name'].tolist())}\n"
                            except:
                                pass
        except:
            pass
            
        # Add schema info to the prompt if available
        system_msg = (
            f"You are a data analyst helping with SQL queries. "
            f"The user asked: '{question}'\n\n"
            f"The SQL query used was: {sql}\n\n"
            f"Results information: {result_info}"
        )
        
        if schema_info:
            system_msg += f"\n\nAdditional database information:\n{schema_info}"
            
        prompt = [
            self.system_message(system_msg),
            self.user_message(
                f"Generate {n} natural follow-up questions that would be logical next steps for analysis. "
                f"Each question should be answerable with SQL and relevant to the available database tables. "
                f"Make questions diverse to explore different aspects of the data. "
                f"Return only the questions, one per line."
            )
        ]
        
        response = self.submit_prompt(prompt)
        
        # Split into list of questions
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
        
        # Limit to n questions
        return questions[:n]
    
    def smart_query(self, question: str, print_results: bool = True, visualize: bool = True):
        """
        Execute a query with automatic retry mechanism and detailed reporting.
        
        Args:
            question: Natural language question
            print_results: Whether to print results
            visualize: Whether to generate visualization
            
        Returns:
            Dictionary with query results and metadata
        """
        # Track query metadata and timing
        metadata = {
            "question": question,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": False,
            "retry_count": 0,
            "error_message": None,
            "original_sql": None,
            "final_sql": None,
        }
        
        timing = {
            "start_time": datetime.datetime.now(),
            "sql_generation_start": None,
            "sql_generation_end": None,
            "sql_execution_start": None,
            "sql_execution_end": None,
            "visualization_start": None,
            "visualization_end": None,
            "explanation_start": None,
            "explanation_end": None,
            "end_time": None
        }
        
        # Function to calculate elapsed time in ms
        def elapsed_ms(start, end):
            if start and end:
                return (end - start).total_seconds() * 1000
            return None
        
        # Generate initial SQL
        try:
            timing["sql_generation_start"] = datetime.datetime.now()
            sql = self.generate_sql(question)
            timing["sql_generation_end"] = datetime.datetime.now()
            metadata["original_sql"] = sql
        except Exception as e:
            error_message = str(e)
            metadata["error_message"] = error_message
            
            timing["end_time"] = datetime.datetime.now()
            total_time_ms = elapsed_ms(timing["start_time"], timing["end_time"])
            sql_generation_time_ms = elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"])
            
            # Record the failed attempt with timing data
            self.record_query_attempt(
                question=question, 
                sql=None, 
                success=False, 
                error_message=error_message,
                total_time_ms=total_time_ms,
                sql_generation_time_ms=sql_generation_time_ms,
                timing_details={
                    "total_ms": total_time_ms,
                    "sql_generation_ms": sql_generation_time_ms
                }
            )
            
            if print_results:
                print(f"Error generating SQL: {error_message}")
                
            return {
                "success": False,
                "error": error_message,
                "metadata": metadata,
                "timing": {
                    "total_ms": total_time_ms,
                    "sql_generation_ms": sql_generation_time_ms
                }
            }
            
        # Print the initial SQL
        if print_results:
            try:
                from IPython.display import display, Code
                print("Generated SQL:")
                display(Code(sql))
            except ImportError:
                print(f"Generated SQL: {sql}")
                
        # Check if database connection is set
        if not self.run_sql_is_set:
            metadata["error_message"] = "No database connection"
            
            timing["end_time"] = datetime.datetime.now()
            total_time_ms = elapsed_ms(timing["start_time"], timing["end_time"])
            sql_generation_time_ms = elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"])
            
            self.record_query_attempt(
                question=question, 
                sql=sql, 
                success=False, 
                error_message=metadata["error_message"],
                total_time_ms=total_time_ms,
                sql_generation_time_ms=sql_generation_time_ms,
                timing_details={
                    "total_ms": total_time_ms,
                    "sql_generation_ms": sql_generation_time_ms
                }
            )
            
            if print_results:
                print("No database connection. Connect to a database to run queries.")
                
            return {
                "success": False,
                "sql": sql,
                "error": "No database connection",
                "metadata": metadata,
                "timing": {
                    "total_ms": total_time_ms,
                    "sql_generation_ms": sql_generation_time_ms
                }
            }
        
        # Execute SQL with retry mechanism
        current_sql = sql
        retry_count = 0
        df = None
        fig = None
        summary = None
        
        while retry_count <= self.max_retry_attempts:
            try:
                if print_results and retry_count > 0:
                    print(f"\nRetry attempt {retry_count}/{self.max_retry_attempts}:")
                    try:
                        from IPython.display import display, Code
                        display(Code(current_sql))
                    except ImportError:
                        print(f"SQL: {current_sql}")
                
                # Execute the SQL
                timing["sql_execution_start"] = datetime.datetime.now()
                df = self.run_sql(current_sql)
                timing["sql_execution_end"] = datetime.datetime.now()
                
                # Success! Break out of retry loop
                metadata["success"] = True
                metadata["final_sql"] = current_sql
                metadata["retry_count"] = retry_count
                
                # Generate visualization if requested
                if visualize and df is not None and self.should_generate_visualization(df):
                    try:
                        if self.debug_mode:
                            print(f"Attempting to generate visualization for query: '{question}'")
                            print(f"DataFrame shape: {df.shape}")
                            print(f"DataFrame columns: {df.columns.tolist()}")
                            print(f"DataFrame sample:\n{df.head(3)}")
                        
                        timing["visualization_start"] = datetime.datetime.now()
                        plotly_code = self.generate_plotly_code(
                            question=question,
                            sql=current_sql,
                            df_metadata=f"DataFrame info: {df.dtypes}"
                        )
                        
                        if self.debug_mode:
                            print(f"Generated Plotly code:\n{plotly_code}")
                        
                        fig = self.get_plotly_figure(plotly_code, df)
                        timing["visualization_end"] = datetime.datetime.now()
                        
                        if print_results:
                            try:
                                from IPython.display import Image
                                img_bytes = fig.to_image(format="png", scale=2)
                                display(Image(img_bytes))
                            except ImportError:
                                # Prevent opening in a new browser tab by setting auto_open to False
                                # fig.show(config={'displayModeBar': True, 'showLink': False}, auto_open=False)
                                pass
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Visualization error: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Generate summary
                if df is not None and len(df) > 0:
                    try:
                        timing["explanation_start"] = datetime.datetime.now()
                        summary = self.explain_results(question, df)
                        timing["explanation_end"] = datetime.datetime.now()
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Summary generation error: {e}")
                
                # Calculate timing information
                timing["end_time"] = datetime.datetime.now()
                total_time_ms = elapsed_ms(timing["start_time"], timing["end_time"])
                sql_generation_time_ms = elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"])
                sql_execution_time_ms = elapsed_ms(timing["sql_execution_start"], timing["sql_execution_end"])
                visualization_time_ms = elapsed_ms(timing["visualization_start"], timing["visualization_end"])
                explanation_time_ms = elapsed_ms(timing["explanation_start"], timing["explanation_end"])
                
                timing_details = {
                    "total_ms": total_time_ms,
                    "sql_generation_ms": sql_generation_time_ms,
                    "sql_execution_ms": sql_execution_time_ms,
                    "visualization_ms": visualization_time_ms,
                    "explanation_ms": explanation_time_ms
                }
                
                # Record the successful attempt with all data
                self.record_query_attempt(
                    question=question,
                    sql=current_sql,
                    success=True,
                    retry_count=retry_count,
                    data=df,
                    columns=list(df.columns) if df is not None else None,
                    visualization=fig,
                    summary=summary,
                    total_time_ms=total_time_ms,
                    sql_generation_time_ms=sql_generation_time_ms,
                    sql_execution_time_ms=sql_execution_time_ms,
                    visualization_time_ms=visualization_time_ms,
                    explanation_time_ms=explanation_time_ms,
                    timing_details=timing_details
                )
                
                break
                
            except Exception as e:
                error_message = str(e)
                
                # Record time of error
                sql_execution_time_ms = None
                if timing["sql_execution_start"]:
                    error_time = datetime.datetime.now()
                    sql_execution_time_ms = elapsed_ms(timing["sql_execution_start"], error_time)
                
                # Record failed attempt
                self.record_query_attempt(
                    question=question,
                    sql=current_sql,
                    success=False,
                    error_message=error_message,
                    retry_count=retry_count,
                    total_time_ms=elapsed_ms(timing["start_time"], datetime.datetime.now()),
                    sql_generation_time_ms=elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"]),
                    sql_execution_time_ms=sql_execution_time_ms,
                    timing_details={
                        "total_ms": elapsed_ms(timing["start_time"], datetime.datetime.now()),
                        "sql_generation_ms": elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"]),
                        "sql_execution_ms": sql_execution_time_ms
                    }
                )
                
                if print_results:
                    print(f"\nSQL Error: {error_message}")
                
                # Check if we've reached max retries
                retry_count += 1
                if retry_count > self.max_retry_attempts:
                    metadata["error_message"] = f"Failed after {self.max_retry_attempts} attempts. Last error: {error_message}"
                    metadata["retry_count"] = retry_count - 1
                    
                    if print_results:
                        print(f"Maximum retry attempts ({self.max_retry_attempts}) exceeded.")
                        
                    break
                
                # Generate corrected SQL
                if print_results:
                    print(f"Attempting to fix query...")
                    
                # Track SQL correction time
                timing["sql_generation_start"] = datetime.datetime.now()
                current_sql = self.generate_sql_with_error_context(
                    question=question,
                    previous_sql=current_sql,
                    error_message=error_message
                )
                timing["sql_generation_end"] = datetime.datetime.now()
        
        # If execution failed after all retries
        if not metadata["success"]:
            timing["end_time"] = datetime.datetime.now()
            total_time_ms = elapsed_ms(timing["start_time"], timing["end_time"])
            
            return {
                "success": False,
                "sql": metadata["original_sql"],
                "corrected_sql": current_sql,
                "error": metadata["error_message"],
                "retry_count": metadata["retry_count"],
                "metadata": metadata,
                "timing": {
                    "total_ms": total_time_ms,
                    "sql_generation_ms": elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"]),
                    "sql_execution_ms": elapsed_ms(timing["sql_execution_start"], timing["sql_execution_end"]),
                }
            }
        
        # Execution succeeded
        if print_results:
            try:
                from IPython.display import display
                display(df)
            except ImportError:
                print(df)
        
        # Add to training if successful
        if df is not None and len(df) > 0:
            self.add_question_sql(question, metadata["final_sql"])
        
        # Calculate final timing information
        timing["end_time"] = datetime.datetime.now()
        total_time_ms = elapsed_ms(timing["start_time"], timing["end_time"])
        sql_generation_time_ms = elapsed_ms(timing["sql_generation_start"], timing["sql_generation_end"])
        sql_execution_time_ms = elapsed_ms(timing["sql_execution_start"], timing["sql_execution_end"])
        visualization_time_ms = elapsed_ms(timing["visualization_start"], timing["visualization_end"])
        explanation_time_ms = elapsed_ms(timing["explanation_start"], timing["explanation_end"])
        
        timing_details = {
            "total_ms": total_time_ms,
            "sql_generation_ms": sql_generation_time_ms,
            "sql_execution_ms": sql_execution_time_ms,
            "visualization_ms": visualization_time_ms,
            "explanation_ms": explanation_time_ms
        }
        
        # Return successful result with timing information
        return {
            "success": True,
            "sql": metadata["original_sql"],
            "final_sql": metadata["final_sql"] if metadata["final_sql"] != metadata["original_sql"] else None,
            "retry_count": metadata["retry_count"],
            "data": df,
            "visualization": fig,
            "summary": summary,
            "metadata": metadata,
            "timing": timing_details
        }
    
    def analyze_data(self, 
                    question: str, 
                    visualize: bool = True, 
                    explain: bool = True,
                    suggest_followups: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis function with retry mechanism.
        
        Args:
            question: Natural language question
            visualize: Whether to generate visualization
            explain: Whether to generate explanation
            suggest_followups: Whether to suggest follow-up questions
            
        Returns:
            Dictionary with analysis results
        """
        # Use smart_query to handle retries
        result = self.smart_query(question, print_results=False, visualize=visualize)
        
        # If query failed, return error information
        if not result["success"]:
            return result
        
        # Extract results
        df = result["data"]
        sql = result["final_sql"] or result["sql"]
        
        # Add explanation if requested
        if explain and df is not None and len(df) > 0:
            result["explanation"] = self.explain_results(question, df)
        
        # Generate follow-up questions if requested
        if suggest_followups and df is not None:
            df_info = f"Columns: {', '.join(df.columns)}"
            followups = self.generate_follow_up_questions(
                question=question,
                sql=sql,
                result_info=df_info
            )
            result["followup_questions"] = followups
        
        return result
    
    def explain_results(self, question: str, df: pd.DataFrame) -> str:
        """
        Generate explanation of query results.
        
        Args:
            question: Original question
            df: Query results DataFrame
            
        Returns:
            Natural language explanation of results
        """
        # Limit the DataFrame representation for the prompt
        df_str = df.head(10).to_markdown() if len(df) > 10 else df.to_markdown()
        
        prompt = [
            self.system_message(
                f"You are a data analyst explaining query results. "
                f"The user asked: '{question}'\n\n"
                f"Query results:\n{df_str}"
            ),
            self.user_message(
                "Please provide a concise explanation of these results that answers the question. "
                "Highlight any notable patterns, outliers, or insights."
            )
        ]
        
        return self.submit_prompt(prompt)
    
    def demo(self, question: str = None) -> None:
        """
        Run a demonstration of Talk2SQL capabilities with retry mechanism.
        
        Args:
            question: Optional question to start with
        """
        if not question:
            question = input("Enter your question: ")
        
        print(f"\n📝 Question: {question}\n")
        
        # Use smart_query to handle retries
        result = self.smart_query(question, print_results=True)
        
        if result["success"]:
            print("\n✅ Query succeeded!")
            
            if result["retry_count"] > 0:
                print(f"🛠️ Query was fixed after {result['retry_count']} retry attempts")
                
            # Generate explanation
            df = result["data"]
            sql = result["final_sql"] or result["sql"]
            
            explanation = self.explain_results(question, df)
            print(f"\n💡 Explanation:\n{explanation}")
            
            # Follow-up questions
            followups = self.generate_follow_up_questions(
                question=question,
                sql=sql,
                result_info=f"Columns: {', '.join(df.columns)}"
            )
            
            print("\n🔄 Follow-up questions:")
            for i, q in enumerate(followups, 1):
                print(f"{i}. {q}")
        else:
            print(f"\n❌ Query failed after {result['retry_count']} retry attempts")
            print(f"Error: {result['error']}")
            
        print("\n✨ Demo complete!")

    # Override _setup_collections method to use qdrant_client instead of client
    def _setup_collections(self):
        """Create collections if they don't exist."""
        # Questions collection
        if not self.qdrant_client.collection_exists(self.questions_collection):
            self.qdrant_client.create_collection(
                collection_name=self.questions_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            
        # Schema collection
        if not self.qdrant_client.collection_exists(self.schema_collection):
            self.qdrant_client.create_collection(
                collection_name=self.schema_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            
        # Documentation collection
        if not self.qdrant_client.collection_exists(self.docs_collection):
            self.qdrant_client.create_collection(
                collection_name=self.docs_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )

    def get_similar_questions(self, question: str) -> list:
        """
        Get similar questions with their SQL from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of question-SQL pairs
        """
        embedding = self.generate_embedding(question)
        
        results = self.qdrant_client.search(
            collection_name=self.questions_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload for point in results]
    
    def get_related_schema(self, question: str) -> list:
        """
        Get related schema information from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of schema strings
        """
        embedding = self.generate_embedding(question)
        
        results = self.qdrant_client.search(
            collection_name=self.schema_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload["schema"] for point in results]
    
    def get_related_documentation(self, question: str) -> list:
        """
        Get related documentation from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of documentation strings
        """
        embedding = self.generate_embedding(question)
        
        results = self.qdrant_client.search(
            collection_name=self.docs_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload["documentation"] for point in results]

    def get_all_training_data(self) -> pd.DataFrame:
        """
        Get all training data as a DataFrame.
        
        Returns:
            DataFrame with all training data
        """
        # Initialize empty DataFrame
        df = pd.DataFrame(columns=["id", "type", "question", "content"])
        
        # Get questions
        questions = self.qdrant_client.scroll(
            collection_name=self.questions_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for point in questions:
            df = pd.concat([df, pd.DataFrame([{
                "id": f"{point.id}-q",
                "type": "question",
                "question": point.payload["question"],
                "content": point.payload["sql"]
            }])], ignore_index=True)
        
        # Get schema
        schemas = self.qdrant_client.scroll(
            collection_name=self.schema_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for point in schemas:
            df = pd.concat([df, pd.DataFrame([{
                "id": f"{point.id}-s",
                "type": "schema",
                "question": None,
                "content": point.payload["schema"]
            }])], ignore_index=True)
        
        # Get documentation
        docs = self.qdrant_client.scroll(
            collection_name=self.docs_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for point in docs:
            df = pd.concat([df, pd.DataFrame([{
                "id": f"{point.id}-d",
                "type": "documentation",
                "question": None,
                "content": point.payload["documentation"]
            }])], ignore_index=True)
        
        return df
    
    def remove_training_data(self, id: str) -> bool:
        """
        Remove training data by ID.
        
        Args:
            id: ID of training data to remove
            
        Returns:
            True if successful
        """
        try:
            # Parse ID to get collection
            if id.endswith("-q"):
                collection = self.questions_collection
                real_id = id[:-2]
            elif id.endswith("-s"):
                collection = self.schema_collection
                real_id = id[:-2]
            elif id.endswith("-d"):
                collection = self.docs_collection
                real_id = id[:-2]
            else:
                return False
            
            # Delete from collection
            self.qdrant_client.delete(
                collection_name=collection,
                points_selector=[real_id]
            )
            
            return True
        except Exception as e:
            print(f"Error removing training data: {e}")
            return False
    
    def reset_collection(self, collection_type: str) -> bool:
        """
        Reset a collection to empty state.
        
        Args:
            collection_type: Type of collection ("questions", "schema", "docs")
            
        Returns:
            True if successful
        """
        try:
            if collection_type == "questions":
                self.qdrant_client.delete_collection(self.questions_collection)
            elif collection_type == "schema":
                self.qdrant_client.delete_collection(self.schema_collection)
            elif collection_type == "docs":
                self.qdrant_client.delete_collection(self.docs_collection)
            else:
                return False
            
            self._setup_collections()
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False

    def get_plotly_figure(self, plotly_code: str, df: pd.DataFrame) -> go.Figure:
        """
        Execute Plotly code to generate figure.
        
        Args:
            plotly_code: Python code for Plotly visualization
            df: DataFrame to visualize
            
        Returns:
            Plotly figure
        """
        # Create local namespace for execution
        local_vars = {"df": df, "go": go, "pd": pd}
        
        try:
            # Execute the generated code in the local namespace
            exec(plotly_code, globals(), local_vars)
            
            # Get the figure from local namespace
            fig = local_vars.get("fig")
            
            if fig is None:
                # Fallback: create a simple figure
                fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1], mode="markers"))
                fig.update_layout(title="Error: Visualization code did not produce a figure")
                
            return fig
            
        except Exception as e:
            # If execution fails, create an error figure
            fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1], mode="markers"))
            fig.update_layout(title=f"Error: {str(e)}")
            return fig 

    def generate_data_summary(self, question: str, result: dict) -> str:
        """
        Generate a summary of the data using Anthropic Claude.
        
        Args:
            question: The original question
            result: Dictionary containing query results and SQL
            
        Returns:
            A natural language summary of the data
        """
        try:
            sql = result.get("sql", "")
            df = result.get("data")
            
            if df is None or len(df) == 0:
                return "No data returned from the query."
            
            # Create a prompt for Claude to summarize the data
            system_message = """You are an expert SQL developer who summarizes data and answers questions.
            You will be provided with a question, a SQL query, and a dataframe.
            Please provide a clear, concise summary of this data that directly answers the user's question, citing the tables and columns used to answer the question.
            Include key insights, trends, or patterns if relevant. Keep it brief and focused."""
            
            user_message = f"""The user asked: "{question}"
            
            I ran the following SQL query:
            {sql}
            
            The query returned a dataframe with {len(df)} rows and {len(df.columns)} columns.
            Column names: {', '.join(df.columns)}
            
            Here's the dataframe:
            {df.to_string()}"""
            
            # Create the messages for the prompt
            prompt = [
                self.system_message(system_message),
                self.user_message(user_message)
            ]
            
            # Use Anthropic Claude to generate the summary
            summary = self.submit_prompt(prompt)
            return summary
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error generating summary: {e}")
                import traceback
                traceback.print_exc()
            return "Sorry, I couldn't generate a summary of the data." 