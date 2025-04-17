from typing import Dict, List, Union, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import datetime

from .qdrant_store import QdrantVectorStore
from .anthropic_llm import AnthropicLLM

class SQLMind(QdrantVectorStore, AnthropicLLM):
    """
    Main SQLMind engine that combines Qdrant vector storage and Anthropic LLM.
    This is the primary class to use for text-to-SQL applications.
    """
    
    def __init__(self, config=None):
        """
        Initialize SQLMind with configuration.
        
        Args:
            config: Configuration dictionary with options for both QdrantVectorStore and AnthropicLLM
              - max_retry_attempts: Maximum number of retries for failed SQL queries (default: 3)
              - save_query_history: Whether to save query history with errors and retries (default: True)
        """
        config = config or {}
        
        # Initialize both parent classes
        QdrantVectorStore.__init__(self, config)
        AnthropicLLM.__init__(self, config)
        
        # Additional configuration
        self.debug_mode = config.get("debug_mode", False)
        self.auto_visualization = config.get("auto_visualization", True)
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        self.save_query_history = config.get("save_query_history", True)
        
        # Initialize query history
        self.query_history = []
    
    def record_query_attempt(self, question: str, sql: str, success: bool, error_message: str = None, retry_count: int = 0):
        """
        Record a query attempt in the history.
        
        Args:
            question: The natural language question
            sql: The SQL query
            success: Whether the query succeeded
            error_message: Error message if the query failed
            retry_count: Number of retries performed
        """
        if self.save_query_history:
            self.query_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "question": question,
                "sql": sql,
                "success": success,
                "error_message": error_message,
                "retry_count": retry_count,
            })
    
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
        result = self.query_history
        
        if successful_only:
            result = [q for q in result if q["success"]]
            
        if with_errors_only:
            result = [q for q in result if q["error_message"] is not None]
            
        if limit:
            result = result[-limit:]
            
        return result
    
    def analyze_error_patterns(self):
        """
        Analyze error patterns in query history.
        
        Returns:
            Dictionary with error analysis
        """
        if not self.query_history:
            return {"message": "No query history available for analysis"}
        
        # Count total queries and errors
        total_queries = len(self.query_history)
        error_queries = len([q for q in self.query_history if not q["success"]])
        retried_queries = len([q for q in self.query_history if q["retry_count"] > 0])
        successful_retries = len([q for q in self.query_history if q["retry_count"] > 0 and q["success"]])
        
        # Group errors by type
        error_types = {}
        for query in self.query_history:
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
        # Track query metadata
        metadata = {
            "question": question,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": False,
            "retry_count": 0,
            "error_message": None,
            "original_sql": None,
            "final_sql": None,
        }
        
        # Generate initial SQL
        try:
            sql = self.generate_sql(question)
            metadata["original_sql"] = sql
        except Exception as e:
            error_message = str(e)
            metadata["error_message"] = error_message
            self.record_query_attempt(question, None, False, error_message)
            
            if print_results:
                print(f"Error generating SQL: {error_message}")
                
            return {
                "success": False,
                "error": error_message,
                "metadata": metadata
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
            self.record_query_attempt(question, sql, False, metadata["error_message"])
            
            if print_results:
                print("No database connection. Connect to a database to run queries.")
                
            return {
                "success": False,
                "sql": sql,
                "error": "No database connection",
                "metadata": metadata
            }
        
        # Execute SQL with retry mechanism
        current_sql = sql
        retry_count = 0
        df = None
        
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
                df = self.run_sql(current_sql)
                
                # Success! Break out of retry loop
                metadata["success"] = True
                metadata["final_sql"] = current_sql
                metadata["retry_count"] = retry_count
                
                self.record_query_attempt(
                    question=question,
                    sql=current_sql,
                    success=True,
                    retry_count=retry_count
                )
                
                break
                
            except Exception as e:
                error_message = str(e)
                
                # Record failed attempt
                self.record_query_attempt(
                    question=question,
                    sql=current_sql,
                    success=False,
                    error_message=error_message,
                    retry_count=retry_count
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
                    
                current_sql = self.generate_sql_with_error_context(
                    question=question,
                    previous_sql=current_sql,
                    error_message=error_message
                )
        
        # If execution failed after all retries
        if not metadata["success"]:
            return {
                "success": False,
                "sql": metadata["original_sql"],
                "corrected_sql": current_sql,
                "error": metadata["error_message"],
                "retry_count": metadata["retry_count"],
                "metadata": metadata
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
            
        # Generate visualization
        fig = None
        if visualize and df is not None and self.should_generate_visualization(df):
            try:
                plotly_code = self.generate_plotly_code(
                    question=question,
                    sql=metadata["final_sql"],
                    df_metadata=f"DataFrame info: {df.dtypes}"
                )
                fig = self.get_plotly_figure(plotly_code, df)
                
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
                if print_results:
                    print(f"Visualization error: {e}")
        
        # Return successful result
        return {
            "success": True,
            "sql": metadata["original_sql"],
            "final_sql": metadata["final_sql"] if metadata["final_sql"] != metadata["original_sql"] else None,
            "retry_count": metadata["retry_count"],
            "data": df,
            "visualization": fig,
            "metadata": metadata
        }
    
    def suggest_query_improvements(self, failed_sql, error_message):
        """
        Generate suggestions for improving a failed SQL query.
        
        Args:
            failed_sql: The SQL query that failed
            error_message: The error message
            
        Returns:
            Dictionary with improvement suggestions
        """
        prompt = [
            self.system_message(
                "You are an expert SQL engineer. You'll be shown a failed SQL query and its error message. "
                "Please provide a detailed analysis of what went wrong and suggest improvements. "
                "Focus on the root cause and explain your reasoning clearly."
            ),
            self.user_message(
                f"SQL Query:\n```sql\n{failed_sql}\n```\n\n"
                f"Error Message:\n{error_message}\n\n"
                "Please analyze the error and provide recommendations on how to fix it."
            )
        ]
        
        analysis = self.submit_prompt(prompt)
        
        # Generate improved SQL
        prompt = [
            self.system_message(
                "You are an expert SQL engineer. You'll be shown a failed SQL query and its error message. "
                "Please provide a corrected version of the SQL query that addresses the issue. "
                "Return only the SQL code without explanations."
            ),
            self.user_message(
                f"SQL Query:\n```sql\n{failed_sql}\n```\n\n"
                f"Error Message:\n{error_message}\n\n"
                "Please provide the corrected SQL query."
            )
        ]
        
        corrected_sql = self.extract_sql(self.submit_prompt(prompt))
        
        return {
            "original_sql": failed_sql,
            "error_message": error_message,
            "analysis": analysis,
            "suggested_fix": corrected_sql
        }
    
    def train_from_errors(self, limit: int = 10):
        """
        Analyze recent errors and generate training material to improve future queries.
        
        Args:
            limit: Maximum number of errors to analyze
            
        Returns:
            Dictionary with training recommendations
        """
        # Get recent errors
        errors = [q for q in self.query_history if not q["success"]][-limit:]
        
        if not errors:
            return {"message": "No errors found in query history"}
        
        # Analyze each error
        analyses = []
        for error in errors:
            prompt = [
                self.system_message(
                    "You are an SQL education expert. You'll be shown a failed SQL query, its error message, "
                    "and the natural language question it was trying to answer. "
                    "Please identify patterns that could be addressed through better training examples."
                ),
                self.user_message(
                    f"Question: {error['question']}\n\n"
                    f"SQL Query:\n```sql\n{error['sql']}\n```\n\n"
                    f"Error Message:\n{error['error_message']}\n\n"
                    "What SQL patterns or concepts should be added to training data to address this error?"
                )
            ]
            
            analysis = self.submit_prompt(prompt)
            analyses.append({
                "question": error["question"],
                "sql": error["sql"],
                "error": error["error_message"],
                "analysis": analysis
            })
        
        # Generate overall recommendations
        prompt = [
            self.system_message(
                "You are an SQL education expert. Based on the error patterns you've analyzed, "
                "provide recommendations for improving the SQL training data."
            ),
            self.user_message(
                "Based on the following error analyses, what training examples or concepts "
                "should be added to improve SQL generation?\n\n" +
                "\n\n".join([f"Error {i+1}:\n{a['analysis']}" for i, a in enumerate(analyses)])
            )
        ]
        
        recommendations = self.submit_prompt(prompt)
        
        return {
            "error_analyses": analyses,
            "recommendations": recommendations
        }
    
    def initialize_with_schema(self, schema_files: List[str]) -> None:
        """
        Initialize the system with database schema files.
        
        Args:
            schema_files: List of paths to schema files (SQL DDL)
        """
        for file_path in schema_files:
            try:
                with open(file_path, 'r') as file:
                    schema = file.read()
                    self.add_schema(schema)
                    print(f"Added schema from {file_path}")
            except Exception as e:
                print(f"Error loading schema from {file_path}: {e}")
    
    def initialize_with_documentation(self, doc_files: List[str]) -> None:
        """
        Initialize the system with documentation files.
        
        Args:
            doc_files: List of paths to documentation files (text/markdown)
        """
        for file_path in doc_files:
            try:
                with open(file_path, 'r') as file:
                    doc = file.read()
                    self.add_documentation(doc)
                    print(f"Added documentation from {file_path}")
            except Exception as e:
                print(f"Error loading documentation from {file_path}: {e}")
    
    def initialize_with_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Initialize the system with question-SQL examples.
        
        Args:
            examples: List of dictionaries with "question" and "sql" keys
        """
        for example in examples:
            if "question" in example and "sql" in example:
                self.add_question_sql(example["question"], example["sql"])
                print(f"Added example: {example['question']}")
    
    def get_sql(self, question: str) -> str:
        """
        Get SQL for a question without executing it.
        
        Args:
            question: Natural language question
            
        Returns:
            Generated SQL query
        """
        return self.generate_sql(question)
    
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
    
    def demo(self, question: str = None) -> None:
        """
        Run a demonstration of SQLMind capabilities with retry mechanism.
        
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
            
            # Suggest improvements
            improvements = self.suggest_query_improvements(
                result["corrected_sql"] or result["sql"],
                result["error"]
            )
            
            print(f"\n🔍 Error Analysis:\n{improvements['analysis']}")
            
        print("\n✨ Demo complete!")