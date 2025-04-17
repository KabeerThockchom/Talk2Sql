from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Callable, Dict, Any, Optional, Iterator
import pandas as pd
import plotly.graph_objects as go
import sqlparse
import re
import traceback
from talk2sql.exceptions import SQLParsingError

class Talk2SQLBase(ABC):
    """Abstract base class for Talk2SQL that defines core interfaces and functionality."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 8000)
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        # Initialize streaming pipeline if enabled
        self._streaming_pipeline = None
        self._enable_streaming = self.config.get("enable_streaming", False)
        self._enable_threading = self.config.get("enable_threading", False)
    
    def log(self, message, title="Info"):
        """Log a message with a title."""
        print(f"{title}: {message}")
    
    def generate_sql(self, question: str, allow_introspection=False, **kwargs) -> str:
        """
        Generate SQL for a given question using the LLM and vector context.
        
        Args:
            question: The natural language question to convert to SQL
            allow_introspection: Whether to allow querying the database for metadata
            
        Returns:
            Generated SQL query
        """
        initial_prompt = self.config.get("initial_prompt", None)
        
        # Get context from vector store
        similar_questions = self.get_similar_questions(question)
        schema_info = self.get_related_schema(question)
        documentation = self.get_related_documentation(question)
        
        # Build prompt with context
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            similar_questions=similar_questions,
            schema_info=schema_info,
            documentation=documentation
        )
        
        self.log(prompt, "SQL Prompt")
        
        # Submit to LLM
        llm_response = self.submit_prompt(prompt)
        self.log(llm_response, "LLM Response")
        
        # Check if introspection is needed and allowed
        if 'intermediate_sql' in llm_response:
            if not allow_introspection:
                return "This question requires database introspection. Set allow_introspection=True to enable this."
            
            # Extract and run the intermediate query for introspection
            intermediate_sql = self.extract_sql(llm_response)
            try:
                self.log(intermediate_sql, "Running Intermediate SQL")
                df = self.run_sql(intermediate_sql)
                
                # Create new prompt with introspection results
                enhanced_prompt = self.get_sql_prompt(
                    initial_prompt=initial_prompt,
                    question=question,
                    similar_questions=similar_questions,
                    schema_info=schema_info,
                    documentation=documentation + [
                        f"Intermediate query results: \n{df.to_markdown()}"
                    ]
                )
                
                self.log(enhanced_prompt, "Enhanced SQL Prompt")
                llm_response = self.submit_prompt(enhanced_prompt)
                self.log(llm_response, "Enhanced LLM Response")
            except Exception as e:
                return f"Error running intermediate SQL: {e}"
        
        # Extract SQL from LLM response
        return self.extract_sql(llm_response)
    
    def generate_sql_with_error_context(self, question: str, previous_sql: str, error_message: str, **kwargs) -> str:
        """
        Generate an improved SQL for a given question using the error message from a failed execution.
        
        Args:
            question: The natural language question
            previous_sql: The previously generated SQL that failed
            error_message: The error message from the failed execution
            
        Returns:
            Corrected SQL query
        """
        initial_prompt = self.config.get("initial_prompt", None)
        
        # Get context from vector store
        similar_questions = self.get_similar_questions(question)
        schema_info = self.get_related_schema(question)
        documentation = self.get_related_documentation(question)
        
        # Build prompt with context and error information
        prompt = self.get_sql_correction_prompt(
            initial_prompt=initial_prompt,
            question=question,
            previous_sql=previous_sql,
            error_message=error_message,
            similar_questions=similar_questions,
            schema_info=schema_info,
            documentation=documentation
        )
        
        self.log(prompt, "SQL Correction Prompt")
        
        # Submit to LLM
        llm_response = self.submit_prompt(prompt)
        self.log(llm_response, "Corrected SQL Response")
        
        # Extract SQL from LLM response
        return self.extract_sql(llm_response)
    
    def get_sql_correction_prompt(self, 
                                 initial_prompt: str,
                                 question: str,
                                 previous_sql: str,
                                 error_message: str,
                                 similar_questions: list,
                                 schema_info: list,
                                 documentation: list) -> list:
        """
        Create a prompt for SQL correction with error context.
        
        Args:
            initial_prompt: Base prompt instructions
            question: The user's question
            previous_sql: Previously generated SQL that failed
            error_message: Error message from failed execution
            similar_questions: List of similar Q&A pairs
            schema_info: Database schema information
            documentation: Additional context
            
        Returns:
            List of messages for the LLM
        """
        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
                "Correct a SQL query that generated an error. Base your response only on the provided context."
        
        # Add schema information
        if schema_info:
            initial_prompt += "\n\n=== Database Schema ===\n"
            for schema in schema_info:
                initial_prompt += f"{schema}\n\n"
        
        # Add documentation context
        if documentation:
            initial_prompt += "\n\n=== Additional Context ===\n"
            for doc in documentation:
                initial_prompt += f"{doc}\n\n"
        
        initial_prompt += "\n=== Error Information ===\n"
        initial_prompt += f"Question: {question}\n"
        initial_prompt += f"Previous SQL Query: {previous_sql}\n"
        initial_prompt += f"Error Message: {error_message}\n\n"
        
        initial_prompt += (
            "\n=== Guidelines ===\n"
            "1. Analyze the error message carefully.\n"
            "2. Identify and fix the issues in the previous SQL query.\n"
            "3. Ensure the corrected query is valid and addresses the original question.\n"
            "4. Always wrap your SQL query in <sql> tags like this:\n"
            "<sql>\n"
            "SELECT column FROM table WHERE condition;\n"
            "</sql>\n"
            "5. Return only the corrected SQL query without any explanations.\n"
            f"6. Ensure the output is valid {self.dialect}.\n"
        )
        
        # Create message sequence
        messages = [self.system_message(initial_prompt)]
        
        # Add examples of similar questions
        for example in similar_questions:
            if "question" in example and "sql" in example:
                # Format the SQL response with XML tags if not already formatted
                from talk2sql.utils import format_sql_with_xml_tags
                formatted_sql = format_sql_with_xml_tags(example["sql"])
                
                messages.append(self.user_message(example["question"]))
                messages.append(self.assistant_message(formatted_sql))
        
        # Add the current error context
        messages.append(self.user_message(
            f"Please fix this SQL query for the question: '{question}'\n\n" +
            f"Failed Query: {previous_sql}\n\n" +
            f"Error: {error_message}"
        ))
        
        return messages
    
    def get_sql_prompt(self, 
                          initial_prompt: str,
                          question: str,
                          similar_questions: list,
                          schema_info: list,
                          documentation: list) -> list:
        """
        Create a prompt for SQL generation.
        
        Args:
            initial_prompt: Base prompt instructions
            question: The user's question
            similar_questions: List of similar Q&A pairs
            schema_info: Database schema information
            documentation: Additional context
            
        Returns:
            List of messages for the LLM
        """
        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. Generate SQL for the given question using the provided context."
        
        # Add schema information
        if schema_info:
            initial_prompt += "\n\n=== Database Schema ===\n"
            for schema in schema_info:
                initial_prompt += f"{schema}\n\n"
        
        # Add documentation context
        if documentation:
            initial_prompt += "\n\n=== Additional Context ===\n"
            for doc in documentation:
                initial_prompt += f"{doc}\n\n"
        
        initial_prompt += (
            "\n=== Guidelines ===\n"
            "1. Analyze the question carefully to understand what data is being requested.\n"
            "2. Use only the tables and columns mentioned in the schema.\n"
            "3. Write efficient, readable SQL that answers the question.\n"
            f"4. Ensure the output is valid {self.dialect}.\n"
            "5. Always wrap your SQL query in <sql> tags like this:\n"
            "<sql>\n"
            "SELECT column FROM table WHERE condition;\n"
            "</sql>\n"
            "6. Only include the final SQL query without explanations.\n"
        )
        
        # Create message sequence
        messages = [self.system_message(initial_prompt)]
        
        # Add examples of similar questions
        for example in similar_questions:
            if "question" in example and "sql" in example:
                # Format the SQL response with XML tags if not already formatted
                from talk2sql.utils import format_sql_with_xml_tags
                formatted_sql = format_sql_with_xml_tags(example["sql"])
                
                messages.append(self.user_message(example["question"]))
                messages.append(self.assistant_message(formatted_sql))
        
        # Add the current question
        messages.append(self.user_message(question))
        
        return messages
    
    def extract_sql(self, llm_response: str) -> str:
        """Extract SQL query from LLM response text."""
        # Match SQL in XML tags
        sqls = re.findall(r"<sql>(.*?)</sql>", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1].strip()
            
        # Match CREATE TABLE ... AS SELECT
        sqls = re.findall(r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", 
                         llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]
            
        # Match WITH clause (CTEs)
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]
            
        # Match SELECT statements
        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1]
            
        # Match SQL code blocks
        sqls = re.findall(r"```sql\s*\n(.*?)```", 
                         llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1].strip()
            
        # Match any code blocks
        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            return sqls[-1].strip()
            
        # If we couldn't find SQL, check if the response is already SQL-like
        if re.search(r"\bSELECT\b", llm_response, re.IGNORECASE):
            self.log("Returning raw response as SQL", "Warning")
            return llm_response
            
        # If all else fails
        raise SQLParsingError("Could not extract SQL query from LLM response", llm_response)
    
    def is_sql_valid(self, sql: str) -> bool:
        """Check if SQL query is valid (default: must be SELECT)."""
        parsed = sqlparse.parse(sql)
        return any(statement.get_type() == 'SELECT' for statement in parsed)
    
    def should_generate_visualization(self, df: pd.DataFrame) -> bool:
        """Determine if results should be visualized."""
        return len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0
    
    def execute_sql_with_retry(self, sql: str, question: str, retry_count=0) -> Tuple[pd.DataFrame, str, int]:
        """
        Execute SQL with retry mechanism for failed queries.
        
        Args:
            sql: SQL query to execute
            question: Original natural language question
            retry_count: Current retry attempt number
            
        Returns:
            Tuple of (DataFrame with results, final SQL used, number of retries performed)
        """
        if retry_count >= self.max_retry_attempts:
            raise Exception(f"Maximum retry attempts ({self.max_retry_attempts}) exceeded. Could not execute query successfully.")
        
        try:
            # Attempt to execute the SQL
            df = self.run_sql(sql)
            return df, sql, retry_count
            
        except Exception as e:
            # Capture the error message and traceback
            error_message = str(e)
            error_traceback = traceback.format_exc()
            self.log(f"SQL execution failed (attempt {retry_count + 1}): {error_message}", "Error")
            self.log(error_traceback, "Traceback")
            
            # Increment retry count
            new_retry_count = retry_count + 1
            
            if new_retry_count < self.max_retry_attempts:
                self.log(f"Attempting correction (retry {new_retry_count} of {self.max_retry_attempts})", "Retry")
                
                # Generate corrected SQL using error information
                corrected_sql = self.generate_sql_with_error_context(
                    question=question,
                    previous_sql=sql,
                    error_message=error_message
                )
                
                self.log(f"Corrected SQL: {corrected_sql}", "Retry")
                
                # Recursively try the corrected SQL
                return self.execute_sql_with_retry(
                    sql=corrected_sql,
                    question=question,
                    retry_count=new_retry_count
                )
            else:
                # Re-raise the exception if max retries exceeded
                raise Exception(f"Failed to execute SQL after {self.max_retry_attempts} attempts. Last error: {error_message}")
    
    def ask(self, 
            question: str = None, 
            print_results: bool = True, 
            auto_train: bool = True,
            visualize: bool = True,
            allow_introspection: bool = False,
            streaming: bool = None,
            stream_handler: Callable = None) -> Union[Tuple[str, pd.DataFrame, go.Figure], str]:
        """
        Process a natural language question to SQL, execute it, and optionally visualize the results.
        
        Args:
            question: Natural language question to convert to SQL
            print_results: Whether to print results
            auto_train: Whether to automatically add successful question-SQL pairs to vector store
            visualize: Whether to automatically generate visualizations
            allow_introspection: Whether to allow querying the database for metadata
            streaming: Whether to stream results (overrides instance-level setting)
            stream_handler: Callback function for streaming events
            
        Returns:
            If streaming=False: Tuple of (SQL, DataFrame, Plotly figure) or just SQL
            If streaming=True: Task ID for streaming process
        """
        if not question:
            return "Please provide a question."
        
        # Determine if streaming is enabled
        use_streaming = self._enable_streaming if streaming is None else streaming
        
        # Use streaming pipeline if enabled
        if use_streaming:
            from talk2sql.streaming import StreamingPipeline, EventType, StreamingEvent
            
            # Initialize streaming pipeline if not already initialized
            if not self._streaming_pipeline:
                self._streaming_pipeline = StreamingPipeline(self)
            
            # Register custom stream handler if provided
            if stream_handler:
                self._streaming_pipeline.executor.register_event_handler("*", stream_handler)
            
            # Process the question and return task ID
            return self._streaming_pipeline.process_question(question, allow_introspection)
        
        # Regular non-streaming execution
        try:
            # Generate SQL
            if print_results:
                print(f"Question: {question}")
                
            sql = self.generate_sql(question, allow_introspection=allow_introspection)
            
            if print_results:
                print(f"Generated SQL: {sql}")
                
            # Check if database connection is set
            if not self.run_sql_is_set:
                if print_results:
                    print("No database connection. Set run_sql to execute queries.")
                return sql
                
            # Execute SQL with retry
            df, error, retry_count = self.execute_sql_with_retry(sql, question)
            
            if error:
                if print_results:
                    print(f"Error executing SQL: {error}")
                return sql
                
            # Print the results
            if print_results:
                if len(df) > 0:
                    try:
                        from IPython.display import display
                        display(df)
                    except ImportError:
                        print(df)
                else:
                    print("Query returned no results.")
                    
            # Add to training data if successful
            if auto_train:
                try:
                    self.add_question_sql(question, sql)
                    if print_results:
                        print("Added to training data.")
                except Exception as e:
                    if print_results:
                        print(f"Failed to add to training data: {e}")
                        
            # Generate visualization
            fig = None
            if visualize and self.should_generate_visualization(df):
                if print_results:
                    print("Generating visualization...")
                    
                try:
                    # Create dataframe metadata for visualization
                    df_metadata = {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.astype(str).to_dict()
                    }
                    
                    # Generate visualization code
                    plotly_code = self.generate_plotly_code(question, sql, df_metadata)
                    
                    # Create a local namespace for execution
                    local_namespace = {"df": df, "go": go, "pd": pd}
                    
                    # Execute the plotly code
                    exec(plotly_code, globals(), local_namespace)
                    
                    # Extract the figure from the local namespace
                    fig = local_namespace.get("fig")
                    
                    if fig and print_results:
                        try:
                            from IPython.display import display
                            display(fig)
                        except ImportError:
                            print("Visualization generated (but cannot display outside of notebook environment).")
                            
                except Exception as e:
                    if print_results:
                        print(f"Error generating visualization: {e}")
                        
            return sql, df, fig
        except Exception as e:
            if print_results:
                print(f"Error: {e}")
                traceback.print_exc()
            return str(e)
            
    def ask_with_streaming(self, question: str, allow_introspection: bool = False) -> Iterator[Dict[str, Any]]:
        """
        Process a question and yield streaming events for each stage of the process.
        
        Args:
            question: Natural language question
            allow_introspection: Whether to allow database introspection
            
        Yields:
            Event dictionaries for each stage of the process
        """
        # Import streaming components
        from talk2sql.streaming import StreamingPipeline, EventType, StreamingEvent
        
        # Initialize streaming pipeline if not already initialized
        if not self._streaming_pipeline:
            self._streaming_pipeline = StreamingPipeline(self)
        
        # Process the question
        task_id = self._streaming_pipeline.process_question(question, allow_introspection)
        
        # Yield events as they occur
        for event in self._streaming_pipeline.stream_events():
            event_dict = event.to_dict()
            # Only yield events for this task
            if event_dict.get("metadata", {}).get("task_id") == task_id:
                yield event_dict
                
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def system_message(self, message: str) -> dict:
        """Create a system message for LLM."""
        pass
    
    @abstractmethod
    def user_message(self, message: str) -> dict:
        """Create a user message for LLM."""
        pass
    
    @abstractmethod
    def assistant_message(self, message: str) -> dict:
        """Create an assistant message for LLM."""
        pass
    
    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit a prompt to the LLM and get response."""
        pass
    
    @abstractmethod
    def generate_plotly_code(self, question, sql, df_metadata) -> str:
        """Generate Plotly visualization code."""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        pass
    
    @abstractmethod
    def get_similar_questions(self, question: str) -> list:
        """Get similar questions with their SQL from vector store."""
        pass
    
    @abstractmethod
    def get_related_schema(self, question: str) -> list:
        """Get related schema information from vector store."""
        pass
    
    @abstractmethod
    def get_related_documentation(self, question: str) -> list:
        """Get related documentation from vector store."""
        pass
    
    @abstractmethod
    def add_question_sql(self, question: str, sql: str) -> str:
        """Add question-SQL pair to vector store."""
        pass
    
    @abstractmethod
    def add_schema(self, schema: str) -> str:
        """Add database schema to vector store."""
        pass
    
    @abstractmethod
    def add_documentation(self, documentation: str) -> str:
        """Add documentation to vector store."""
        pass