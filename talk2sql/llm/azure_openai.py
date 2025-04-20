import os
import re
from typing import List, Dict, Any

import openai
from openai import AzureOpenAI

from talk2sql.base import Talk2SQLBase

class AzureOpenAILLM(Talk2SQLBase):
    """LLM implementation using Azure OpenAI's GPT models and text-embedding models."""
    
    def __init__(self, config=None):
        """
        Initialize Azure OpenAI LLM.
        
        Args:
            config: Configuration dictionary with options:
                - azure_api_key: Azure OpenAI API key (or use AZURE_OPENAI_API_KEY env var)
                - azure_endpoint: Azure OpenAI endpoint (or use AZURE_ENDPOINT env var)
                - azure_api_version: Azure API version (or use AZURE_API_VERSION env var)
                - azure_deployment: GPT deployment name (or use AZURE_DEPLOYMENT env var)
                - azure_embedding_deployment: Embedding deployment name (default: "text-embedding-ada-002")
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens in response (default: 4000)
        """
        super().__init__(config)
        
        # Get Azure OpenAI API credentials
        self.api_key = config.get("azure_api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        self.endpoint = config.get("azure_endpoint", os.getenv("AZURE_ENDPOINT"))
        self.api_version = config.get("azure_api_version", os.getenv("AZURE_API_VERSION", "2024-02-15-preview"))
        
        # Get deployment names
        self.deployment = config.get("azure_deployment", os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini"))
        self.embedding_deployment = config.get("azure_embedding_deployment", "text-embedding-ada-002")
        
        # Get model parameters
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4000)
        
        # Validate required parameters
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required. Provide in config or set AZURE_OPENAI_API_KEY env var.")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Provide in config or set AZURE_ENDPOINT env var.")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )
    
    def system_message(self, message: str) -> Dict[str, str]:
        """Create a system message for Azure OpenAI."""
        return {"role": "system", "content": message}
    
    def user_message(self, message: str) -> Dict[str, str]:
        """Create a user message for Azure OpenAI."""
        return {"role": "user", "content": message}
    
    def assistant_message(self, message: str) -> Dict[str, str]:
        """Create an assistant message for Azure OpenAI."""
        return {"role": "assistant", "content": message}
    
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Submit a prompt to Azure OpenAI and get a response.
        
        Args:
            prompt: List of message dictionaries
            
        Returns:
            Response text
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Set parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            # Send to Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Return the response content
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error submitting prompt to Azure OpenAI: {e}")
            return f"Error generating response: {str(e)}"
    
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
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
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
    
    def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None) -> str:
        """
        Generate Plotly visualization code.
        
        Args:
            question: Natural language question
            sql: SQL query
            df_metadata: DataFrame metadata information
            
        Returns:
            Python code for Plotly visualization
        """
        # Create system message
        system_msg = "You are an expert data visualization developer."
        
        if question:
            system_msg += f" The user asked: '{question}'"
            
        if sql:
            system_msg += f"\n\nThe SQL query used was: {sql}"
            
        if df_metadata:
            system_msg += f"\n\nDataFrame information: {df_metadata}"
            
        # Create prompt
        prompt = [
            self.system_message(system_msg),
            self.user_message(
                "Generate Python code using Plotly to visualize this data. "
                "Use 'df' as the DataFrame variable. "
                "Return only the Python code without explanations or markdown formatting. Do not end with fig.show()"
                "The final plot should be assigned to a variable named 'fig'.\n\n"
                "IMPORTANT: Use ONLY the following brand colors in your visualization:\n"
                "- Primary colors: '#CC785C' (Book Cloth), '#D4A27F' (Kraft), '#EBDBBC' (Manilla)\n"
                "- Secondary colors: '#191919' (Slate Dark), '#262625' (Slate Medium), '#40403E' (Slate Light)\n"
                "- Background colors: '#FFFFFF' (White), '#F0F0EB' (Ivory Medium), '#FAFAF7' (Ivory Light)\n"
                "- Set paper_bgcolor to '#FFFFFF' and plot_bgcolor to '#F0F0EB'\n"
                "- Use '#191919' for text color\n"
                "- Create a clean, minimal design with appropriate spacing\n"
                "- Font family should be 'Styrene A, sans-serif'"
            )
        ]
        
        # Get response from Azure OpenAI
        response = self.submit_prompt(prompt)
        
        # Extract code
        return self._extract_python_code(response)
    
    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code from markdown blocks
        code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(code_pattern, response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the response as is
        return response.strip()
        
    def explain_sql(self, sql: str) -> str:
        """
        Generate an explanation of SQL query.
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Natural language explanation
        """
        prompt = [
            self.system_message("You are an expert SQL educator. Explain SQL queries in clear, simple terms."),
            self.user_message(f"Please explain this SQL query in simple terms:\n\n{sql}")
        ]
        
        return self.submit_prompt(prompt)
    
    def generate_follow_up_questions(self, question: str, sql: str, result_info: str, n=3) -> List[str]:
        """
        Generate follow-up questions based on previous query.
        
        Args:
            question: Original question
            sql: SQL query used
            result_info: Information about query results
            n: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        prompt = [
            self.system_message(
                f"You are a data analyst helping with SQL queries. "
                f"The user asked: '{question}'\n\n"
                f"The SQL query used was: {sql}\n\n"
                f"Results information: {result_info}"
            ),
            self.user_message(
                f"Generate {n} natural follow-up questions that would be logical next steps for analysis. "
                f"Each question should be answerable with SQL. "
                f"Return only the questions, one per line."
            )
        ]
        
        response = self.submit_prompt(prompt)
        
        # Split into list of questions
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
        
        # Limit to n questions
        return questions[:n]
        
    def generate_starter_questions(self, schema: str, n=5) -> List[str]:
        """
        Generate starter questions based on database schema.
        
        Args:
            schema: Database schema information
            n: Number of starter questions to generate
            
        Returns:
            List of starter questions
        """
        prompt = [
            self.system_message(
                f"You are a data analyst helping users explore a database. "
                f"The database has the following schema:\n\n{schema}"
            ),
            self.user_message(
                f"Generate {n} natural starter questions that would help a user begin exploring this database. "
                f"Focus on common data exploration queries that would provide insights about the data. "
                f"Each question should be answerable with SQL. "
                f"Make questions specific to the tables and columns in the schema. "
                f"Return only the questions, one per line."
            )
        ]
        
        response = self.submit_prompt(prompt)
        
        # Split into list of questions
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
        
        # Limit to n questions
        return questions[:n]