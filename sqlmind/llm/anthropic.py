import os
import re
from typing import List, Dict, Any

from openai import OpenAI

from sqlmind.base import SQLMindBase

class AnthropicLLM(SQLMindBase):
    """LLM implementation using Anthropic's Claude models via OpenAI SDK compatibility layer."""
    
    def __init__(self, config=None):
        """
        Initialize Anthropic LLM using OpenAI SDK compatibility.
        
        Args:
            config: Configuration dictionary with options:
                - anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
                - claude_model: Claude model name (default: "claude-3-5-haiku-20241022")
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens in response (default: 4000)
        """
        super().__init__(config)
        
        # Get Anthropic API credentials
        self.api_key = config.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        
        # Get model parameters
        self.model = config.get("claude_model", "claude-3-5-haiku-20241022")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4000)
        
        # Validate required parameters
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Provide in config or set ANTHROPIC_API_KEY env var.")
        
        # Initialize OpenAI client with Anthropic compatibility settings
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.anthropic.com/v1/"
        )
    
    def system_message(self, message: str) -> Dict[str, str]:
        """Create a system message for Anthropic."""
        return {"role": "system", "content": message}
    
    def user_message(self, message: str) -> Dict[str, str]:
        """Create a user message for Anthropic."""
        return {"role": "user", "content": message}
    
    def assistant_message(self, message: str) -> Dict[str, str]:
        """Create an assistant message for Anthropic."""
        return {"role": "assistant", "content": message}
    
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Submit a prompt to Anthropic and get a response.
        
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
        
        # Enable extended thinking if specified
        extra_body = {}
        if kwargs.get("enable_thinking", False):
            extra_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": kwargs.get("thinking_budget", 2000)
            }
        
        try:
            # Send to Anthropic via OpenAI compatibility layer
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None
            )
            
            # Return the response content
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error submitting prompt to Anthropic: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        This is a placeholder - in SQLMindAnthropic we'll use Azure for embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        raise NotImplementedError("Embedding generation should be implemented in a subclass")
    
    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code from markdown blocks
        code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(code_pattern, response)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the response as is
        return response.strip()
        
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
                "Return only the Python code without explanations or markdown formatting. "
                "The final plot should be assigned to a variable named 'fig'.\n\n"
                "IMPORTANT: Use ONLY the following brand colors in your visualization or a color palette that is similar to the brand colors:\n"
                "- Primary colors: '#CC785C' (Book Cloth), '#D4A27F' (Kraft), '#EBDBBC' (Manilla)\n"
                "- Secondary colors: '#191919' (Slate Dark), '#262625' (Slate Medium), '#40403E' (Slate Light)\n"
                "- Background colors: '#FFFFFF' (White), '#F0F0EB' (Ivory Medium), '#FAFAF7' (Ivory Light)\n"
                "- Set paper_bgcolor to '#FFFFFF' and plot_bgcolor to '#F0F0EB'\n"
                "- Use '#191919' for text color\n"
                "- Create a clean, minimal design with appropriate spacing\n"
                "- Font family should be 'Styrene A, sans-serif'"
            )
        ]
        
        # Get response from Anthropic
        response = self.submit_prompt(prompt)
        
        # Extract code
        return self._extract_python_code(response)
        
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