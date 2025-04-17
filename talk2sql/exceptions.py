"""
Exceptions for the Talk2SQL package.
"""

class Talk2SQLException(Exception):
    """Base exception for all Talk2SQL errors."""
    pass

class SQLGenerationError(Talk2SQLException):
    """Exception raised when SQL generation fails."""
    pass

class SQLExecutionError(Talk2SQLException):
    """Exception raised when SQL execution fails."""
    pass

class SQLParsingError(Talk2SQLException):
    """Exception raised when SQL parsing fails."""
    def __init__(self, message, response_text=None):
        super().__init__(message)
        self.response_text = response_text

class SQLValidationError(Talk2SQLException):
    """Exception raised when SQL validation fails."""
    pass

class VectorStoreError(Talk2SQLException):
    """Exception raised for vector store operations."""
    pass

class LLMError(Talk2SQLException):
    """Exception raised for LLM-related errors."""
    pass

class ConfigError(Talk2SQLException):
    """Exception raised for configuration errors."""
    pass

class XMLTagError(Talk2SQLException):
    """Exception raised when XML tag parsing fails."""
    def __init__(self, message, text=None):
        super().__init__(message)
        self.text = text
