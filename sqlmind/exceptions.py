"""
Exceptions for the SQLMind package.
"""

class SQLMindException(Exception):
    """Base exception for all SQLMind errors."""
    pass

class SQLGenerationError(SQLMindException):
    """Exception raised when SQL generation fails."""
    pass

class SQLExecutionError(SQLMindException):
    """Exception raised when SQL execution fails."""
    pass

class SQLParsingError(SQLMindException):
    """Exception raised when SQL parsing fails."""
    def __init__(self, message, response_text=None):
        super().__init__(message)
        self.response_text = response_text

class SQLValidationError(SQLMindException):
    """Exception raised when SQL validation fails."""
    pass

class VectorStoreError(SQLMindException):
    """Exception raised for vector store operations."""
    pass

class LLMError(SQLMindException):
    """Exception raised for LLM-related errors."""
    pass

class ConfigError(SQLMindException):
    """Exception raised for configuration errors."""
    pass

class XMLTagError(SQLMindException):
    """Exception raised when XML tag parsing fails."""
    def __init__(self, message, text=None):
        super().__init__(message)
        self.text = text
