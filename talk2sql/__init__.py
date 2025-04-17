"""
Talk2SQL - SQL generation from natural language using Azure OpenAI
"""

from talk2sql.engine import Talk2SQLAzure
from talk2sql.streaming import StreamingPipeline, ThreadedExecutor, StreamingEvent, EventType, create_flask_streaming_endpoint

__version__ = "0.1.0"
__all__ = [
    "Talk2SQLAzure",
    "StreamingPipeline",
    "ThreadedExecutor",
    "StreamingEvent",
    "EventType",
    "create_flask_streaming_endpoint"
]
