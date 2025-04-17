"""
SQLMind - SQL generation from natural language using Azure OpenAI
"""

from sqlmind.engine import SQLMindAzure
from sqlmind.streaming import StreamingPipeline, ThreadedExecutor, StreamingEvent, EventType, create_flask_streaming_endpoint

__version__ = "0.1.0"
__all__ = [
    "SQLMindAzure",
    "StreamingPipeline",
    "ThreadedExecutor",
    "StreamingEvent",
    "EventType",
    "create_flask_streaming_endpoint"
]
