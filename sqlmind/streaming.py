"""
Streaming and multithreading support for SQLMind.
Provides Server-Sent Events (SSE) and concurrent execution capabilities.
"""

import threading
import queue
import time
import json
from typing import Dict, List, Any, Optional, Callable, Iterator, Union
import pandas as pd
import plotly.graph_objects as go

# Event types for streaming
class EventType:
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    DATAFRAME_READY = "dataframe_ready"
    VISUALIZATION_READY = "visualization_ready"
    LLM_SUMMARY = "llm_summary"
    ERROR = "error"


class StreamingEvent:
    """Represents a streaming event in the SQLMind pipeline."""
    
    def __init__(self, event_type: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a streaming event.
        
        Args:
            event_type: Type of event (from EventType)
            data: Event data payload
            metadata: Additional metadata about the event
        """
        self.event_type = event_type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event": self.event_type,
            "data": self._prepare_data_for_serialization(self.data),
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_sse(self) -> str:
        """Format event as Server-Sent Event string."""
        event_data = json.dumps(self.to_dict())
        return f"event: {self.event_type}\ndata: {event_data}\n\n"
    
    def _prepare_data_for_serialization(self, data: Any) -> Any:
        """Convert complex data types to serializable formats."""
        if isinstance(data, pd.DataFrame):
            return {
                "type": "dataframe",
                "data": data.to_dict(orient="records"),
                "columns": data.columns.tolist(),
                "index": data.index.tolist()
            }
        elif isinstance(data, go.Figure):
            return {
                "type": "plotly_figure",
                "data": data.to_json()
            }
        elif isinstance(data, Exception):
            return {
                "type": "error",
                "message": str(data),
                "traceback": getattr(data, "__traceback__", None) and "".join(
                    traceback.format_tb(data.__traceback__)
                )
            }
        return data


class ThreadedExecutor:
    """Handles multithreaded execution of SQLMind pipeline steps."""
    
    def __init__(self, max_workers: int = 3):
        """
        Initialize ThreadedExecutor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.workers = []
        self.tasks_queue = queue.Queue()
        self.results = {}
        self.event_handlers = {}
        self._running = False
    
    def start(self):
        """Start the executor."""
        if self._running:
            return
            
        self._running = True
        
        # Start worker threads
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop the executor."""
        self._running = False
        
        # Put None tasks to signal workers to stop
        for _ in range(self.max_workers):
            self.tasks_queue.put(None)
            
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
    
    def _worker_loop(self):
        """Worker thread main loop."""
        while self._running:
            task = self.tasks_queue.get()
            
            if task is None:  # Sentinel to stop worker
                self.tasks_queue.task_done()
                break
                
            try:
                task_id, func, args, kwargs, callback = task
                result = func(*args, **kwargs)
                
                if callback:
                    callback(result)
                    
                self.results[task_id] = {
                    "status": "completed",
                    "result": result
                }
                
            except Exception as e:
                self.results[task_id] = {
                    "status": "error",
                    "error": str(e)
                }
                
                # Notify error event handlers
                self._notify_event_handlers(
                    StreamingEvent(EventType.ERROR, e, {"task_id": task_id})
                )
                
            finally:
                self.tasks_queue.task_done()
    
    def submit(self, func: Callable, *args, callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            callback: Optional callback for when task completes
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task ID
        """
        task_id = f"task_{len(self.results)}"
        self.results[task_id] = {"status": "pending"}
        self.tasks_queue.put((task_id, func, args, kwargs, callback))
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task.
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait for result
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If timeout is reached before result is available
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if task_id in self.results and self.results[task_id]["status"] != "pending":
                result = self.results[task_id]
                
                if result["status"] == "error":
                    raise RuntimeError(result["error"])
                    
                return result["result"]
                
            time.sleep(0.1)
            
        raise TimeoutError(f"Task {task_id} did not complete within timeout")
    
    def register_event_handler(self, event_type: str, handler: Callable[[StreamingEvent], None]):
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Event type to handle
            handler: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler)
    
    def _notify_event_handlers(self, event: StreamingEvent):
        """
        Notify all registered handlers for an event.
        
        Args:
            event: Event to notify handlers about
        """
        # Notify handlers for this specific event type
        for handler in self.event_handlers.get(event.event_type, []):
            handler(event)
            
        # Notify handlers registered for all events
        for handler in self.event_handlers.get("*", []):
            handler(event)


class StreamingPipeline:
    """Manages the streaming pipeline for SQLMind operations."""
    
    def __init__(self, sqlmind_instance):
        """
        Initialize StreamingPipeline.
        
        Args:
            sqlmind_instance: SQLMind instance to use for operations
        """
        self.sqlmind = sqlmind_instance
        self.executor = ThreadedExecutor()
        self.event_queue = queue.Queue()
        self.executor.start()
        
        # Register internal event handler
        self.executor.register_event_handler("*", self._queue_event)
    
    def _queue_event(self, event: StreamingEvent):
        """Add an event to the queue."""
        self.event_queue.put(event)
    
    def stream_events(self) -> Iterator[StreamingEvent]:
        """
        Stream events as they occur.
        
        Returns:
            Iterator of events
        """
        while True:
            try:
                event = self.event_queue.get(timeout=0.1)
                yield event
                self.event_queue.task_done()
            except queue.Empty:
                continue
    
    def stream_sse(self) -> Iterator[str]:
        """
        Stream events as Server-Sent Events.
        
        Returns:
            Iterator of SSE-formatted event strings
        """
        for event in self.stream_events():
            yield event.to_sse()
    
    def process_question(self, question: str, allow_introspection: bool = False) -> str:
        """
        Process a question through the full pipeline with streaming events.
        
        Args:
            question: Natural language question
            allow_introspection: Whether to allow database introspection
            
        Returns:
            Task ID for the pipeline execution
        """
        # Create task ID
        task_id = f"question_{int(time.time())}"
        
        # Start the pipeline as a background task
        self.executor.submit(
            self._process_question_pipeline,
            question,
            allow_introspection,
            task_id
        )
        
        return task_id
    
    def _process_question_pipeline(self, question: str, allow_introspection: bool, task_id: str):
        """
        Internal implementation of the question processing pipeline.
        
        Args:
            question: Natural language question
            allow_introspection: Whether to allow database introspection
            task_id: Task ID for the pipeline
        """
        try:
            # Step 1: Generate SQL
            self._notify_event(EventType.SQL_GENERATION, "Generating SQL query...", 
                              {"task_id": task_id, "status": "started", "question": question})
            
            sql = self.sqlmind.generate_sql(question, allow_introspection=allow_introspection)
            
            self._notify_event(EventType.SQL_GENERATION, sql, 
                              {"task_id": task_id, "status": "completed"})
            
            # Step 2: Execute SQL
            self._notify_event(EventType.SQL_EXECUTION, "Executing SQL query...", 
                              {"task_id": task_id, "status": "started", "sql": sql})
            
            df, error, retry_count = self.sqlmind.execute_sql_with_retry(sql, question)
            
            if error:
                self._notify_event(EventType.ERROR, error, 
                                  {"task_id": task_id, "stage": "sql_execution", "retry_count": retry_count})
                return
                
            self._notify_event(EventType.SQL_EXECUTION, "SQL query executed successfully", 
                              {"task_id": task_id, "status": "completed", "retry_count": retry_count})
            
            # Step 3: Dataframe Ready
            self._notify_event(EventType.DATAFRAME_READY, df, {"task_id": task_id, "rows": len(df)})
            
            # Step 4: Generate visualization if appropriate
            if self.sqlmind.should_generate_visualization(df):
                self._notify_event(EventType.VISUALIZATION_READY, "Generating visualization...", 
                                  {"task_id": task_id, "status": "started"})
                
                # Generate metadata for visualization
                df_metadata = {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict()
                }
                
                # Generate visualization
                plotly_code = self.sqlmind.generate_plotly_code(question, sql, df_metadata)
                
                # Create a local namespace for execution
                local_namespace = {"df": df, "go": go, "pd": pd}
                
                # Execute the plotly code
                exec(plotly_code, globals(), local_namespace)
                
                # Extract the figure from the local namespace
                fig = local_namespace.get("fig")
                
                if fig:
                    self._notify_event(EventType.VISUALIZATION_READY, fig, 
                                      {"task_id": task_id, "status": "completed"})
            
            # Step 5: Generate LLM summary of the dataframe
            self._notify_event(EventType.LLM_SUMMARY, "Generating insights from data...", 
                              {"task_id": task_id, "status": "started"})
            
            # Convert dataframe to markdown format for the LLM
            df_str = df.to_markdown() if len(df) <= 50 else df.head(50).to_markdown() + f"\n\n... and {len(df) - 50} more rows"
            
            # Generate the summary prompt
            summary_prompt = [
                self.sqlmind.system_message(
                    "You are a data analyst assistant. Provide a detailed analysis of the following data. "
                    "Highlight key insights, patterns, and noteworthy information. Be concise and factual."
                ),
                self.sqlmind.user_message(
                    f"Question: {question}\n\n"
                    f"SQL Query: {sql}\n\n"
                    f"Result Data:\n{df_str}\n\n"
                    "Please provide a detailed analysis of these results. Include key insights, trends, and important findings. "
                    "Be specific about what the data shows in relation to the original question."
                )
            ]
            
            # Get the summary from the LLM
            summary = self.sqlmind.submit_prompt(summary_prompt)
            
            self._notify_event(EventType.LLM_SUMMARY, summary, 
                              {"task_id": task_id, "status": "completed"})
            
        except Exception as e:
            import traceback
            self._notify_event(EventType.ERROR, str(e), {
                "task_id": task_id, 
                "traceback": traceback.format_exc()
            })
    
    def _notify_event(self, event_type: str, data: Any, metadata: Dict[str, Any] = None):
        """
        Create and queue a new event.
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata
        """
        event = StreamingEvent(event_type, data, metadata)
        self._queue_event(event)


def create_flask_streaming_endpoint(app, streaming_pipeline):
    """
    Create a Flask endpoint for SSE streaming.
    
    Args:
        app: Flask application
        streaming_pipeline: StreamingPipeline instance
    """
    from flask import Response, request
    
    @app.route('/api/stream/query', methods=['POST'])
    def stream_query():
        data = request.json
        question = data.get('question')
        allow_introspection = data.get('allow_introspection', False)
        
        if not question:
            return {"error": "Question is required"}, 400
        
        # Process the question and get task ID
        task_id = streaming_pipeline.process_question(question, allow_introspection)
        
        # Return the task ID that can be used to stream results
        return {"task_id": task_id}
    
    @app.route('/api/stream/events/<task_id>', methods=['GET'])
    def stream_events(task_id):
        def generate():
            yield "retry: 1000\n\n"  # Reconnection timeout in ms
            
            for sse_event in streaming_pipeline.stream_sse():
                yield sse_event
        
        return Response(generate(), content_type='text/event-stream') 