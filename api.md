# Talk2SQL API Documentation

This document outlines the REST API endpoints available in the Talk2SQL application for querying SQL databases using natural language.

## Database Management

### Connect to Database
- **URL:** `/connect`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "db_path": "/path/to/database.sqlite"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Connected to /path/to/database.sqlite",
    "schema_loaded": true,
    "examples_loaded": true,
    "db_name": "database.sqlite",
    "using_persistent_vectors": true,
    "thread_safe": true
  }
  ```

### List Available Databases
- **URL:** `/databases`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "databases": [
      {
        "name": "database.sqlite",
        "path": "/path/to/database.sqlite",
        "has_persisted_vectors": true
      }
    ],
    "current_db": "database.sqlite",
    "using_persistent_vectors": true
  }
  ```

### Get Vector Store Status
- **URL:** `/vector_store_status`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "vector_store": "persistent",
    "url": "https://qdrant.example.com",
    "collections": ["collection_questions", "collection_schema", "collection_docs"],
    "current_db_collections": {
      "questions": 10,
      "schema": 5,
      "docs": 2
    }
  }
  ```

### Upload Database File
- **URL:** `/upload_database`
- **Method:** `POST`
- **Body:** Form data with "file" field containing a .sqlite or .db file
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Database example.sqlite uploaded successfully",
    "path": "/path/to/databases/example.sqlite"
  }
  ```

## Training Data Management

### Upload Training Data
- **URL:** `/upload_training_data`
- **Method:** `POST`
- **Body:** Form data with "file" field containing a .json file of training examples
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Training data examples.json uploaded successfully",
    "examples_loaded": true,
    "path": "/path/to/training_data/database_training.json"
  }
  ```

### Record User Feedback
- **URL:** `/feedback`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "feedback": "up",  // or "down"
    "question": "How many teams are in the NBA?",
    "sql": "SELECT COUNT(*) FROM teams"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Feedback recorded and added to training examples",
    "feedback": "up",
    "stored_in_vectors": true
  }
  ```

### Cleanup Duplicate Training Examples
- **URL:** `/cleanup_duplicates`
- **Method:** `POST`
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Removed 5 duplicate entries from 2 files",
    "duplicates_removed": 5,
    "files_cleaned": 2
  }
  ```

### Get Training Example Format
- **URL:** `/training_example_format`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "example_format": {
      "natural_language": "How many teams are in the NBA?",
      "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
      "type": "counting"
    },
    "instructions": "To create your own training data..."
  }
  ```

## Query Management

### Ask Question
- **URL:** `/ask`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "question": "How many teams are in the NBA?",
    "db_id": "nba.sqlite",
    "visualize": true,
    "save_query": true
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
    "retry_count": 0,
    "question": "How many teams are in the NBA?",
    "timing": { ... },
    "data": [{"team_count": 30}],
    "columns": ["team_count"],
    "summary": "There are 30 teams in the NBA...",
    "visualization": "..."
  }
  ```

### Explain SQL Query
- **URL:** `/explain_sql`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
    "explanation": "This query counts the number of rows in the 'team' table..."
  }
  ```

### Generate Follow-up Questions
- **URL:** `/follow_up_questions`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "question": "How many teams are in the NBA?",
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
    "result_info": "The query returned 30 teams",
    "n": 3
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "question": "How many teams are in the NBA?",
    "followup_questions": [
      "Which conference has more teams, Eastern or Western?",
      "When was the most recent team added to the NBA?",
      "Which NBA team has won the most championships?"
    ]
  }
  ```

### Get Query History
- **URL:** `/history`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "history": [
      {
        "id": "unique_id",
        "question": "How many teams are in the NBA?",
        "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
        "timestamp": "2023-04-01T12:00:00",
        "success": true,
        "data": [{"team_count": 30}],
        "columns": ["team_count"],
        "visualization": "...",
        "summary": "There are 30 teams in the NBA..."
      }
    ]
  }
  ```

### Analyze Query Patterns
- **URL:** `/analyze`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "analysis": {
      "common_errors": [...],
      "success_rate": 0.85,
      "retry_stats": {...}
    }
  }
  ```

### Export Query History
- **URL:** `/export_history`
- **Method:** `GET`
- **Parameters:**
  - `format`: One of "json", "csv", or "full_csv"
  - `id`: (Optional) Specific query ID to export
- **Response:** File download in the specified format

### Get Metrics
- **URL:** `/metrics`
- **Method:** `GET`
- **Parameters:**
  - `time_range`: One of "all", "day", "week", or "month"
  - `limit`: Maximum number of queries to analyze
- **Response:**
  ```json
  {
    "status": "success",
    "time_range": "week",
    "total_queries": 100,
    "successful_queries": 85,
    "error_queries": 15,
    "success_rate": 85.0,
    "retry_metrics": {...},
    "performance_metrics": {...},
    "time_series": {...},
    "complexity_metrics": {...},
    "error_analysis": [...],
    "query_pattern_analysis": [...]
  }
  ```

## Voice Assistant Features

### Record Audio
- **URL:** `/record_audio`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "duration": 10
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "audio_path": "/audio_cache/temp_audio.wav",
    "duration": 10
  }
  ```

### Upload Audio File
- **URL:** `/upload_audio`
- **Method:** `POST`
- **Body:** Form data with "file" field containing an audio file
- **Response:**
  ```json
  {
    "transcription": "How many teams are in the NBA?",
    "status": "success"
  }
  ```

### Transcribe Audio
- **URL:** `/transcribe`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "audio_path": "/audio_cache/temp_audio.wav"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "text": "How many teams are in the NBA?"
  }
  ```

### Text to Speech
- **URL:** `/text_to_speech`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "text": "There are 30 teams in the NBA.",
    "voice": "Celeste-PlayAI"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "audio_path": "/path/to/audio_cache/speech_12345.wav",
    "audio_base64": "base64_encoded_audio_data"
  }
  ```

### Voice Assistant
- **URL:** `/voice_assistant`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "audio_path": "/audio_cache/temp_audio.wav",
    "voice": "Celeste-PlayAI"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "question": "How many teams are in the NBA?",
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
    "data": [{"team_count": 30}],
    "columns": ["team_count"],
    "summary": "There are 30 teams in the NBA...",
    "has_visualization": true,
    "visualization": "...",
    "audio_path": "/path/to/audio_cache/response_speech_12345.wav",
    "audio_base64": "base64_encoded_audio_data"
  }
  ```

### Streaming Voice Assistant
- **URL:** `/voice_assistant_stream`
- **Method:** `GET`
- **Parameters:**
  - `audio_path`: Path to audio file to process
  - `voice`: Voice ID to use for response
- **Response:** Server-sent events stream with the following event types:
  - `transcription`: The transcribed question
  - `sql`: The generated SQL query
  - `data`: The query results
  - `visualization`: Visualization of the data (if available)
  - `summary`: Summary of the query results
  - `audio`: Audio data for the spoken response
  - `error`: Any error messages

### Get Available Voices
- **URL:** `/available_voices`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "success",
    "english_voices": [
      "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI",
      "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI",
      "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI",
      "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI",
      "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
    ],
    "arabic_voices": [
      "Abla-PlayAI", "Bashir-PlayAI", "Daliya-PlayAI", "Essa-PlayAI"
    ]
  }
  ```
