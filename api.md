# Talk2SQL API Documentation

This document provides details and examples for the Talk2SQL API endpoints.

## Table of Contents

- [Database Management](#database-management)
  - [Connect to Database](#connect-to-database)
  - [List Databases](#list-databases)
  - [Upload Database](#upload-database)
  - [Vector Store Status](#vector-store-status)
  - [Query History Status](#query-history-status)
- [Training & Learning](#training--learning)
  - [Upload Training Data](#upload-training-data)
  - [Record Feedback](#record-feedback)
  - [Cleanup Duplicates](#cleanup-duplicates)
  - [Training Example Format](#training-example-format)
- [Query Operations](#query-operations)
  - [Ask Question](#ask-question)
  - [Ask Question (Streaming)](#ask-question-streaming)
  - [Follow-up Questions](#follow-up-questions)
  - [Query History](#query-history)
  - [Export History](#export-history)
  - [Analyze Patterns](#analyze-patterns)
  - [Metrics](#metrics)
- [Voice Features](#voice-features)
  - [Record Audio](#record-audio)
  - [Upload Audio](#upload-audio)
  - [Transcribe Audio](#transcribe-audio)
  - [Text to Speech](#text-to-speech)
  - [Voice Assistant](#voice-assistant)
  - [Voice Assistant (Streaming)](#voice-assistant-streaming)
  - [Available Voices](#available-voices)

---

## Database Management

### Connect to Database

Connects to a specified SQLite database.

**Endpoint:** `/connect`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "db_path": "/path/to/your/database.sqlite"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Connected to /path/to/your/database.sqlite",
  "schema_loaded": true,
  "examples_loaded": true,
  "db_name": "database.sqlite",
  "using_persistent_vectors": true,
  "thread_safe": true
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/connect \
  -H "Content-Type: application/json" \
  -d '{"db_path": "/databases/query_history.sqlite"}'
```

### List Databases

Lists all available databases in the configured database folder.

**Endpoint:** `/databases`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "databases": [
    {
      "name": "database1.sqlite",
      "path": "/path/to/database1.sqlite",
      "has_persisted_vectors": true,
      "is_query_history": false
    },
    {
      "name": "database2.sqlite",
      "path": "/path/to/database2.sqlite",
      "has_persisted_vectors": false,
      "is_query_history": false
    }
  ],
  "current_db": "database1.sqlite",
  "using_persistent_vectors": true
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/databases
```

### Upload Database

Uploads a new SQLite database file.

**Endpoint:** `/upload_database`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

**Request Parameters:**
- `file`: The SQLite database file to upload (.sqlite or .db)

**Response:**

```json
{
  "status": "success",
  "message": "Database example.sqlite uploaded successfully",
  "path": "/path/to/databases/example.sqlite"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/upload_database \
  -F "file=@/local/path/to/example.sqlite"
```

### Vector Store Status

Gets the status of the Qdrant vector store (for admin purposes).

**Endpoint:** `/vector_store_status`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "vector_store": "persistent",
  "url": "https://your-qdrant-instance.cloud",
  "collections": ["db1_questions", "db1_schema", "db1_docs"],
  "current_db_collections": {
    "questions": 50,
    "schema": 10,
    "docs": 25
  }
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/vector_store_status
```

### Query History Status

Checks the status of the query history database.

**Endpoint:** `/query_history_status`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "query_history_path": "/path/to/databases/query_history.sqlite",
  "exists": true,
  "in_databases_folder": true,
  "file_info": {
    "size_bytes": 1024000,
    "created": "2023-05-10T12:34:56",
    "modified": "2023-05-15T09:12:34",
    "query_count": 150
  }
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/query_history_status
```

---

## Training & Learning

### Upload Training Data

Uploads a JSON file containing training examples for the current database.

**Endpoint:** `/upload_training_data`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

**Request Parameters:**
- `file`: The JSON file containing training examples

**Response:**

```json
{
  "status": "success",
  "message": "Training data examples.json uploaded successfully",
  "examples_loaded": true,
  "path": "/path/to/training_data/database_training.json"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/upload_training_data \
  -F "file=@/local/path/to/examples.json"
```

### Record Feedback

Records user feedback (thumbs up/down) on query results.

**Endpoint:** `/feedback`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "feedback": "up",
  "question": "How many users registered in January?",
  "sql": "SELECT COUNT(*) FROM users WHERE strftime('%m', registration_date) = '01'"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Feedback recorded and added to training examples",
  "feedback": "up",
  "stored_in_vectors": true
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": "up",
    "question": "How many users registered in January?",
    "sql": "SELECT COUNT(*) FROM users WHERE strftime('\''%m'\'', registration_date) = '\''01'\''"
  }'
```

### Cleanup Duplicates

Removes duplicate entries from feedback and training files.

**Endpoint:** `/cleanup_duplicates`
**Method:** `POST`

**Response:**

```json
{
  "status": "success",
  "message": "Removed 5 duplicate entries from 2 files",
  "duplicates_removed": 5,
  "files_cleaned": 2
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/cleanup_duplicates
```

### Training Example Format

Gets the expected format for training examples.

**Endpoint:** `/training_example_format`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "example_format": {
    "natural_language": "How many teams are in the NBA?",
    "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
    "type": "counting"
  },
  "instructions": "To create your own training data:\n1. Create a JSON file with an array of examples\n2. Each example should have 'natural_language' and 'sql' fields\n3. The 'type' field is optional and can be used for categorization\n4. Upload the file to add examples to the system"
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/training_example_format
```

---

## Query Operations

### Ask Question

Asks a natural language question to be converted to SQL and executed.

**Endpoint:** `/ask`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "question": "How many users registered last month?",
  "db_id": "users.sqlite",
  "visualize": true,
  "save_query": true
}
```

**Response:**

```json
{
  "status": "success",
  "sql": "SELECT COUNT(*) AS user_count FROM users WHERE strftime('%Y-%m', registration_date) = strftime('%Y-%m', 'now', '-1 month')",
  "retry_count": 0,
  "question": "How many users registered last month?",
  "used_memory": false,
  "timing": {
    "total_time_ms": 1250,
    "sql_generation_time_ms": 950,
    "sql_execution_time_ms": 250,
    "visualization_time_ms": 50
  },
  "data": [{"user_count": 156}],
  "columns": ["user_count"],
  "summary": "There were 156 users who registered last month according to the database."
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many users registered last month?",
    "db_id": "users.sqlite",
    "visualize": true
  }'
```

### Ask Question (Streaming)

Streaming version of the ask endpoint that returns results in real-time.

**Endpoint:** `/ask_stream`
**Method:** `GET`

**Query Parameters:**
- `question`: The natural language question to ask

**Response:**
Server-sent events stream with the following event types:
- `question`: Acknowledges receipt of question
- `sql`: Generated SQL query
- `data`: Query results
- `visualization`: Visualization data (if available)
- `summary`: Text summary of the results
- `followups`: Suggested follow-up questions
- `error`: Error message (if something goes wrong)
- `complete`: Signals the end of the response

**Example:**

```bash
curl -X GET "http://localhost:8080/ask_stream?question=How%20many%20users%20registered%20last%20month?"
```

### Follow-up Questions

Generates follow-up questions based on a previous query.

**Endpoint:** `/follow_up_questions`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "question": "How many users registered last month?",
  "sql": "SELECT COUNT(*) AS user_count FROM users WHERE strftime('%Y-%m', registration_date) = strftime('%Y-%m', 'now', '-1 month')",
  "result_info": "There were 156 users who registered last month.",
  "n": 3
}
```

**Response:**

```json
{
  "status": "success",
  "question": "How many users registered last month?",
  "followup_questions": [
    "How does this compare to user registrations from the previous month?",
    "What was the daily average of user registrations last month?",
    "Which days of the week had the most registrations last month?"
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/follow_up_questions \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many users registered last month?",
    "sql": "SELECT COUNT(*) AS user_count FROM users WHERE strftime('\''%Y-%m'\'', registration_date) = strftime('\''%Y-%m'\'', '\''now'\'', '\''-1 month'\'')",
    "result_info": "There were 156 users who registered last month.",
    "n": 3
  }'
```

### Query History

Gets the history of executed queries.

**Endpoint:** `/history`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "history": [
    {
      "id": "query_123",
      "question": "How many users registered last month?",
      "sql": "SELECT COUNT(*) AS user_count FROM users WHERE strftime('%Y-%m', registration_date) = strftime('%Y-%m', 'now', '-1 month')",
      "timestamp": "2023-05-15T10:23:45",
      "success": true,
      "used_memory": false,
      "total_time_ms": 1250,
      "sql_generation_time_ms": 950,
      "sql_execution_time_ms": 250,
      "visualization_time_ms": 50,
      "explanation_time_ms": 150,
      "data": [{"user_count": 156}],
      "columns": ["user_count"],
      "visualization": "{\"data\":[{\"type\":\"bar\",\"x\":[\"Last Month\"],\"y\":[156]}]}",
      "summary": "There were 156 users who registered last month."
    }
  ]
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/history
```

### Export History

Exports query history in various formats (JSON, CSV, or full CSV ZIP).

**Endpoint:** `/export_history`
**Method:** `GET`

**Query Parameters:**
- `format`: Export format (`json`, `csv`, or `full_csv`)
- `id` (optional): ID of a specific query to export

**Response:**
- For `json`: JSON data
- For `csv` or `full_csv`: ZIP file download

**Example:**

```bash
# Export all history as JSON
curl -X GET "http://localhost:8080/export_history?format=json"

# Export a specific query as CSV
curl -X GET "http://localhost:8080/export_history?format=csv&id=query_123"

# Export comprehensive history data
curl -X GET "http://localhost:8080/export_history?format=full_csv"
```

### Analyze Patterns

Analyzes query patterns and error trends.

**Endpoint:** `/analyze`
**Method:** `GET`

**Response:**

```json
{
  "status": "success",
  "analysis": {
    "error_patterns": [
      {
        "error_type": "syntax_error",
        "count": 15,
        "percentage": 7.5
      },
      {
        "error_type": "table_not_found",
        "count": 8,
        "percentage": 4.0
      }
    ],
    "common_questions": [
      {
        "pattern": "count_records",
        "examples": ["How many users", "Count the total"],
        "percentage": 30.0
      }
    ]
  }
}
```

**Example:**

```bash
curl -X GET http://localhost:8080/analyze
```

### Metrics

Gets evaluation metrics about query history and performance.

**Endpoint:** `/metrics`
**Method:** `GET`

**Query Parameters:**
- `time_range`: Time range for metrics (`all`, `day`, `week`, `month`)
- `limit`: Maximum number of history items to analyze

**Response:**
```json
{
  "status": "success",
  "total_queries": 100,
  "successful_queries": 85,
  "error_queries": 15,
  "success_rate": 85.0,
  "latency": {
    "p50_total_ms": 1200,
    "p95_total_ms": 3500,
    "stage_p95_ms": {
      "generation_ms": 2000,
      "execution_ms": 500,
      "visualization_ms": 300,
      "explanation_ms": 700
    },
    "mean_breakdown_pct": {
      "generation_pct": 60,
      "execution_pct": 15,
      "visualization_pct": 10,
      "explanation_pct": 15
    }
  },
  "retry_metrics": {
    "queries_with_retry": 20,
    "total_retries": 25,
    "retry_rate_pct": 20.0,
    "retry_success_rate_pct": 75.0
  },
  "memory_metrics": {
    "queries_with_memory": 30,
    "memory_usage_rate_pct": 30.0,
    "with_memory_success_rate_pct": 90.0,
    "without_memory_success_rate_pct": 82.85
  },
  "top_errors": [
    {"type": "syntax_error", "count": 8},
    {"type": "table_not_found", "count": 4},
    {"type": "column_not_found", "count": 2},
    {"type": "permission", "count": 1}
  ],
  "time_series": {
    "dates": ["2023-05-01", "2023-05-02", "2023-05-03"],
    "counts": [25, 35, 40],
    "success_counts": [20, 30, 35],
    "success_rates": [80.0, 85.71, 87.5],
    "retries": [5, 8, 7]
  }
}
```

**Example:**

```bash
curl -X GET "http://localhost:8080/metrics?time_range=week&limit=500"
```

---

## Voice Features

### Record Audio

Records audio from the user's microphone.

**Endpoint:** `/record_audio`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "duration": 10
}
```

**Response:**

```json
{
  "status": "success",
  "audio_path": "/audio_cache/temp_audio.wav",
  "duration": 10
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/record_audio \
  -H "Content-Type: application/json" \
  -d '{"duration": 10}'
```

### Upload Audio

Uploads an audio file for transcription.

**Endpoint:** `/upload_audio`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

**Request Parameters:**
- `file`: The audio file to upload

**Response:**

```json
{
  "transcription": "How many users registered last month?",
  "status": "success"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/upload_audio \
  -F "file=@/path/to/audio.wav"
```

### Transcribe Audio

Transcribes an audio file to text.

**Endpoint:** `/transcribe`
**Method:** `POST` or `GET`

**For POST:**
**Content-Type:** `application/json`
**Request Body:**

```json
{
  "audio_path": "/audio_cache/temp_audio.wav"
}
```

**For GET:**
**Query Parameters:**
- `audio_path`: Path to the audio file

**Response:**

```json
{
  "status": "success",
  "text": "How many users registered last month?"
}
```

**Example:**

```bash
# POST request
curl -X POST http://localhost:8080/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/audio_cache/temp_audio.wav"}'

# GET request
curl -X GET "http://localhost:8080/transcribe?audio_path=/audio_cache/temp_audio.wav"
```

### Text to Speech

Converts text to speech using Groq.

**Endpoint:** `/text_to_speech`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "text": "There were 156 users who registered last month.",
  "voice": "Celeste-PlayAI"
}
```

**Response:**

```json
{
  "status": "success",
  "audio_path": "/path/to/audio_cache/speech_1684234567.wav",
  "audio_base64": "base64_encoded_audio_data"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/text_to_speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "There were 156 users who registered last month.",
    "voice": "Celeste-PlayAI"
  }'
```

### Voice Assistant

Voice-based SQL assistant that processes audio input.

**Endpoint:** `/voice_assistant`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "audio_path": "/audio_cache/temp_audio.wav",
  "voice": "Celeste-PlayAI"
}
```

**Response:**

```json
{
  "status": "success",
  "question": "How many users registered last month?",
  "sql": "SELECT COUNT(*) AS user_count FROM users WHERE strftime('%Y-%m', registration_date) = strftime('%Y-%m', 'now', '-1 month')",
  "data": [{"user_count": 156}],
  "columns": ["user_count"],
  "summary": "There were 156 users who registered last month according to the database.",
  "has_visualization": true,
  "visualization": "{\"data\":[{\"type\":\"bar\",\"x\":[\"Last Month\"],\"y\":[156]}]}",
  "audio_path": "/path/to/audio_cache/response_speech_1684234567.wav",
  "audio_base64": "base64_encoded_audio_data"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/voice_assistant \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/audio_cache/temp_audio.wav",
    "voice": "Celeste-PlayAI"
  }'
```

### Voice Assistant (Streaming)

Streaming version of the voice assistant that returns results in real-time.

**Endpoint:** `/voice_assistant_stream`
**Method:** `GET`

**Query Parameters:**
- `audio_path`: Path to the audio file
- `voice`: Voice to use for text-to-speech (default: `Celeste-PlayAI`)

**Response:**
Server-sent events stream with the following event types:
- `transcription`: Transcribed question
- `sql`: Generated SQL query
- `data`: Query results
- `visualization`: Visualization data (if available)
- `summary`: Text summary of the results
- `audio`: Base64-encoded audio response
- `error`: Error message (if something goes wrong)

**Example:**

```bash
curl -X GET "http://localhost:8080/voice_assistant_stream?audio_path=/audio_cache/temp_audio.wav&voice=Celeste-PlayAI"
```

### Available Voices

Gets a list of available voices for text-to-speech.

**Endpoint:** `/available_voices`
**Method:** `GET`

**Response:**

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

**Example:**

```bash
curl -X GET http://localhost:8080/available_voices
```