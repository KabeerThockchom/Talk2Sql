import os
import json
import pandas as pd
import glob
import tempfile
import sounddevice as sd
import soundfile as sf
import time
import numpy as np
import base64
import hashlib
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from talk2sql.engine import Talk2SQLAzure
# from talk2sql.engine import Talk2SQL_anthropic
from talk2sql.utils import format_sql_with_xml_tags, extract_content_from_xml_tags
import groq
import threading
import sqlite3
import logging
import datetime
import io
import csv
import zipfile
import statistics
import traceback
import uuid
from collections import Counter, defaultdict
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with 'pip install python-dotenv' to load environment variables from .env file")

app = Flask(__name__)

DB_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'databases')

# Initialize Talk2SQLAzure with Azure OpenAI
config = {
    # "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
    # "claude_model": os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022"),
    # "temperature": 0.3,
    # "enable_thinking": False,
    # "thinking_budget_tokens": 2000,
    "azure_api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    "azure_api_version": os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
    "azure_deployment": os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini"),
    "azure_embedding_deployment": "text-embedding-ada-002",
    "temperature": 0.3,
    
    # Vector store settings - Use Qdrant Cloud if credentials are available
    "location": os.environ.get("QDRANT_URL", ":memory:"),  # URL or :memory:
    "api_key": os.environ.get("QDRANT_API_KEY", None),     # API key for Qdrant Cloud
    "prefer_grpc": True,  # Use gRPC for better performance
    
    # Retry settings
    "max_retry_attempts": 3,
    "save_query_history": True,
    "history_db_path": os.path.join(DB_FOLDER, "query_history.sqlite"),  # Store query history in databases folder
    
    # General settings
    "debug_mode": True,
}

# If we have a Qdrant URL and API key, switch to persistent vector store
using_persistent_vectors = (config["location"] != ":memory:" and config["api_key"] is not None)
if using_persistent_vectors:
    print(f"Using persistent vector storage at {config['location']}")
else:
    print("Using in-memory vector storage - embeddings will be lost when app restarts")

# Initialize Talk2SQL
Talk2SQL = Talk2SQLAzure(config)
# Talk2SQL = Talk2SQLAnthropic(config)

# Ensure query history saving is enabled
if not Talk2SQL.save_query_history:
    print("Enabling query history saving")
    Talk2SQL.save_query_history = True

# Log the query history database path
query_history_path = getattr(Talk2SQL, "history_db_path", config.get("history_db_path", "default location"))
print(f"Query history is being saved to: {query_history_path}")
print(f"Absolute path to query history: {os.path.abspath(query_history_path)}")
print(f"Database folder path: {DB_FOLDER}")
print(f"Database folder exists: {os.path.exists(DB_FOLDER)}")

# Ensure the path is set as an attribute if it's not already
if not hasattr(Talk2SQL, "history_db_path"):
    Talk2SQL.history_db_path = config.get("history_db_path")
    print(f"Setting history_db_path attribute to: {Talk2SQL.history_db_path}")

# Initialize Groq client for speech capabilities
groq_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
TRAINING_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
AUDIO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_cache')

# Configure Flask app
app.config['UPLOAD_FOLDER'] = AUDIO_FOLDER

# Ensure directories exist
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

print(f"Created or verified database folder at: {DB_FOLDER}")
print(f"Expected query history location: {os.path.join(DB_FOLDER, 'query_history.sqlit')}")

# Store the current database path
current_db_path = None
current_db_name = None
db_collection_created = {}  # Keep track of which databases have collections

# Thread-safe SQLite connection
_thread_local = threading.local()

def get_thread_safe_connection():
    """
    Gets a thread-safe connection to the current database.
    If there's no connection for the current thread, creates one.
    """
    thread_id = threading.get_ident()
    if not hasattr(_thread_local, 'connection'):
        logging.info(f"Creating new connection for thread {thread_id}")
        if app.config.get('DATABASE_PATH'):
            # Create a new connection with check_same_thread=False for this thread
            db_path = app.config.get('DATABASE_PATH')
            _thread_local.connection = sqlite3.connect(db_path, check_same_thread=False)
            logging.info(f"Created new SQLite connection for thread {thread_id}")
        else:
            logging.warning(f"No database selected for thread {thread_id}")
    
    return _thread_local.connection if hasattr(_thread_local, 'connection') else None

# Generate deterministic collection name for a database
def get_collection_name_for_db(db_path):
    # Create a hash of the database path
    db_hash = hashlib.md5(db_path.encode()).hexdigest()[:8]
    base_name = os.path.basename(db_path).replace('.', '_').replace('-', '_')
    # Combine the base name and hash for a unique but recognizable name
    return f"{base_name}_{db_hash}"

# Connect to a selected database
@app.route('/connect', methods=['POST'])
def connect_to_database():
    global current_db_path, current_db_name, db_collection_created, _thread_local
    
    db_path = request.json.get('db_path')
    
    # If no path provided, list available databases
    if not db_path:
        return jsonify({
            "status": "error", 
            "message": "No database path provided"
        })
    
    try:
        # Print some debug info
        print(f"Attempting to connect to database at: {db_path}")
        print(f"File exists: {os.path.exists(db_path)}")
        if os.path.exists(db_path):
            print(f"File size: {os.path.getsize(db_path)} bytes")
        
        # First connect to the database to ensure we can access it
        Talk2SQL.connect_to_sqlite(db_path)
        current_db_path = db_path
        current_db_name = os.path.basename(db_path)
        
        # Store the database path in app configuration for thread-safe access
        app.config['DATABASE_PATH'] = db_path
        
        # Reset thread-local storage since we're connecting to a new database
        # This ensures each thread will create a new connection to the new database
        if hasattr(_thread_local, 'connection'):
            # Close existing connection if it exists
            try:
                _thread_local.connection.close()
            except:
                pass
            _thread_local.connection = None
        
        # Create a connection for the main thread
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _thread_local.connection = conn
        logging.info(f"Created thread-safe connection for main thread: {threading.get_ident()}")
        
        # Check if we've already created a collection for this database
        collections_exist = False
        if using_persistent_vectors:
            # Get the collection name for this database
            collection_name = get_collection_name_for_db(db_path)
            
            # Set the collection names for this database
            Talk2SQL.questions_collection = f"{collection_name}_questions"
            Talk2SQL.schema_collection = f"{collection_name}_schema"
            Talk2SQL.docs_collection = f"{collection_name}_docs"
            
            # Check if we've already created these collections and they have data
            if db_collection_created.get(db_path, False):
                print(f"Using existing collections for {db_path}")
                collections_exist = True
            else:
                try:
                    # Check if collections exist and have data
                    exists_and_has_data = False
                    try:
                        # Try to count records in the questions collection
                        count = Talk2SQL.qdrant_client.count(
                            collection_name=Talk2SQL.questions_collection
                        ).count
                        exists_and_has_data = count > 0
                        if exists_and_has_data:
                            print(f"Found existing collection with {count} examples")
                    except Exception as e:
                        print(f"Collection doesn't exist yet or error checking: {e}")
                    
                    if not exists_and_has_data:
                        # Create collections if they don't exist or are empty
                        Talk2SQL._setup_collections()
                        print(f"Created vector collections for {db_path}")
                    else:
                        print(f"Using existing collections with data for {db_path}")
                        collections_exist = True
                        
                    db_collection_created[db_path] = True
                except Exception as e:
                    print(f"Error creating collections: {e}")
        
        # Load database schema - only if we haven't created this collection before
        # or if we're not using persistent vectors
        schema = ""
        if not using_persistent_vectors or not collections_exist:
            schema = get_db_schema()
            if schema:
                print(f"Loaded database schema: {len(schema)} characters")
            else:
                print("Warning: No schema loaded")
        else:
            print("Using existing schema from vector store")
            
        # Load training examples for this database - only if needed
        # or if we're not using persistent vectors
        examples_loaded = False
        if not using_persistent_vectors or not collections_exist:
            examples_loaded = load_training_examples()
            if examples_loaded:
                print("Successfully loaded training examples")
            else:
                print("Warning: Failed to load training examples")
        else:
            examples_loaded = True
            print("Using existing training examples from vector store")
        
        # Generate starter questions when connecting to a database
        starter_questions = []
        if schema:
            try:
                # Generate 5 starter questions based on the schema
                starter_questions = Talk2SQL.generate_starter_questions(schema, 5)
                print(f"Generated {len(starter_questions)} starter questions")
            except Exception as e:
                print(f"Error generating starter questions: {e}")
                # Don't let this error prevent connection
            
        return jsonify({
            "status": "success", 
            "message": f"Connected to {db_path}",
            "schema_loaded": bool(schema) or collections_exist,
            "examples_loaded": examples_loaded,
            "db_name": current_db_name,
            "using_persistent_vectors": using_persistent_vectors,
            "thread_safe": True,
            "starter_questions": starter_questions
        })
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# Get a list of available databases
@app.route('/databases', methods=['GET'])
def list_databases():
    try:
        # List all .sqlite files in the databases folder
        db_files = glob.glob(os.path.join(DB_FOLDER, '*.sqlite'))
        db_files.extend(glob.glob(os.path.join(DB_FOLDER, '*.db')))
        
        # Log the found databases
        print(f"Found database files: {db_files}")
        
        # Format for frontend
        databases = []
        for db_file in db_files:
            db_name = os.path.basename(db_file)
            # Check if we have a persisted collection for this database
            has_persisted = False
            if using_persistent_vectors:
                has_persisted = db_collection_created.get(db_file, False)
            
            # Check if this is the query history database
            query_history_path = getattr(Talk2SQL, "history_db_path", config.get("history_db_path", ""))
            is_query_history = db_file == query_history_path
            
            databases.append({
                'name': db_name,
                'path': db_file,
                'has_persisted_vectors': has_persisted,
                'is_query_history': is_query_history
            })
            
        # Also check if the default NBA database exists
        default_path = '/Users/kabeerthockchom/Desktop/Talk2SQL/Talk2SQL/nba.sqlite'
        if os.path.exists(default_path) and default_path not in [db['path'] for db in databases]:
            databases.append({
                'name': 'nba.sqlite (default)',
                'path': default_path,
                'has_persisted_vectors': False,
                'is_query_history': False
            })
            
        return jsonify({
            "status": "success",
            "databases": databases,
            "current_db": current_db_name,
            "using_persistent_vectors": using_persistent_vectors
        })
    except Exception as e:
        print(f"Error listing databases: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Get Qdrant vector store status (for admin purposes)
@app.route('/vector_store_status', methods=['GET'])
def vector_store_status():
    try:
        if not using_persistent_vectors:
            return jsonify({
                "status": "success",
                "vector_store": "in-memory",
                "message": "Using in-memory vector storage"
            })
        
        # Get list of collections
        collections = Talk2SQL.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Get counts for current database collections
        counts = {}
        if current_db_path:
            collection_name = get_collection_name_for_db(current_db_path)
            for coll_type in ["questions", "schema", "docs"]:
                full_name = f"{collection_name}_{coll_type}"
                if full_name in collection_names:
                    try:
                        count = Talk2SQL.qdrant_client.count(
                            collection_name=full_name
                        ).count
                        counts[coll_type] = count
                    except:
                        counts[coll_type] = "error"
        
        return jsonify({
            "status": "success",
            "vector_store": "persistent",
            "url": config["location"],
            "collections": collection_names,
            "current_db_collections": counts if current_db_path else None
        })
    except Exception as e:
        print(f"Error getting vector store status: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Check query history database status
@app.route('/query_history_status', methods=['GET'])
def query_history_status():
    try:
        query_history_path = getattr(Talk2SQL, "history_db_path", config.get("history_db_path", "unknown"))
        exists = os.path.exists(query_history_path) if isinstance(query_history_path, str) else False
        
        print(f"Checking query history status: {query_history_path}")
        print(f"File exists: {exists}")
        
        # If the file exists, get some basic info
        file_info = {}
        if exists:
            file_info["size_bytes"] = os.path.getsize(query_history_path)
            file_info["created"] = datetime.datetime.fromtimestamp(os.path.getctime(query_history_path)).isoformat()
            file_info["modified"] = datetime.datetime.fromtimestamp(os.path.getmtime(query_history_path)).isoformat()
            
            # Try to get query count
            try:
                history = Talk2SQL.get_query_history(limit=None)
                
                # Make sure data is serializable before storing in file_info
                serializable_history = []
                for item in history:
                    # Convert any bytes objects in each history item
                    for key, value in list(item.items()):
                        if isinstance(value, bytes):
                            try:
                                # Convert bytes to base64 string
                                import base64
                                item[key] = base64.b64encode(value).decode('utf-8')
                            except:
                                # If conversion fails, remove the key
                                item[key] = None
                    serializable_history.append(item)
                    
                file_info["query_count"] = len(serializable_history) if serializable_history else 0
                print(f"Found {file_info['query_count']} queries in history database")
            except Exception as e:
                file_info["query_count_error"] = str(e)
                print(f"Error getting query count: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({
            "status": "success",
            "query_history_path": query_history_path,
            "exists": exists,
            "in_databases_folder": query_history_path.startswith(DB_FOLDER) if isinstance(query_history_path, str) else False,
            "file_info": file_info if exists else None
        })
    except Exception as e:
        print(f"Error checking query history status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})
        
# Upload a new database file
@app.route('/upload_database', methods=['POST'])
def upload_database():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
        
    if not (file.filename.endswith('.sqlite') or file.filename.endswith('.db')):
        return jsonify({"status": "error", "message": "Only .sqlite or .db files are allowed"})
    
    try:
        # Save the file to the databases folder
        file_path = os.path.join(DB_FOLDER, file.filename)
        file.save(file_path)
        
        return jsonify({
            "status": "success", 
            "message": f"Database {file.filename} uploaded successfully",
            "path": file_path
        })
    except Exception as e:
        print(f"Error uploading database: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Upload training data file
@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
        
    if not file.filename.endswith('.json'):
        return jsonify({"status": "error", "message": "Only .json files are allowed"})
    
    try:
        # Save the file to the training_data folder with a name related to the current DB
        if current_db_name:
            base_name = os.path.splitext(current_db_name)[0]
            file_path = os.path.join(TRAINING_DATA_FOLDER, f"{base_name}_training.json")
        else:
            file_path = os.path.join(TRAINING_DATA_FOLDER, file.filename)
            
        file.save(file_path)
        
        # Load the training data immediately
        examples_loaded = load_training_examples()
        
        # If using persistent vectors, mark this database as having vectors
        if using_persistent_vectors and current_db_path:
            db_collection_created[current_db_path] = True
        
        return jsonify({
            "status": "success", 
            "message": f"Training data {file.filename} uploaded successfully",
            "examples_loaded": examples_loaded,
            "path": file_path
        })
    except Exception as e:
        print(f"Error uploading training data: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Load training examples from all relevant sources
def load_training_examples():
    try:
        # Try to load from database-specific training data file first
        examples_loaded = False
        loaded_count = 0
        
        # If we have a current database, look for a database-specific training file
        if current_db_name:
            base_name = os.path.splitext(current_db_name)[0]
            db_specific_path = os.path.join(TRAINING_DATA_FOLDER, f"{base_name}_training.json")
            
            if os.path.exists(db_specific_path):
                print(f"Loading database-specific training examples from: {db_specific_path}")
                examples_loaded, count = load_training_file(db_specific_path)
                loaded_count += count
                if examples_loaded:
                    print(f"Loaded {count} database-specific examples into vector store")
        
        # Also load user feedback for this database
        if current_db_name:
            user_feedback_path = os.path.join(TRAINING_DATA_FOLDER, f"{base_name}_feedback.json")
            if os.path.exists(user_feedback_path):
                print(f"Loading user feedback examples from: {user_feedback_path}")
                feedback_loaded, count = load_training_file(user_feedback_path)
                examples_loaded = feedback_loaded or examples_loaded
                loaded_count += count
                if feedback_loaded:
                    print(f"Loaded {count} user feedback examples into vector store")
        
        # If no database-specific file, try the default one
        if not examples_loaded:
            # Look in the current directory first
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ground_truth_data.json')
            if os.path.exists(default_path):
                print(f"Loading default training examples from: {default_path}")
                default_loaded, count = load_training_file(default_path)
                examples_loaded = default_loaded
                loaded_count += count
                if default_loaded:
                    print(f"Loaded {count} default examples into vector store")
        
        if examples_loaded and using_persistent_vectors:
            print(f"Total of {loaded_count} examples stored in persistent vector database")
        
        return examples_loaded
    except Exception as e:
        print(f"Error loading training examples: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load training examples from a specific file
def load_training_file(file_path):
    try:
        print(f"Loading training examples from: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Add examples to Talk2SQL
        added_count = 0
        for example in data:
            question = example['natural_language']
            sql = example['sql']
            
            # Format SQL with XML tags for storage
            formatted_sql = format_sql_with_xml_tags(sql)
            
            # Add to Talk2SQL
            try:
                # Extract the SQL without tags for adding to the database
                Talk2SQL.add_question_sql(question, sql)
                added_count += 1
            except Exception as e:
                print(f"Error adding example: {question}, error: {e}")
                continue
        
        print(f"Added {added_count} of {len(data)} training examples")
        return added_count > 0, added_count
    except Exception as e:
        print(f"Error loading training file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

# Record user feedback (thumbs up/down)
@app.route('/feedback', methods=['POST'])
def record_feedback():
    feedback = request.json.get('feedback', '')
    question = request.json.get('question', '')
    sql = request.json.get('sql', '')
    
    if not question or not sql or feedback not in ['up', 'down']:
        return jsonify({"status": "error", "message": "Invalid feedback data"})
    
    try:
        # For thumbs up, add the example to the database-specific feedback file
        if feedback == 'up' and current_db_name:
            base_name = os.path.splitext(current_db_name)[0]
            feedback_path = os.path.join(TRAINING_DATA_FOLDER, f"{base_name}_feedback.json")
            
            # Create or load existing feedback file
            examples = []
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    examples = json.load(f)
            
            # Check for duplicates before adding
            is_duplicate = False
            for example in examples:
                # Check if the same question and SQL already exist
                if example.get('natural_language') == question and example.get('sql') == sql:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print(f"Duplicate entry found, not adding to feedback: {question}")
                return jsonify({
                    "status": "success", 
                    "message": "This example is already in the training data",
                    "feedback": feedback,
                    "duplicate": True
                })
            
            # Add new example
            examples.append({
                "natural_language": question,
                "sql": sql,
                "type": "user_feedback"
            })
            
            # Save updated file
            with open(feedback_path, 'w') as f:
                json.dump(examples, f, indent=2)
            
            # Also add to Talk2SQL immediately
            try:
                Talk2SQL.add_question_sql(question, sql)
                print(f"Added feedback example to vector store: {question}")
                
                # Update persistent storage tracking if using Qdrant
                if using_persistent_vectors and current_db_path:
                    db_collection_created[current_db_path] = True
                
                return jsonify({
                    "status": "success", 
                    "message": "Feedback recorded and added to training examples",
                    "feedback": feedback,
                    "stored_in_vectors": True
                })
            except Exception as e:
                print(f"Error adding feedback to vector store: {e}")
                return jsonify({
                    "status": "success", 
                    "message": "Feedback recorded in file but not added to vector store",
                    "feedback": feedback,
                    "stored_in_vectors": False,
                    "error": str(e)
                })
        else:
            return jsonify({
                "status": "success", 
                "message": "Feedback recorded",
                "feedback": feedback
            })
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Clean up duplicate entries from feedback and training files
@app.route('/cleanup_duplicates', methods=['POST'])
def cleanup_duplicates():
    try:
        duplicates_removed = 0
        files_cleaned = 0
        
        # Find all JSON files in the training data folder
        training_files = glob.glob(os.path.join(TRAINING_DATA_FOLDER, '*.json'))
        
        for file_path in training_files:
            try:
                with open(file_path, 'r') as f:
                    examples = json.load(f)
                
                # Track unique examples by question + SQL combination
                unique_examples = []
                seen_pairs = set()
                
                for example in examples:
                    question = example.get('natural_language', '')
                    sql = example.get('sql', '')
                    
                    # Create a key for this example
                    example_key = f"{question}::{sql}"
                    
                    # Add only if not seen before
                    if example_key not in seen_pairs:
                        seen_pairs.add(example_key)
                        unique_examples.append(example)
                    else:
                        duplicates_removed += 1
                
                # Only write back if duplicates were found
                if len(unique_examples) < len(examples):
                    with open(file_path, 'w') as f:
                        json.dump(unique_examples, f, indent=2)
                    files_cleaned += 1
                    print(f"Cleaned {len(examples) - len(unique_examples)} duplicates from {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Re-load training examples if duplicates were removed
        if duplicates_removed > 0:
            load_training_examples()
        
        return jsonify({
            "status": "success",
            "message": f"Removed {duplicates_removed} duplicate entries from {files_cleaned} files",
            "duplicates_removed": duplicates_removed,
            "files_cleaned": files_cleaned
        })
    except Exception as e:
        print(f"Error cleaning up duplicates: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Get schema information from the database
def get_db_schema():
    try:
        print("Attempting to extract database schema...")
        
        # First, verify database connection
        try:
            test_df = Talk2SQL.run_sql("SELECT 1")
            print(f"Database connection test: {not test_df.empty}")
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return ""
        
        # Get all tables using a more robust query
        print("Querying for tables...")
        tables_df = Talk2SQL.run_sql("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        print(f"Tables found: {len(tables_df) if not tables_df.empty else 0}")
        if not tables_df.empty:
            print(f"Table names: {', '.join(tables_df['name'].tolist())}")
        
        if tables_df.empty:
            print("No tables found in database - checking if database file exists and has content")
            # This could indicate an issue with the database file
            return ""
            
        schema_definitions = []
        table_count = 0
        
        # Get create statements for each table
        for table in tables_df['name']:
            print(f"Processing table: {table}")
            try:
                create_stmt_df = Talk2SQL.run_sql(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
                if not create_stmt_df.empty and create_stmt_df['sql'][0] is not None:
                    schema_definitions.append(create_stmt_df['sql'][0])
                    
                    # Also get a sample of the data to better understand the schema
                    sample_df = Talk2SQL.run_sql(f"SELECT * FROM {table} LIMIT 1")
                    columns = list(sample_df.columns)
                    schema_definitions.append(f"-- Table {table} columns: {', '.join(columns)}")
                    
                    # Add column descriptions (simplified to avoid potential errors)
                    column_info = []
                    for col in columns:
                        try:
                            # Only get distinct values for columns (simpler approach)
                            distinct_df = Talk2SQL.run_sql(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
                            if not distinct_df.empty and distinct_df.iloc[0, 0] < 10:
                                values_df = Talk2SQL.run_sql(f"SELECT DISTINCT {col} FROM {table} LIMIT 10")
                                values = values_df[values_df.columns[0]].tolist()
                                column_info.append(f"-- Column {col} possible values: {', '.join(map(str, values))}")
                        except Exception as e:
                            print(f"Error getting column stats for {table}.{col}: {e}")
                    
                    schema_definitions.extend(column_info)
                    table_count += 1
            except Exception as e:
                print(f"Error getting schema for table {table}: {e}")
        
        print(f"Extracted schema for {table_count} tables")
        
        # Join all schema definitions
        full_schema = '\n\n'.join(schema_definitions)
        
        # Add schema to Talk2SQL
        if full_schema:
            Talk2SQL.add_schema(full_schema)
            
            # Add a more readable description
            description = f"""
            This is a database with {table_count} tables.
            """
            Talk2SQL.add_documentation(description)
            
        return full_schema
    except Exception as e:
        print(f"Error getting schema: {e}")
        import traceback
        traceback.print_exc()
        return ""

# Ask question to Talk2SQL
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the question, database ID, and visualization flag
        question = request.json.get('question')
        db_id = request.json.get('db_id', request.json.get('database'))
        visualize = request.json.get('visualize', True)
        
        # Check if we should save the query to history
        save_query = request.json.get('save_query', True)
        original_save_query = Talk2SQL.save_query_history
        
        if not save_query:
            Talk2SQL.save_query_history = False
        
        # Ensure database connection
        get_thread_safe_connection()
        
        # Execute the query
        result = Talk2SQL.smart_query(question, print_results=False, visualize=visualize)
        
        # Restore save_query_history setting
        Talk2SQL.save_query_history = original_save_query
        
        # Prepare the response
        response = {
            "status": "success" if result["success"] else "error",
            "sql": result["sql"],
            "retry_count": result.get("retry_count", 0),
            "question": question,
            "used_memory": result.get("used_memory", False)  # Include memory usage
        }
        
        # Add timing information if available
        if "timing" in result:
            response["timing"] = result["timing"]
        
        if result["success"]:
            # If successful, include data and visualization
            df = result["data"]
            if df is not None:
                response["data"] = df.to_dict(orient='records')
                response["columns"] = df.columns.tolist()
                
                # The summary is already generated in smart_query and stored in the result
                if "summary" in result:
                    response["summary"] = result["summary"]
                
                # Include visualization if available
                if result["visualization"] is not None:
                    try:
                        response["visualization"] = result["visualization"].to_json()
                    except Exception as e:
                        print(f"Error converting visualization to JSON: {e}")
        else:
            # If error, include the error message
            response["error"] = result.get("error", "Unknown error")
            
            # Include the corrected SQL if available
            if "corrected_sql" in result:
                response["corrected_sql"] = result["corrected_sql"]
                
        return jsonify(response)
    except Exception as e:
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)})

# Generate follow-up questions
@app.route('/follow_up_questions', methods=['POST'])
def follow_up_questions_endpoint():
    question = request.json.get('question', '')
    sql = request.json.get('sql', '')
    result_info = request.json.get('result_info', '')
    n = request.json.get('n', 3)
    
    if not question or not sql:
        return jsonify({"status": "error", "message": "Question and SQL query are required"})
    
    try:
        # Call the generate_follow_up_questions method
        followups = Talk2SQL.generate_follow_up_questions(
            question=question,
            sql=sql,
            result_info=result_info,
            n=n
        )
        
        return jsonify({
            "status": "success",
            "question": question,
            "followup_questions": followups
        })
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# Generate a summary of the data using Azure OpenAI
def generate_data_summary(question, result):
    try:
                # Start timing
        summary_start_time = datetime.datetime.now()
        # Create a prompt for OpenAI to summarize the data
        prompt = f"""
        The user asked: "{question}"
        
        I ran the following SQL query:
        {result["sql"]}
        
        The query returned a dataframe with {len(result["data"])} rows and {len(result["data"].columns)} columns.
        Column names: {', '.join(result["data"].columns)}
        
        Here's a sample of the data:
        {result["data"].head(5).to_string()}
        
        Please provide a clear, concise summary of this data that directly answers the user's question, citing the tables and columns used to answer the question.
        Do not repeat the data, just summarize it.
        
        Include key insights, trends, or patterns if relevant. Keep it brief and focused.

        Example of your task:
        Question: How many teams are in the NBA?
        SQL: SELECT t.full_name, ROUND(AVG(gi.attendance), 0) as avg_attendance
                    FROM game g
                    JOIN game_info gi ON g.game_id = gi.game_id
                    JOIN team t ON g.team_id_home = t.id
                    WHERE gi.attendance > 0
                    GROUP BY t.id, t.full_name
                    ORDER BY avg_attendance DESC
                    LIMIT 1
        Data:full_name	avg_attendance
        18622	Toronto Raptors

        Assistant:
        The Toronto Raptors have the highest average attendance in the NBA with 18,622 fans per game, this was inferred using the table game_info and the column attendance.
        
        """
        
        # Create the messages for the prompt
        messages = [
            {"role": "system", "content": "You are a helpful AI that summarizes data and answers questions."},
            {"role": "user", "content": prompt}
        ]
        
        # Use Talk2SQL's client (Anthropic client uses claude_model)
        response = Talk2SQL.client.chat.completions.create(
            # model=config["claude_model"],
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        
        summary = response.choices[0].message.content
        
        summary_end_time = datetime.datetime.now()
        summary_time_ms = (summary_end_time - summary_start_time).total_seconds() * 1000
        # Get recent history and update with summary
        recent_history = Talk2SQL.get_query_history(successful_only=True, limit=10)
        for entry in recent_history:
            if entry.get("question") == question and "summary" not in entry:
                # Re-record the query attempt with the summary added
                Talk2SQL.record_query_attempt(
                    question=entry.get("question"),
                    sql=entry.get("sql"),
                    success=True,
                    retry_count=entry.get("retry_count", 0),
                    data=entry.get("data"),
                    columns=entry.get("columns"),
                    visualization=entry.get("visualization"),
                    summary=summary,
                    explanation_time_ms=summary_time_ms
                )
                print(f"Added summary to query history for: '{question}'")
                break
                
        return summary, summary_time_ms
    except Exception as e:
        print(f"Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I couldn't generate a summary of the data."

# Get query history
@app.route('/history', methods=['GET'])
def get_history():
    try:
        # Get filter parameters from request
        time_range = request.args.get('time_range', 'all')  # all | day | week | month
        status_filter = request.args.get('status', 'all')   # all | success | failed
        
        print(f"DEBUG: history request with time_range={time_range}, status_filter={status_filter}")
        
        # Get full history first
        history = Talk2SQL.get_query_history()
        print(f"DEBUG: Raw history returned from Talk2SQL.get_query_history(): {len(history) if history else 0} items")
        
        if not history:
            print("DEBUG: No history found from Talk2SQL.get_query_history()")
            # Return empty array for frontend to handle
            return jsonify({"status": "success", "history": []})
        
        # Apply time filter if specified
        if time_range != "all":
            now = datetime.datetime.now()
            filtered_history = []
            for item in history:
                try:
                    ts = item.get("timestamp", "")
                    if ts:
                        item_date = _parse_iso(ts)
                        # Filter based on time range
                        delta = {"day": 1, "week": 7, "month": 30}[time_range]
                        if (now - item_date).days <= delta:
                            filtered_history.append(item)
                except Exception as e:
                    print(f"Error filtering by time: {e}")
            history = filtered_history
            print(f"DEBUG: After time filtering: {len(history)} items")
            
        # Apply status filter if specified
        if status_filter != "all":
            filtered_history = []
            for item in history:
                success_status = item.get("success", False)
                if (status_filter == "success" and success_status) or (status_filter == "failed" and not success_status):
                    filtered_history.append(item)
            history = filtered_history
            print(f"DEBUG: After status filtering: {len(history)} items")
        
        # Enhance history items with additional data if available
        enhanced_history = []
        for item in history:
            # Include basic information
            history_item = {
                "id": item.get("id", str(hash(f"{item.get('question', '')}-{item.get('timestamp', '')}"))),
                "question": item.get("question", ""),
                "sql": item.get("sql", ""),
                "timestamp": item.get("timestamp", ""),
                "success": item.get("success", False),
                "used_memory": item.get("used_memory", False),  # Include memory usage
                # Add timing fields
                "total_time_ms": item.get("total_time_ms"),
                "sql_generation_time_ms": item.get("sql_generation_time_ms"),
                "sql_execution_time_ms": item.get("sql_execution_time_ms"),
                "visualization_time_ms": item.get("visualization_time_ms"),
                "explanation_time_ms": item.get("explanation_time_ms")
            }
            
            # Add error if present
            if item.get("error_message"):
                history_item["error"] = item.get("error_message")
            
            # Add data and columns if available
            if item.get("data") is not None:
                try:
                    if hasattr(item.get("data"), 'to_dict'):
                        history_item["data"] = item.get("data").to_dict(orient='records')
                    elif isinstance(item.get("data"), bytes):
                        # Convert bytes to base64 string
                        import base64
                        history_item["data"] = base64.b64encode(item.get("data")).decode('utf-8')
                    elif isinstance(item.get("data"), str) and item.get("data").startswith('b\''):
                        # Already a string representation of bytes
                        history_item["data"] = item.get("data")
                    else:
                        # Try to convert any other type
                        history_item["data"] = str(item.get("data"))
                except Exception as e:
                    print(f"Error converting data: {e}")
                    history_item["data"] = str(item.get("data"))
                
                # Handle columns
                if hasattr(item.get("data"), 'columns'):
                    history_item["columns"] = list(item.get("data").columns)
                elif item.get("columns"):
                    history_item["columns"] = item.get("columns")
                else:
                    history_item["columns"] = []
            
            # Add visualization if available
            if item.get("visualization") is not None:
                try:
                    if hasattr(item.get("visualization"), 'to_json'):
                        history_item["visualization"] = item.get("visualization").to_json()
                    elif isinstance(item.get("visualization"), bytes):
                        # Convert bytes to base64 string
                        import base64
                        history_item["visualization"] = base64.b64encode(item.get("visualization")).decode('utf-8')
                    elif not isinstance(item.get("visualization"), str):
                        history_item['visualization'] = json.dumps(str(item['visualization']))
                    else:
                        history_item["visualization"] = item['visualization']
                except Exception as e:
                    print(f"Error converting visualization: {e}")
                    history_item["visualization"] = None
            
            # Add summary if available
            if item.get("summary"):
                history_item["summary"] = item.get("summary")
                
            enhanced_history.append(history_item)
            
        print(f"DEBUG: Final enhanced history size: {len(enhanced_history)} items")
        return jsonify({"status": "success", "history": enhanced_history})
    except Exception as e:
        print(f"Error getting history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# Analyze query patterns
@app.route('/analyze', methods=['GET'])
def analyze_patterns():
    try:
        analysis = Talk2SQL.analyze_error_patterns()
        return jsonify({"status": "success", "analysis": analysis})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Get example format
@app.route('/training_example_format', methods=['GET'])
def get_training_example_format():
    example_format = {
        "natural_language": "How many teams are in the NBA?",
        "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
        "type": "counting"  # Optional field
    }
    
    instructions = """
    To create your own training data:
    1. Create a JSON file with an array of examples
    2. Each example should have 'natural_language' and 'sql' fields
    3. The 'type' field is optional and can be used for categorization
    4. Upload the file to add examples to the system
    """
    
    return jsonify({
        "status": "success", 
        "example_format": example_format,
        "instructions": instructions
    })

# Voice-related endpoints and functions

# Record audio from the user's microphone
@app.route('/record_audio', methods=['POST'])
def record_audio():
    try:
        # Get recording parameters
        duration = request.json.get('duration', 10)  # Default 10 seconds
        fs = 44100  # Sample rate
        
        # Record audio
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        # Save to temporary file - use the same path for consistency
        temp_file = os.path.join(AUDIO_FOLDER, 'temp_audio.wav')
        sf.write(temp_file, recording, fs)
        
        return jsonify({
            "status": "success",
            "audio_path": "/audio_cache/temp_audio.wav",
            "duration": duration
        })
    except Exception as e:
        print(f"Error recording audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# Upload audio file for transcription
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Save the audio file temporarily
    audio_path = os.path.join(AUDIO_FOLDER, 'temp_audio.wav')
    audio_file.save(audio_path)
    
    try:
        # Ensure we're using a thread-safe connection
        get_thread_safe_connection()
        
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        if not transcription:
            return jsonify({"error": "Failed to transcribe audio"}), 500
        
        # Return the transcription
        return jsonify({
            "transcription": transcription,
            "status": "success"
        })
    except Exception as e:
        print(f"Error in upload_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Transcribe audio using Groq's Whisper API
@app.route('/transcribe', methods=['POST', 'GET'])
def transcribe_endpoint():
    # For POST requests, get audio_path from JSON body
    # For GET requests, get it from URL parameters
    if request.method == 'POST':
        audio_path = request.json.get('audio_path')
    else:
        audio_path = request.args.get('audio_path')
    
    if not audio_path or not os.path.exists(audio_path):
        # Convert web path to file system path if needed
        if audio_path and audio_path.startswith('/audio_cache'):
            converted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_path[1:])
            if os.path.exists(converted_path):
                audio_path = converted_path
            else:
                return jsonify({"status": "error", "message": f"Invalid audio path: {audio_path} (converted to {converted_path})"})
        else:
            return jsonify({"status": "error", "message": f"Invalid audio path: {audio_path}"})
    
    try:
        # Get transcription
        text = transcribe_audio(audio_path)
        
        if not text:
            return jsonify({"status": "error", "message": "Failed to transcribe audio"})
            
        print(f"Transcription: {text}")
        
        return jsonify({
            "status": "success", 
            "text": text
        })
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Standalone transcription function that can be called from other endpoints
def transcribe_audio(audio_path):
    """
    Transcribe audio using Groq's Whisper model.
    This function is thread-safe and can be called from any thread.
    
    Args:
        audio_path: Path to the audio file to transcribe
        
    Returns:
        Transcription text or None if transcription failed
    """
    # Handle relative paths - if path starts with /audio_cache
    if audio_path and audio_path.startswith('/audio_cache'):
        audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_path[1:])
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Invalid audio path: {audio_path}")
        return None
    
    try:
        # Open the audio file
        with open(audio_path, "rb") as file:
            # Use Groq's speech-to-text
            transcription = groq_client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                language="en"
            )
        
        return transcription.text
    except Exception as e:
        print(f"Error in transcribe_audio function: {e}")
        import traceback
        traceback.print_exc()
        return None

# Text-to-speech using Groq
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text')
    voice = request.json.get('voice', 'Celeste-PlayAI')  # Default voice
    
    if not text:
        return jsonify({"status": "error", "message": "No text provided"})
    
    try:
        # Generate speech using Groq
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # Save to a temporary file
        output_path = os.path.join(AUDIO_FOLDER, f"speech_{int(time.time())}.wav")
        response.write_to_file(output_path)
        
        # Convert to base64 for web playback
        with open(output_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "audio_path": output_path,
            "audio_base64": encoded_audio
        })
    except Exception as e:
        print(f"Error generating speech: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Voice SQL assistant endpoint
@app.route('/voice_assistant', methods=['POST'])
def voice_assistant():
    audio_path = request.json.get('audio_path')
    voice = request.json.get('voice', 'Celeste-PlayAI')  # Get selected voice
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"status": "error", "message": "Invalid audio path"})
    
    try:
        # Step 1: Transcribe audio using our thread-safe function
        question = transcribe_audio(audio_path)
        
        if not question:
            return jsonify({"status": "error", "message": "Failed to transcribe audio"})
            
        print(f"Transcribed question: {question}")
        
        # Step 2: Generate SQL and execute query using thread-safe connection
        try:
            # Define a thread-safe run_sql function for this request
            def thread_safe_run_sql(sql_query):
                conn = get_thread_safe_connection()
                return pd.read_sql_query(sql_query, conn)
            
            # Temporarily replace the run_sql function
            original_run_sql = Talk2SQL.run_sql
            Talk2SQL.run_sql = thread_safe_run_sql
            
            # Now run the query with the thread-safe function
            result = Talk2SQL.smart_query(question, print_results=False, visualize=True)
            
            # Restore original function
            Talk2SQL.run_sql = original_run_sql
        except Exception as e:
            print(f"Error in thread-safe SQL execution: {e}")
            import traceback
            traceback.print_exc()
            error_message = f"I couldn't answer that. {str(e)}"
            
            # Generate speech for error message
            speech_response = groq_client.audio.speech.create(
                model="playai-tts",
                voice=voice,  # Use the selected voice
                input=error_message,
                response_format="wav"
            )
            
            output_path = os.path.join(AUDIO_FOLDER, f"error_speech_{int(time.time())}.wav")
            speech_response.write_to_file(output_path)
            
            with open(output_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
            return jsonify({
                "status": "error",
                "question": question,
                "error": str(e),
                "audio_path": output_path,
                "audio_base64": encoded_audio
            })
        
        if not result["success"]:
            error_message = f"I couldn't answer that. {result.get('error', 'There was an error processing your query.')}"
            # Generate speech for error message
            speech_response = groq_client.audio.speech.create(
                model="playai-tts",
                voice=voice,  # Use the selected voice
                input=error_message,
                response_format="wav"
            )
            
            output_path = os.path.join(AUDIO_FOLDER, f"error_speech_{int(time.time())}.wav")
            speech_response.write_to_file(output_path)
            
            with open(output_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
            return jsonify({
                "status": "error",
                "question": question,
                "error": result.get("error", "Unknown error"),
                "audio_path": output_path,
                "audio_base64": encoded_audio
            })
        
        # Step 3: Generate summary of the data
        summary = generate_data_summary(question, result)
        
        # Step 4: Prepare a vocal response
        response_text = f"For your question: {question}. {summary}"
        
        # Step 5: Convert to speech using the selected voice
        speech_response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,  # Use the selected voice
            input=response_text,
            response_format="wav"
        )
        
        output_path = os.path.join(AUDIO_FOLDER, f"response_speech_{int(time.time())}.wav")
        speech_response.write_to_file(output_path)
        
        # Convert to base64 for web playback
        with open(output_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Convert Plotly figure to JSON if it exists
        visualization_json = None
        if "visualization" in result and result["visualization"] is not None:
            try:
                visualization_json = result["visualization"].to_json()
            except Exception as e:
                print(f"Error converting visualization to JSON: {e}")
        
        return jsonify({
            "status": "success",
            "question": question,
            "sql": result["sql"],
            "data": result["data"].to_dict(orient='records'),
            "columns": list(result["data"].columns),
            "summary": summary,
            "has_visualization": visualization_json is not None,
            "visualization": visualization_json,
            "audio_path": output_path,
            "audio_base64": encoded_audio
        })
    except Exception as e:
        print(f"Error in voice assistant: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# Streaming voice assistant endpoint
@app.route('/voice_assistant_stream')
def voice_assistant_stream():
    audio_path = request.args.get('audio_path')
    voice = request.args.get('voice', 'Celeste-PlayAI')  # Get selected voice
    
    # Print debug info
    print(f"Received stream request with audio_path: {audio_path}")
    
    # Convert web path to file system path if needed
    if audio_path and audio_path.startswith('/audio_cache'):
        audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_path[1:])
        print(f"Converted path to: {audio_path}")
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"status": "error", "message": f"Invalid audio path: {audio_path}"}), 400
    
    def generate():
        try:
            # Set appropriate headers for Server-Sent Events
            yield "retry: 10000\n"
            
            # Step 1: Transcribe audio using our thread-safe function and yield result
            question = transcribe_audio(audio_path)
            
            if not question:
                error_data = json.dumps({"type": "error", "message": "Failed to transcribe audio"})
                yield f"data: {error_data}\n\n"
                return
                
            print(f"Transcribed question: {question}")
            transcription_data = json.dumps({"type": "transcription", "text": question})
            yield f"data: {transcription_data}\n\n"
            
            # Step 2: Generate SQL and execute query
            # Create a thread-local connection for this request
            try:
                # Define a thread-safe run_sql function for this request
                def thread_safe_run_sql(sql_query):
                    conn = get_thread_safe_connection()
                    return pd.read_sql_query(sql_query, conn)
                
                # Temporarily replace the run_sql function
                original_run_sql = Talk2SQL.run_sql
                Talk2SQL.run_sql = thread_safe_run_sql
                
                # Now run the query with the thread-safe function
                result = Talk2SQL.smart_query(question, print_results=False, visualize=True)
                
                # Restore original function
                Talk2SQL.run_sql = original_run_sql
            except Exception as e:
                print(f"Error in thread-safe SQL execution: {e}")
                import traceback
                traceback.print_exc()
                error_message = f"I couldn't answer that. {str(e)}"
                error_data = json.dumps({"type": "error", "message": error_message})
                yield f"data: {error_data}\n\n"
                return
            
            if not result["success"]:
                error_message = f"I couldn't answer that. {result.get('error', 'There was an error processing your query.')}"
                error_data = json.dumps({"type": "error", "message": error_message})
                yield f"data: {error_data}\n\n"
                return
            
            # Send SQL query to client
            sql_data = json.dumps({
                "type": "sql", 
                "sql": result["sql"],
                "used_memory": result.get("used_memory", False)
            })
            yield f"data: {sql_data}\n\n"
            
            # Send data to client
            data_obj = {
                "type": "data", 
                "data": result["data"].to_dict(orient='records') if hasattr(result["data"], 'to_dict') else [], 
                "columns": list(result["data"].columns) if hasattr(result["data"], 'columns') else []
            }
            
            # Ensure all values are JSON serializable
            def ensure_serializable(obj):
                if isinstance(obj, bytes):
                    import base64
                    return base64.b64encode(obj).decode('utf-8')
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            
            # Process data to ensure it's serializable
            serializable_data = []
            for record in data_obj["data"]:
                serializable_record = {}
                for key, value in record.items():
                    serializable_record[key] = ensure_serializable(value)
                serializable_data.append(serializable_record)
            
            data_obj["data"] = serializable_data
            data_json = json.dumps(data_obj)
            yield f"data: {data_json}\n\n"
            
            # Step 3: Generate visualization if available
            visualization_json = None
            if "visualization" in result and result["visualization"] is not None:
                try:
                    visualization_json = result["visualization"].to_json()
                    viz_data = json.dumps({"type": "visualization", "visualization": visualization_json})
                    yield f"data: {viz_data}\n\n"
                except Exception as e:
                    print(f"Error converting visualization to JSON: {e}")
            
            # Step 4: Generate summary
            summary = generate_data_summary(question, result)
            summary_data = json.dumps({"type": "summary", "summary": summary})
            yield f"data: {summary_data}\n\n"
            
            # Step 5: Generate speech using the selected voice
            response_text = f"For your question: {question}. {summary}"
            speech_response = groq_client.audio.speech.create(
                model="playai-tts",
                voice=voice,  # Use the selected voice
                input=response_text,
                response_format="wav"
            )
            
            output_path = os.path.join(AUDIO_FOLDER, f"response_speech_{int(time.time())}.wav")
            speech_response.write_to_file(output_path)
            
            # Convert to base64 for web playback
            with open(output_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
            audio_data = json.dumps({"type": "audio", "audio_base64": encoded_audio})
            yield f"data: {audio_data}\n\n"
            
        except Exception as e:
            print(f"Error in streaming voice assistant: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'
                   })

# Streaming text query endpoint
@app.route('/ask_stream')
def ask_stream():
    question = request.args.get('question')
    
    # Print debug info
    print(f"Received streaming query: {question}")
    
    if not question:
        return jsonify({"status": "error", "message": "No question provided"}), 400
    
    def generate():
        try:
            # Set appropriate headers for Server-Sent Events
            yield "retry: 10000\n"
            
            # Step 1: Acknowledge receipt of question
            question_data = json.dumps({"type": "question", "text": question})
            yield f"data: {question_data}\n\n"
            
            # Step 2: Generate SQL and execute query with thread-safe connection
            try:
                # Define a thread-safe run_sql function for this request
                def thread_safe_run_sql(sql_query):
                    conn = get_thread_safe_connection()
                    return pd.read_sql_query(sql_query, conn)
                
                # Temporarily replace the run_sql function
                original_run_sql = Talk2SQL.run_sql
                Talk2SQL.run_sql = thread_safe_run_sql
                
                # Now run the query with the thread-safe function
                result = Talk2SQL.smart_query(question, print_results=False, visualize=True)
                
                # Restore original function
                Talk2SQL.run_sql = original_run_sql
            except Exception as e:
                print(f"Error in thread-safe SQL execution: {e}")
                import traceback
                traceback.print_exc()
                error_message = f"I couldn't answer that. {str(e)}"
                error_data = json.dumps({"type": "error", "message": error_message})
                yield f"data: {error_data}\n\n"
                return
            
            if not result["success"]:
                error_message = f"I couldn't answer that. {result.get('error', 'There was an error processing your query.')}"
                error_data = json.dumps({"type": "error", "message": error_message})
                yield f"data: {error_data}\n\n"
                return
            
            # Send SQL query to client
            sql_data = json.dumps({
                "type": "sql", 
                "sql": result["sql"],
                "used_memory": result.get("used_memory", False)
            })
            yield f"data: {sql_data}\n\n"
            
            # Send data to client
            data_obj = {
                "type": "data", 
                "data": result["data"].to_dict(orient='records') if hasattr(result["data"], 'to_dict') else [], 
                "columns": list(result["data"].columns) if hasattr(result["data"], 'columns') else []
            }
            
            # Ensure all values are JSON serializable
            def ensure_serializable(obj):
                if isinstance(obj, bytes):
                    import base64
                    return base64.b64encode(obj).decode('utf-8')
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            
            # Process data to ensure it's serializable
            serializable_data = []
            for record in data_obj["data"]:
                serializable_record = {}
                for key, value in record.items():
                    serializable_record[key] = ensure_serializable(value)
                serializable_data.append(serializable_record)
            
            data_obj["data"] = serializable_data
            data_json = json.dumps(data_obj)
            yield f"data: {data_json}\n\n"
            
            # Step 3: Generate visualization if available
            visualization_json = None
            if "visualization" in result and result["visualization"] is not None:
                try:
                    visualization_json = result["visualization"].to_json()
                    viz_data = json.dumps({"type": "visualization", "visualization": visualization_json})
                    yield f"data: {viz_data}\n\n"
                except Exception as e:
                    print(f"Error converting visualization to JSON: {e}")
            
            # Step 4: Generate summary
            summary, explanation_time_ms = generate_data_summary(question, result)
            summary_data = json.dumps({
                "type": "summary", 
                "summary": summary,
                "explanation_time_ms": explanation_time_ms
            })
            yield f"data: {summary_data}\n\n"
            
            # Step 5: Generate follow-up questions
            try:
                followups = Talk2SQL.generate_follow_up_questions(
                    question=question,
                    sql=result["sql"],
                    result_info=summary,
                    n=3
                )
                
                followups_data = json.dumps({
                    "type": "followups",
                    "questions": followups
                })
                yield f"data: {followups_data}\n\n"
            except Exception as e:
                print(f"Error generating follow-up questions: {e}")
                
            # Send completion event
            complete_data = json.dumps({"type": "complete"})
            yield f"data: {complete_data}\n\n"
            
        except Exception as e:
            print(f"Error in streaming query: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'
                   })

# Get available voice options
@app.route('/available_voices', methods=['GET'])
def get_available_voices():
    # Groq playai-tts available voices
    english_voices = [
        "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI", 
        "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI", 
        "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI", 
        "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI", 
        "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
    ]
    
    # Groq playai-tts-arabic available voices (if needed)
    arabic_voices = [
        "Abla-PlayAI", "Bashir-PlayAI", "Daliya-PlayAI", "Essa-PlayAI"
    ]
    
    return jsonify({
        "status": "success",
        "english_voices": english_voices,
        "arabic_voices": arabic_voices
    })


# # Serve the HTML frontend (local testing)
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def index(path):
#     if path != "" and os.path.exists(os.path.join('local', path)):
#         return send_file(os.path.join('dist', path))
#     return send_file('local/index.html')

# Serve the HTML frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    if path != "" and os.path.exists(os.path.join('dist', path)):
        return send_file(os.path.join('dist', path))
    return send_file('dist/index.html')

# Export query history in various formats
@app.route('/export_history', methods=['GET'])
def export_history():
    """
    Export query history in JSON, CSV or full CSV (ZIP) format.
    Optional query parameter 'id' to export a specific query.
    """
    try:
        # Get export format and optional query ID
        export_format = request.args.get('format', 'json')
        query_id = request.args.get('id')
        
        # Get query history from SQLite database
        if query_id:
            # Fetch a specific query by ID
            history_to_export = []
            recent_history = Talk2SQL.get_query_history()
            for item in recent_history:
                if item.get('id') == query_id:
                    history_to_export = [item]
                    break
                    
            if not history_to_export:
                return jsonify({'status': 'error', 'message': f'Query with ID {query_id} not found'}), 404
        else:
            # Fetch all history
            history_to_export = Talk2SQL.get_query_history()
            
        if not history_to_export:
            return jsonify({'status': 'error', 'message': 'No query history found'}), 404
        
        # Process history to ensure it's serializable
        serializable_history = []
        for item in history_to_export:
            serializable_item = {
                'id': item.get('id', ''),
                'question': item.get('question', ''),
                'sql': item.get('sql', ''),
                'success': item.get('success', False),
                'error_message': item.get('error_message', ''),
                'retry_count': item.get('retry_count', 0),
                'timestamp': item.get('timestamp', ''),
                'summary': item.get('summary', ''),
                'used_memory': item.get('used_memory', False),
                # Add timing fields
                'total_time_ms': item.get('total_time_ms'),
                'sql_generation_time_ms': item.get('sql_generation_time_ms'),
                'sql_execution_time_ms': item.get('sql_execution_time_ms'),
                'visualization_time_ms': item.get('visualization_time_ms'),
                'explanation_time_ms': item.get('explanation_time_ms')
            }
            
            # Extract timing details if available
            if 'timing_details' in item and item['timing_details']:
                try:
                    if isinstance(item['timing_details'], dict):
                        serializable_item.update(item['timing_details'])
                    elif isinstance(item['timing_details'], str):
                        timing_details = json.loads(item['timing_details'])
                        serializable_item.update(timing_details)
                    elif isinstance(item['timing_details'], bytes):
                        # Handle bytes by decoding to string first
                        timing_details = json.loads(item['timing_details'].decode('utf-8'))
                        serializable_item.update(timing_details)
                except Exception as e:
                    print(f"Error processing timing details: {e}")
            
            # Convert DataFrame to records if it exists
            if 'data' in item and item['data'] is not None:
                try:
                    if hasattr(item['data'], 'to_dict'):
                        serializable_item['data'] = item['data'].to_dict(orient='records')
                        serializable_item['columns'] = item['data'].columns.tolist()
                    elif isinstance(item['data'], bytes):
                        # Convert bytes to base64 string for JSON serialization
                        import base64
                        serializable_item['data'] = base64.b64encode(item['data']).decode('utf-8')
                    else:
                        serializable_item['data'] = str(item['data'])
                except Exception as e:
                    print(f"Error converting DataFrame: {e}")
                    serializable_item['data'] = str(item['data'])
            
            # Convert visualization to serializable format if it exists
            if 'visualization' in item and item['visualization'] is not None:
                try:
                    if hasattr(item['visualization'], 'to_json'):
                        serializable_item['visualization'] = item['visualization'].to_json()
                    elif isinstance(item['visualization'], bytes):
                        # Convert bytes to base64 string
                        import base64
                        serializable_item['visualization'] = base64.b64encode(item['visualization']).decode('utf-8')
                    elif not isinstance(item['visualization'], str):
                        serializable_item['visualization'] = json.dumps(str(item['visualization']))
                    else:
                        serializable_item['visualization'] = item['visualization']
                except Exception as e:
                    print(f"Error converting visualization: {e}")
                    serializable_item['visualization'] = str(item['visualization'])
            
            serializable_history.append(serializable_item)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export as JSON
        if export_format == 'json':
            return jsonify(serializable_history)
        
        # Export as CSV (summary only with data in zip)
        elif export_format == 'csv':
            # For regular CSV, we'll create a zip with summary CSV and individual data files
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                # First create the summary CSV
                csv_data = io.StringIO()
                writer = csv.writer(csv_data)
                
                # Write headers - include all relevant columns including timing metrics
                headers = [
                    'ID', 'Timestamp', 'Question', 'SQL', 'Success', 'Error', 'Retry Count', 
                    'Total Time (ms)', 'SQL Generation (ms)', 'SQL Execution (ms)', 
                    'Visualization (ms)', 'Explanation (ms)', 'Row Count', 'Column Count', 'Used Memory', 'Summary'
                ]
                writer.writerow(headers)
                
                # Write data rows
                for item in serializable_history:
                    # Calculate row and column counts
                    row_count = 0
                    col_count = 0
                    
                    if 'data' in item and item['data']:
                        if isinstance(item['data'], list):
                            row_count = len(item['data'])
                        elif hasattr(item['data'], 'shape'):
                            row_count = item['data'].shape[0]
                            
                    if 'columns' in item and item['columns']:
                        col_count = len(item['columns'])
                    
                    row = [
                        item.get('id', ''),
                        item.get('timestamp', ''),
                        item.get('question', ''),
                        item.get('sql', ''),
                        item.get('success', False),
                        item.get('error_message', ''),
                        item.get('retry_count', 0),
                        item.get('total_time_ms', ''),
                        item.get('sql_generation_time_ms', ''),
                        item.get('sql_execution_time_ms', ''),
                        item.get('visualization_time_ms', ''),
                        item.get('explanation_time_ms', ''),
                        row_count,
                        col_count,
                        item.get('used_memory', False),
                        item.get('summary', '')
                    ]
                    writer.writerow(row)
                
                csv_data.seek(0)
                zip_file.writestr('query_history_summary.csv', csv_data.getvalue())
                
                # Now add individual CSV files for each query's data
                for item in serializable_history:
                    if 'data' in item and item['data'] and 'columns' in item and item['columns']:
                        try:
                            # Convert records to DataFrame if needed
                            if isinstance(item['data'], list) and len(item['data']) > 0 and isinstance(item['data'][0], dict):
                                df = pd.DataFrame(item['data'])
                            elif hasattr(item['data'], 'to_csv'):
                                df = item['data']
                            else:
                                # Skip if we can't convert to DataFrame
                                continue
                            
                            # Convert to CSV
                            query_csv = io.StringIO()
                            df.to_csv(query_csv, index=False)
                            query_csv.seek(0)
                            
                            # Add to zip with ID as filename
                            query_id = item.get('id', hashlib.md5(item.get('question', '').encode()).hexdigest()[:8])
                            zip_file.writestr(f"query_data_{query_id}.csv", query_csv.getvalue())
                        except Exception as e:
                            print(f"Error creating CSV for query {item.get('id')}: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Add a README file
                readme_content = """Query History Export

This ZIP file contains:
1. query_history_summary.csv - Summary of all queries including timing information
2. query_data_*.csv - Individual CSV files for each query's data results

Timing Information:
- Total Time (ms): Total query processing time
- SQL Generation (ms): Time taken to generate SQL from natural language
- SQL Execution (ms): Time taken to execute the SQL query
- Visualization (ms): Time taken to generate visualizations
- Explanation (ms): Time taken to generate summary explanations

Export generated on: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                zip_file.writestr('README.txt', readme_content)
            
            zip_buffer.seek(0)
            
            # Build the filename
            filename = f"query_history_{timestamp}.zip"
            if query_id:
                filename = f"query_{query_id}_{timestamp}.zip"
            
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )
        
        # Export as comprehensive ZIP with summary + data stats + individual query data
        elif export_format == 'full_csv':
            # Create a temporary zip file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                # Add the summary CSV
                csv_data = io.StringIO()
                writer = csv.writer(csv_data)
                
                # Write headers - include all relevant columns including timing metrics
                headers = [
                    'ID', 'Timestamp', 'Question', 'SQL', 'Success', 'Error', 'Retry Count', 
                    'Total Time (ms)', 'SQL Generation (ms)', 'SQL Execution (ms)', 
                    'Visualization (ms)', 'Explanation (ms)', 'Summary'
                ]
                writer.writerow(headers)
                
                # Write data rows
                for item in serializable_history:
                    row = [
                        item.get('id', ''),
                        item.get('timestamp', ''),
                        item.get('question', ''),
                        item.get('sql', ''),
                        item.get('success', False),
                        item.get('error_message', ''),
                        item.get('retry_count', 0),
                        item.get('total_time_ms', ''),
                        item.get('sql_generation_time_ms', ''),
                        item.get('sql_execution_time_ms', ''),
                        item.get('visualization_time_ms', ''),
                        item.get('explanation_time_ms', ''),
                        item.get('summary', '')
                    ]
                    writer.writerow(row)
                
                csv_data.seek(0)
                zip_file.writestr('query_history_summary.csv', csv_data.getvalue())
                
                # Create a data stats CSV that summarizes each query's result size
                data_stats_csv = io.StringIO()
                stats_writer = csv.writer(data_stats_csv)
                stats_writer.writerow(['ID', 'Question', 'Row Count', 'Column Count', 'Has Visualization', 'Total Time (ms)'])
                
                # Add individual CSV files for each query's data
                for item in serializable_history:
                    # Add to data stats
                    row_count = 0
                    col_count = 0
                    has_viz = False
                    
                    if 'data' in item and item['data']:
                        if isinstance(item['data'], list):
                            row_count = len(item['data'])
                        elif hasattr(item['data'], 'shape'):
                            row_count = item['data'].shape[0]
                            
                    if 'columns' in item and item['columns']:
                        col_count = len(item['columns'])
                        
                    if 'visualization' in item and item['visualization']:
                        has_viz = True
                        
                    stats_writer.writerow([
                        item.get('id', ''),
                        item.get('question', ''),
                        row_count,
                        col_count,
                        'Yes' if has_viz else 'No',
                        item.get('total_time_ms', '')
                    ])
                
                    # Process the actual data for individual CSV files
                    if 'data' in item and item['data'] and 'columns' in item and item['columns']:
                        try:
                            # Convert records to DataFrame if needed
                            if isinstance(item['data'], list) and len(item['data']) > 0 and isinstance(item['data'][0], dict):
                                df = pd.DataFrame(item['data'])
                            elif hasattr(item['data'], 'to_csv'):
                                df = item['data']
                            else:
                                # Skip if we can't convert to DataFrame
                                continue
                            
                            # Convert to CSV
                            query_csv = io.StringIO()
                            df.to_csv(query_csv, index=False)
                            query_csv.seek(0)
                            
                            # Add to zip with ID as filename
                            query_id = item.get('id', hashlib.md5(item.get('question', '').encode()).hexdigest()[:8])
                            zip_file.writestr(f"query_data_{query_id}.csv", query_csv.getvalue())
                        except Exception as e:
                            print(f"Error creating CSV for query {item.get('id')}: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Add the data stats CSV
                data_stats_csv.seek(0)
                zip_file.writestr('data_stats.csv', data_stats_csv.getvalue())
                
                # Add a README file
                readme_content = """Query History Export (Full)

This ZIP file contains:
1. query_history_summary.csv - Summary of all queries including timing information
2. data_stats.csv - Statistics about query results (row/column counts, visualization status)
3. query_data_*.csv - Individual CSV files for each query's data results

Timing Information:
- Total Time (ms): Total query processing time
- SQL Generation (ms): Time taken to generate SQL from natural language
- SQL Execution (ms): Time taken to execute the SQL query
- Visualization (ms): Time taken to generate visualizations
- Explanation (ms): Time taken to generate summary explanations

Export generated on: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                zip_file.writestr('README.txt', readme_content)
            
            zip_buffer.seek(0)
            
            # Build the filename
            filename = f"query_history_full_{timestamp}.zip"
            if query_id:
                filename = f"query_{query_id}_full_{timestamp}.zip"
            
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )
        
        else:
            return jsonify({'status': 'error', 'message': f'Invalid export format: {export_format}'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def _parse_iso(ts: str) -> datetime.datetime:
    """Robust ISO‑timestamp → datetime (fallback = now)."""
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return datetime.datetime.now()


def _percentile(values, q):
    """Safe percentile (returns 0 if data missing)."""
    try:
        return float(np.percentile(values, q)) if values else 0
    except Exception:
        return 0


def _mean(values):
    return float(statistics.mean(values)) if values else 0

# Get evaluation metrics from query history
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return aggregated observability metrics for the SQL‑LLM app."""
    try:
        # 1️⃣ Parameters
        time_range = request.args.get("time_range", "all")  # all | day | week | month
        limit = int(request.args.get("limit", 1000))

        # 2️⃣ Pull history (latest first)
        history = Talk2SQL.get_query_history(limit=limit)
        
        # Create a default empty response that the frontend can safely render
        default_response = {
            "status": "success",
            "total_queries": 0,
            "successful_queries": 0,
            "error_queries": 0,
            "success_rate": 0,
            "latency": {
                "p50_total_ms": 0,
                "p95_total_ms": 0,
                "stage_p95_ms": {
                    "generation_ms": 0,
                    "execution_ms": 0,
                    "visualization_ms": 0,
                    "explanation_ms": 0,
                },
                "mean_breakdown_pct": {
                    "generation_pct": 25,
                    "execution_pct": 25,
                    "visualization_pct": 25,
                    "explanation_pct": 25,
                },
            },
            "retry_metrics": {
                "queries_with_retry": 0,
                "total_retries": 0,
                "retry_rate_pct": 0,
                "retry_success_rate_pct": 0,
            },
            "memory_metrics": {
                "queries_with_memory": 0,
                "memory_usage_rate_pct": 0,
                "with_memory_success_rate_pct": 0,
                "without_memory_success_rate_pct": 0,
            },
            "top_errors": [],
            "time_series": {
                "dates": [],
                "counts": [],
                "success_counts": [],
                "success_rates": [],
                "retries": [],
            },
        }
        
        if not history:
            default_response["message"] = "No query history"
            return jsonify(default_response)

        # 3️⃣ Filter by time window
        if time_range != "all":
            now = datetime.datetime.now()
            delta = {"day": 1, "week": 7, "month": 30}[time_range]
            history = [item for item in history if (now - _parse_iso(item.get("timestamp", ""))).days <= delta]

        if not history:
            default_response["message"] = "No query history in selected time range"
            return jsonify(default_response)

        # Pre‑extract common lists for fast stats
        total_times = [item.get("total_time_ms", 0) for item in history if item.get("total_time_ms")]
        gen_times   = [item.get("sql_generation_time_ms", 0) for item in history if item.get("sql_generation_time_ms")]
        exec_times  = [item.get("sql_execution_time_ms", 0) for item in history if item.get("sql_execution_time_ms")]
        viz_times   = [item.get("visualization_time_ms", 0) for item in history if item.get("visualization_time_ms")]
        expl_times  = [item.get("explanation_time_ms", 0) for item in history if item.get("explanation_time_ms")]

        # 4️⃣ High‑level counts
        total_q        = len(history)
        success_q      = sum(1 for h in history if h.get("success"))
        error_q        = total_q - success_q
        success_rate   = (success_q / total_q) * 100 if total_q else 0

        # 5️⃣ Percentiles (p50 & p95)
        latency_p50 = _percentile(total_times, 50)
        latency_p95 = _percentile(total_times, 95)

        stage_p95 = {
            "generation_ms": _percentile(gen_times, 95),
            "execution_ms":  _percentile(exec_times, 95),
            "visualization_ms": _percentile(viz_times, 95),
            "explanation_ms":  _percentile(expl_times, 95),
        }

        # 6️⃣ Latency breakdown (mean share)
        mean_gen = _mean(gen_times)
        mean_exec = _mean(exec_times)
        mean_viz = _mean(viz_times)
        mean_expl = _mean(expl_times)
        mean_total = _mean(total_times)
        breakdown_share = {
            "generation_pct": (mean_gen / mean_total) * 100 if mean_total else 25,
            "execution_pct":  (mean_exec / mean_total) * 100 if mean_total else 25,
            "visualization_pct": (mean_viz / mean_total) * 100 if mean_total else 25,
            "explanation_pct":  (mean_expl / mean_total) * 100 if mean_total else 25,
        }

        # 7️⃣ Retry insights
        retry_counts = [h.get("retry_count", 0) for h in history]
        queries_with_retry = sum(1 for r in retry_counts if r > 0)
        total_retries = sum(retry_counts)
        retry_rate   = (queries_with_retry / total_q) * 100 if total_q else 0
        retry_success = sum(1 for h in history if h.get("retry_count", 0) > 0 and h.get("success"))
        retry_success_rate = (retry_success / queries_with_retry) * 100 if queries_with_retry else 0

        # 8️⃣ Memory usage stats
        uses_mem = [h for h in history if h.get("used_memory")]
        mem_used_q = len(uses_mem)
        mem_success_q = sum(1 for h in uses_mem if h.get("success"))
        mem_success_rate = (mem_success_q / mem_used_q) * 100 if mem_used_q else 0
        no_mem_q = total_q - mem_used_q
        no_mem_success_rate = (success_q - mem_success_q) / no_mem_q * 100 if no_mem_q else 0

        # 9️⃣ Error taxonomy (top 5)
        err_bucket = Counter()
        for h in history:
            if h.get("success"):
                continue
            msg = (h.get("error_message") or "").lower()
            if "syntax" in msg:
                err_bucket["syntax_error"] += 1
            elif "timeout" in msg:
                err_bucket["timeout"] += 1
            elif any(t in msg for t in ["permission", "access"]):
                err_bucket["permission"] += 1
            elif "connection" in msg:
                err_bucket["connection"] += 1
            elif "not exist" in msg or "not found" in msg:
                if "table" in msg:
                    err_bucket["table_not_found"] += 1
                elif "column" in msg:
                    err_bucket["column_not_found"] += 1
            else:
                err_bucket["other"] += 1
        top_errors = err_bucket.most_common(5)

        # 🔟 Time‑series (per‑day)
        ts_counts = defaultdict(lambda: {"total": 0, "success": 0, "retry": 0})
        for h in history:
            d = _parse_iso(h.get("timestamp", "")).date().isoformat()
            ts_counts[d]["total"] += 1
            if h.get("success"):
                ts_counts[d]["success"] += 1
            if h.get("retry_count", 0) > 0:
                ts_counts[d]["retry"] += 1
        ts_sorted = sorted(ts_counts.items())
        dates             = [d for d, _ in ts_sorted]
        daily_total       = [v["total"] for _, v in ts_sorted]
        daily_success     = [v["success"] for _, v in ts_sorted]
        daily_retry       = [v["retry"] for _, v in ts_sorted]
        daily_success_pct = [(s / t) * 100 if t else 0 for s, t in zip(daily_success, daily_total)]

        # 1️⃣1️⃣ Package response
        response = {
            "total_queries": total_q,
            "successful_queries": success_q,
            "error_queries": error_q,
            "success_rate": success_rate,

            "latency": {
                "p50_total_ms": latency_p50,
                "p95_total_ms": latency_p95,
                "stage_p95_ms": stage_p95,
                "mean_breakdown_pct": breakdown_share,
            },

            "retry_metrics": {
                "queries_with_retry": queries_with_retry,
                "total_retries": total_retries,
                "retry_rate_pct": retry_rate,
                "retry_success_rate_pct": retry_success_rate,
            },

            "memory_metrics": {
                "queries_with_memory": mem_used_q,
                "memory_usage_rate_pct": (mem_used_q / total_q) * 100 if total_q else 0,
                "with_memory_success_rate_pct": mem_success_rate,
                "without_memory_success_rate_pct": no_mem_success_rate,
            },

            "top_errors": [{"type": k, "count": v} for k, v in top_errors],

            "time_series": {
                "dates": dates,
                "counts": daily_total,
                "success_counts": daily_success,
                "success_rates": daily_success_pct,
                "retries": daily_retry,
            },
        }

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


# @app.route('/starter_questions', methods=['GET'])
# def get_starter_questions():
#     """
#     Get starter questions for the database.
#     Uses the connected database to generate relevant example questions.
#     """
#     try:
#         # Get database path from query parameter, fallback to the global connection
#         db_path = request.args.get('db_path')
        
#         # Detailed debug logs
#         app.logger.info(f"[STARTER_QUESTIONS] Request received with db_path: {db_path}")
#         app.logger.info(f"[STARTER_QUESTIONS] All request args: {dict(request.args)}")
        
#         if not db_path:
#             app.logger.error("[STARTER_QUESTIONS] No database path provided")
#             return jsonify({"status": "error", "message": "Database path is required"}), 400
        
#         # Get database name from path for tailoring questions
#         db_name = os.path.basename(db_path).lower()
#         app.logger.info(f"[STARTER_QUESTIONS] Database name: {db_name}")
        
#         # Create a temporary connection to get schema info
#         schema_info = ""
#         temp_conn = None
        
#         try:
#             # Connect to the database to get schema
#             app.logger.info(f"[STARTER_QUESTIONS] Connecting to database: {db_path}")
#             temp_conn = sqlite3.connect(db_path)
#             cursor = temp_conn.cursor()
            
#             # Get table list
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
#             tables = cursor.fetchall()
#             table_names = [table[0] for table in tables]
            
#             app.logger.info(f"[STARTER_QUESTIONS] Found tables: {table_names}")
            
#             # Get schema details for each table
#             for table_name in table_names:
#                 schema_info += f"Table: {table_name}\n"
#                 cursor.execute(f"PRAGMA table_info({table_name})")
#                 columns = cursor.fetchall()
#                 for col in columns:
#                     schema_info += f"  - {col[1]} ({col[2]})\n"
#                 schema_info += "\n"
                
#                 # For the first few tables, also get sample data count
#                 if tables.index((table_name,)) < 3:
#                     try:
#                         cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
#                         count = cursor.fetchone()[0]
#                         schema_info += f"  Count: {count} rows\n\n"
#                     except:
#                         pass
#         except Exception as e:
#             app.logger.error(f"[STARTER_QUESTIONS] Error getting schema: {str(e)}")
#             schema_info = "Error getting schema"
#         finally:
#             # Close the temporary connection
#             if temp_conn:
#                 temp_conn.close()
#                 app.logger.info("[STARTER_QUESTIONS] Temporary connection closed")
        
#         # Generate database-specific questions
#         count = request.args.get('count', default=10, type=int)
#         questions = []
        
#         # STEP 1: Generate database-specific questions based on name
#         if 'pokemon' in db_name:
#             app.logger.info("[STARTER_QUESTIONS] Using pokemon-specific questions")
#             questions = [
#                 "What are the names and Pokedex numbers of all Pokémon classified as 'Starter'?",
#                 "How many Pokémon are there in each generation?",
#                 "Which Pokémon have the highest attack stats?",
#                 "List all Legendary Pokémon and their types",
#                 "What is the distribution of Pokémon by type?",
#                 "Which Pokémon can evolve and what are their evolutions?",
#                 "What are the top 10 Pokémon by total base stats?",
#                 "How many Pokémon have dual types?",
#                 "What is the average height and weight by type?",
#                 "Which Pokémon have unique type combinations?"
#             ]
#         elif 'nba' in db_name:
#             app.logger.info("[STARTER_QUESTIONS] Using NBA-specific questions")
#             questions = [
#                 "Who are the top 10 scorers in the NBA?",
#                 "What teams have won the most championships?",
#                 "Which players have the highest field goal percentage?",
#                 "What was the average points per game last season?",
#                 "Who are the tallest players in the league?",
#                 "Which teams had the most wins in the last season?",
#                 "Who are the top 5 players with the most assists?",
#                 "What is the distribution of player heights across different positions?",
#                 "Which players have scored the most 3-pointers?",
#                 "What is the correlation between player salary and performance stats?"
#             ]
#         elif 'fifa' in db_name:
#             app.logger.info("[STARTER_QUESTIONS] Using FIFA-specific questions")
#             questions = [
#                 "Who are the top-rated players in the game?",
#                 "Which teams have the highest overall ratings?",
#                 "Who are the fastest players in the game?",
#                 "What is the distribution of player nationalities?",
#                 "Which players have the highest potential growth?",
#                 "Who are the best goalkeepers in the game?",
#                 "What is the average age of players by league?",
#                 "Which players have the best shooting stats?",
#                 "What positions have the highest average wages?",
#                 "Who are the best young players under 21 years old?"
#             ]
#         else:
#             # STEP 2: For other databases, generate schema-specific questions
#             app.logger.info("[STARTER_QUESTIONS] Generating schema-based questions")
            
#             # Default generic questions that work for any database
#             generic_questions = [
#                 "What tables are in this database?",
#                 "Show me the first 10 records from the main table",
#                 "What is the count of records in each table?",
#                 "How many unique values are in each column?",
#                 "What are the minimum and maximum values for numeric columns?"
#             ]
            
#             # Add table-specific questions if we have schema info
#             table_specific_questions = []
            
#             if table_names:
#                 # Extract table names from schema
#                 for table_name in table_names[:3]:  # Use first 3 tables for questions
#                     table_specific_questions.append(f"Show me all columns from the {table_name} table")
#                     table_specific_questions.append(f"How many records are in the {table_name} table?")
                    
#                     # Get column names for this table to ask more specific questions
#                     try:
#                         cursor = temp_conn.cursor() if temp_conn else None
#                         if cursor:
#                             cursor.execute(f"PRAGMA table_info({table_name})")
#                             columns = cursor.fetchall()
#                             column_names = [col[1] for col in columns]
                            
#                             # Add column-specific questions
#                             if len(column_names) >= 2:
#                                 col1, col2 = column_names[0], column_names[1]
#                                 table_specific_questions.append(f"What are the most common values in the {col1} column of the {table_name} table?")
#                                 table_specific_questions.append(f"Show me the relationship between {col1} and {col2} in the {table_name} table")
#                     except:
#                         pass
                
#                 # Add relationship questions if we have multiple tables
#                 if len(table_names) >= 2:
#                     table_specific_questions.append(f"Show me any relationships between {table_names[0]} and {table_names[1]}")
                
#                 # Combine questions, prioritizing table-specific ones
#                 questions = table_specific_questions + generic_questions
#             else:
#                 questions = generic_questions
        
#         # Limit to requested count and ensure uniqueness
#         questions = list(dict.fromkeys(questions))[:count]
        
#         app.logger.info(f"[STARTER_QUESTIONS] Final questions to return: {questions}")
        
#         return jsonify({
#             "status": "success", 
#             "questions": questions,
#             "debug_info": {
#                 "db_path": db_path,
#                 "db_name": db_name,
#                 "schema_length": len(schema_info),
#                 "tables_found": table_names if 'table_names' in locals() else [],
#                 "questions_count": len(questions)
#             }
#         })
        
#     except Exception as e:
#         app.logger.error(f"[STARTER_QUESTIONS] General error in endpoint: {str(e)}")
#         traceback_str = traceback.format_exc()
#         app.logger.error(f"[STARTER_QUESTIONS] Traceback: {traceback_str}")
        
#         return jsonify({
#             "status": "error", 
#             "message": str(e),
#             "traceback": traceback_str,
#             "debug_info": {
#                 "error": str(e),
#                 "db_path": request.args.get('db_path', "not provided")
#             }
#         }), 500

@app.route('/starter_questions', methods=['GET'])
def get_starter_questions():
    """
    Get starter questions for the database.
    Uses the connected database to generate relevant example questions.
    """
    try:
        # Get database path from query parameter, fallback to the global connection
        db_path = request.args.get('db_path')
        
        # Detailed debug logs
        app.logger.info(f"[STARTER_QUESTIONS] Request received with db_path: {db_path}")
        app.logger.info(f"[STARTER_QUESTIONS] All request args: {dict(request.args)}")
        
        if not db_path:
            app.logger.error("[STARTER_QUESTIONS] No database path provided")
            return jsonify({"status": "error", "message": "Database path is required"}), 400
        
        # Get database name from path for tailoring questions
        db_name = os.path.basename(db_path).lower()
        app.logger.info(f"[STARTER_QUESTIONS] Database name: {db_name}")
        
        # Create a temporary connection to get schema info
        schema_info = ""
        temp_conn = None
        
        try:
            # Connect to the database to get schema
            app.logger.info(f"[STARTER_QUESTIONS] Connecting to database: {db_path}")
            temp_conn = sqlite3.connect(db_path)
            cursor = temp_conn.cursor()
            
            # Get table list
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            app.logger.info(f"[STARTER_QUESTIONS] Found tables: {table_names}")
            
            # Get schema details for each table
            for table_name in table_names:
                schema_info += f"Table: {table_name}\n"
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                for col in columns:
                    schema_info += f"  - {col[1]} ({col[2]})\n"
                schema_info += "\n"
                
                # For the first few tables, also get sample data count
                if tables.index((table_name,)) < 3:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        schema_info += f"  Count: {count} rows\n\n"
                    except:
                        pass
        except Exception as e:
            app.logger.error(f"[STARTER_QUESTIONS] Error getting schema: {str(e)}")
            schema_info = "Error getting schema"
        finally:
            # Close the temporary connection
            if temp_conn:
                temp_conn.close()
                app.logger.info("[STARTER_QUESTIONS] Temporary connection closed")
        
        # Get the requested number of questions
        count = request.args.get('count', default=10, type=int)
        
        # Use Talk2SQL.generate_starter_questions to generate questions
        app.logger.info("[STARTER_QUESTIONS] Using Talk2SQL to generate schema-based questions")
        question_objects = Talk2SQL.generate_starter_questions(schema_info, count)
        
        # Extract just the questions from the returned objects
        questions = [q['question'] for q in question_objects]
        
        app.logger.info(f"[STARTER_QUESTIONS] Final questions to return: {questions}")
        
        return jsonify({
            "status": "success", 
            "questions": questions,
            "debug_info": {
                "db_path": db_path,
                "db_name": db_name,
                "schema_length": len(schema_info),
                "tables_found": table_names if 'table_names' in locals() else [],
                "questions_count": len(questions)
            }
        })
        
    except Exception as e:
        app.logger.error(f"[STARTER_QUESTIONS] General error in endpoint: {str(e)}")
        traceback_str = traceback.format_exc()
        app.logger.error(f"[STARTER_QUESTIONS] Traceback: {traceback_str}")
        
        return jsonify({
            "status": "error", 
            "message": str(e),
            "traceback": traceback_str,
            "debug_info": {
                "error": str(e),
                "db_path": request.args.get('db_path', "not provided")
            }
        }), 500

@app.route('/add_documentation', methods=['POST'])
def add_documentation():
    """
    Add documentation to the connected SQLite database for improved context in queries.
    The documentation will be stored in the vector database for retrieval during queries.
    """
    try:
        if not Talk2SQL:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        data = request.json
        if not data or 'documentation' not in data:
            return jsonify({"success": False, "error": "Documentation text is required"}), 400
            
        documentation = data['documentation']
        doc_id = Talk2SQL.add_documentation(documentation)
        
        return jsonify({
            "success": True,
            "message": "Documentation added successfully",
            "id": doc_id
        })
    except Exception as e:
        app.logger.error(f"Error adding documentation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/training_data', methods=['GET', 'POST', 'DELETE'])
def manage_training_data():
    """
    Manage training data for the Talk2SQL system.
    GET: Retrieve all training data
    POST: Add new training example
    DELETE: Remove training example by ID
    """
    try:
        if not Talk2SQL:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        # GET - Retrieve all training data
        if request.method == 'GET':
            all_data = Talk2SQL.get_all_training_data()
            return jsonify({
                "success": True,
                "data": all_data.to_dict(orient='records')
            })
            
        # POST - Add new training example
        elif request.method == 'POST':
            data = request.json
            if not data:
                return jsonify({"success": False, "error": "Invalid request data"}), 400
                
            # Add question-SQL pair
            if 'question' in data and 'sql' in data:
                example_id = Talk2SQL.add_question_sql(data['question'], data['sql'])
                return jsonify({
                    "success": True,
                    "message": "Training example added successfully",
                    "id": example_id
                })
            # Add schema information
            elif 'schema' in data:
                schema_id = Talk2SQL.add_schema(data['schema'])
                return jsonify({
                    "success": True, 
                    "message": "Schema added successfully",
                    "id": schema_id
                })
            # Add documentation
            elif 'documentation' in data:
                doc_id = Talk2SQL.add_documentation(data['documentation'])
                return jsonify({
                    "success": True,
                    "message": "Documentation added successfully",
                    "id": doc_id
                })
            else:
                return jsonify({"success": False, "error": "Missing required fields"}), 400
                
        # DELETE - Remove training example
        elif request.method == 'DELETE':
            data = request.json
            if not data or 'id' not in data:
                return jsonify({"success": False, "error": "ID is required"}), 400
                
            success = Talk2SQL.remove_training_data(data['id'])
            if success:
                return jsonify({
                    "success": True,
                    "message": "Training data removed successfully"
                })
            else:
                return jsonify({"success": False, "error": "Failed to remove training data"}), 500
    except Exception as e:
        app.logger.error(f"Error managing training data: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/schema_visualization', methods=['GET'])
def get_schema_visualization():
    """
    Generate a visualization of the database schema in a tree-like format.
    This provides a quick overview of tables, columns, and relationships.
    """
    try:
        if not Talk2SQL or not Talk2SQL.conn:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        # Get database schema
        schema_info = get_db_schema()
        
        # Format the schema as a tree structure for visualization
        schema_tree = format_schema_as_tree(schema_info['tables'])
        
        return jsonify({
            "success": True,
            "schema": schema_tree
        })
    except Exception as e:
        app.logger.error(f"Error generating schema visualization: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def format_schema_as_tree(tables):
    """
    Format database schema as a tree structure for visualization.
    
    Args:
        tables: Dictionary of tables and their columns
        
    Returns:
        Tree structure of the schema
    """
    tree = []
    
    for table_name, table_info in tables.items():
        table_node = {
            "id": f"table-{table_name}",
            "name": table_name,
            "type": "table",
            "children": []
        }
        
        # Add columns as children
        for column in table_info['columns']:
            column_node = {
                "id": f"column-{table_name}-{column['name']}",
                "name": column['name'],
                "type": "column",
                "dataType": column['type'],
                "isPrimaryKey": column['pk'] == 1,
                "isNullable": column['notnull'] == 0
            }
            table_node["children"].append(column_node)
            
        # Add foreign keys if available
        if 'foreign_keys' in table_info:
            for fk in table_info['foreign_keys']:
                for column_name, reference in fk.items():
                    # Find the column node
                    for column in table_node["children"]:
                        if column["name"] == column_name:
                            column["isForeignKey"] = True
                            column["references"] = reference
                            break
        
        tree.append(table_node)
    
    return tree

@app.route('/update_documentation', methods=['PUT'])
def update_documentation():
    """
    Update existing documentation by ID.
    """
    try:
        if not Talk2SQL:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        data = request.json
        if not data or 'id' not in data or 'documentation' not in data:
            return jsonify({"success": False, "error": "Documentation ID and text are required"}), 400
            
        doc_id = data['id']
        documentation = data['documentation']
        
        # First delete the existing documentation
        success = Talk2SQL.remove_training_data(doc_id)
        if not success:
            return jsonify({"success": False, "error": "Failed to update documentation: ID not found"}), 404
            
        # Then add the new documentation (will get a new ID)
        new_doc_id = Talk2SQL.add_documentation(documentation)
        
        return jsonify({
            "success": True,
            "message": "Documentation updated successfully",
            "old_id": doc_id,
            "new_id": new_doc_id
        })
    except Exception as e:
        app.logger.error(f"Error updating documentation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/bulk_training_data', methods=['POST'])
def bulk_manage_training_data():
    """
    Bulk operations for training data.
    - Upload multiple training examples at once
    - Reset entire collections
    """
    try:
        if not Talk2SQL:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        data = request.json
        if not data or 'operation' not in data:
            return jsonify({"success": False, "error": "Operation type is required"}), 400
            
        operation = data['operation']
        
        # Bulk upload operation
        if operation == 'bulk_upload':
            if 'examples' not in data or not isinstance(data['examples'], list):
                return jsonify({"success": False, "error": "Examples list is required for bulk upload"}), 400
                
            results = []
            for example in data['examples']:
                try:
                    if 'question' in example and 'sql' in example:
                        example_id = Talk2SQL.add_question_sql(example['question'], example['sql'])
                        results.append({"id": example_id, "type": "question", "success": True})
                    elif 'schema' in example:
                        schema_id = Talk2SQL.add_schema(example['schema'])
                        results.append({"id": schema_id, "type": "schema", "success": True})
                    elif 'documentation' in example:
                        doc_id = Talk2SQL.add_documentation(example['documentation'])
                        results.append({"id": doc_id, "type": "documentation", "success": True})
                    else:
                        results.append({"error": "Invalid example format", "success": False})
                except Exception as e:
                    results.append({"error": str(e), "success": False})
                    
            return jsonify({
                "success": True,
                "message": f"Processed {len(results)} examples",
                "results": results
            })
            
        # Reset collection operation
        elif operation == 'reset_collection':
            if 'collection_type' not in data:
                return jsonify({"success": False, "error": "Collection type is required for reset operation"}), 400
                
            collection_type = data['collection_type']
            if collection_type not in ['questions', 'schema', 'docs']:
                return jsonify({"success": False, "error": "Invalid collection type. Must be one of: questions, schema, docs"}), 400
                
            success = Talk2SQL.reset_collection(collection_type)
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Collection '{collection_type}' reset successfully"
                })
            else:
                return jsonify({"success": False, "error": f"Failed to reset collection '{collection_type}'"}), 500
                
        else:
            return jsonify({"success": False, "error": f"Unknown operation: {operation}"}), 400
            
    except Exception as e:
        app.logger.error(f"Error in bulk training data operation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/training_data_stats', methods=['GET'])
def get_training_data_stats():
    """
    Get statistics about the training data.
    """
    try:
        if not Talk2SQL:
            return jsonify({"success": False, "error": "No database connection"}), 400
            
        # Get all training data
        all_data = Talk2SQL.get_all_training_data()
        
        # Calculate statistics
        total_count = len(all_data)
        questions_count = len(all_data[all_data['type'] == 'question'])
        schema_count = len(all_data[all_data['type'] == 'schema'])
        docs_count = len(all_data[all_data['type'] == 'documentation'])
        
        return jsonify({
            "success": True,
            "stats": {
                "total_count": total_count,
                "questions_count": questions_count,
                "schema_count": schema_count,
                "docs_count": docs_count
            }
        })
    except Exception as e:
        app.logger.error(f"Error getting training data stats: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/debug_starter_questions', methods=['GET'])
def debug_starter_questions():
    """
    Diagnostic endpoint to check if the generate_starter_questions method is working.
    """
    try:
        db_path = request.args.get('db_path')
        schema = request.args.get('schema')
        
        app.logger.info(f"[DEBUG] Received debug request for starter questions. db_path: {db_path}, schema provided: {schema is not None}")
        
        # Create a response object to track the process
        response_data = {
            "status": "processing",
            "steps": [],
            "final_status": None,
            "final_questions": None
        }
        
        # Step 1: Check if Talk2SQL instance exists
        response_data["steps"].append({
            "step": "Check Talk2SQL instance",
            "status": "Talk2SQL exists" if Talk2SQL else "Talk2SQL does not exist"
        })
        
        # Step 2: Check if the method exists
        has_method = hasattr(Talk2SQL, 'generate_starter_questions') if Talk2SQL else False
        response_data["steps"].append({
            "step": "Check method existence",
            "status": "Method exists" if has_method else "Method does not exist"
        })
        
        # Try the different methods of generating questions
        if Talk2SQL and has_method:
            try:
                # Try using existing Talk2SQL
                app.logger.info("[DEBUG] Trying existing Talk2SQL instance")
                response_data["steps"].append({
                    "step": "Try existing Talk2SQL instance",
                    "status": "attempting"
                })
                
                # If schema was provided, use it, otherwise generate from the database
                if not schema and db_path:
                    # Get schema from database
                    schema_info = ""
                    try:
                        # Connect temporarily if needed
                        if not Talk2SQL.conn:
                            app.logger.info(f"[DEBUG] Connecting to {db_path}")
                            Talk2SQL.connect_to_sqlite(db_path)
                            
                        cursor = Talk2SQL.conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                        tables = cursor.fetchall()
                        
                        for table in tables:
                            table_name = table[0]
                            schema_info += f"Table: {table_name}\n"
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()
                            for col in columns:
                                schema_info += f"  - {col[1]} ({col[2]})\n"
                            schema_info += "\n"
                    except Exception as e:
                        app.logger.error(f"[DEBUG] Error getting schema: {str(e)}")
                        response_data["steps"].append({
                            "step": "Get schema",
                            "status": f"failed: {str(e)}"
                        })
                        schema_info = "Error getting schema"
                else:
                    schema_info = schema or "No schema provided"
                
                response_data["steps"].append({
                    "step": "Get schema",
                    "status": "success",
                    "schema_length": len(schema_info),
                    "schema_preview": schema_info[:100] + "..." if len(schema_info) > 100 else schema_info
                })
                
                # Now try to generate questions
                questions = Talk2SQL.generate_starter_questions(schema_info, 5)
                
                response_data["steps"].append({
                    "step": "Generate questions with Talk2SQL",
                    "status": "success", 
                    "questions": questions
                })
                
                response_data["final_status"] = "success"
                response_data["final_questions"] = questions
                
            except Exception as e:
                app.logger.error(f"[DEBUG] Error using existing Talk2SQL: {str(e)}")
                response_data["steps"].append({
                    "step": "Try existing Talk2SQL instance",
                    "status": f"failed: {str(e)}"
                })
        
        # Try with a new instance
        if not response_data["final_questions"]:
            try:
                app.logger.info("[DEBUG] Trying with new Talk2SQL instance")
                response_data["steps"].append({
                    "step": "Try new Talk2SQL instance",
                    "status": "attempting"
                })
                
                # Create a new instance
                from Talk2SQL.engine.Talk2SQL import Talk2SQL
                new_instance = Talk2SQL()
                
                # Check if the method exists on the new instance
                new_has_method = hasattr(new_instance, 'generate_starter_questions')
                response_data["steps"].append({
                    "step": "Check method on new instance",
                    "status": "Method exists" if new_has_method else "Method does not exist"
                })
                
                if new_has_method:
                    # If schema was provided, use it, otherwise use a simple test schema
                    if not schema:
                        schema = """
                        Table: users
                          - id (INTEGER)
                          - name (TEXT)
                          - email (TEXT)
                        
                        Table: orders
                          - id (INTEGER)
                          - user_id (INTEGER)
                          - amount (REAL)
                          - date (TEXT)
                        """
                    
                    # Try to generate questions with the new instance
                    new_questions = new_instance.generate_starter_questions(schema, 5)
                    
                    response_data["steps"].append({
                        "step": "Generate questions with new instance",
                        "status": "success",
                        "questions": new_questions
                    })
                    
                    if not response_data["final_status"] or response_data["final_status"] != "success":
                        response_data["final_status"] = "success"
                        response_data["final_questions"] = new_questions
                
            except Exception as e:
                app.logger.error(f"[DEBUG] Error with new Talk2SQL instance: {str(e)}")
                response_data["steps"].append({
                    "step": "Try new Talk2SQL instance",
                    "status": f"failed: {str(e)}"
                })
        
        # Final fallback: Try the direct implementation
        if not response_data["final_questions"]:
            try:
                app.logger.info("[DEBUG] Trying direct implementation")
                response_data["steps"].append({
                    "step": "Try direct implementation",
                    "status": "attempting"
                })
                
                # If no schema was provided, use a simple test schema
                if not schema:
                    schema = """
                    Table: users
                      - id (INTEGER)
                      - name (TEXT)
                      - email (TEXT)
                    
                    Table: orders
                      - id (INTEGER)
                      - user_id (INTEGER)
                      - amount (REAL)
                      - date (TEXT)
                    """
                
                # Extract table names
                table_names = re.findall(r"Table: (\w+)", schema)
                
                # Generate questions directly
                direct_questions = [
                    f"How many records are in the {table_names[0]} table?",
                    f"What are all the columns in the {table_names[0]} table?",
                ]
                
                if len(table_names) > 1:
                    direct_questions.append(f"Show me the relationship between {table_names[0]} and {table_names[1]}")
                    
                direct_questions.extend([
                    "What tables are in this database?",
                    "Show me the first 10 records from the main table"
                ])
                
                response_data["steps"].append({
                    "step": "Generate questions directly",
                    "status": "success",
                    "questions": direct_questions
                })
                
                if not response_data["final_status"] or response_data["final_status"] != "success":
                    response_data["final_status"] = "success"
                    response_data["final_questions"] = direct_questions
                    
            except Exception as e:
                app.logger.error(f"[DEBUG] Error with direct implementation: {str(e)}")
                response_data["steps"].append({
                    "step": "Try direct implementation",
                    "status": f"failed: {str(e)}"
                })
        
        # If we still don't have questions, provide a default set
        if not response_data["final_questions"]:
            response_data["final_status"] = "fallback"
            response_data["final_questions"] = [
                "What tables are in this database?",
                "Show me the first 10 records from the main table",
                "What is the count of records in each table?",
                "What are the most common values in the main columns?",
                "Show me any relationships between tables"
            ]
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"[DEBUG] General error in debug endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

#prod
if __name__ == '__main__':
    # Verify query history file location
    print("\n--- QUERY HISTORY VERIFICATION ---")
    query_history_path = getattr(Talk2SQL, "history_db_path", config.get("history_db_path", "unknown"))
    print(f"Using query history at: {query_history_path}")
    if os.path.exists(query_history_path):
        print(f"Verified file exists: {os.path.abspath(query_history_path)}")
        print(f"File size: {os.path.getsize(query_history_path)} bytes")
    else:
        print(f"Warning: Query history file doesn't exist yet, it will be created when needed.")
    
    if query_history_path.endswith('.sqlite'):
        print("File has correct .sqlite extension ✓")
    else:
        print(f"Warning: File does not have .sqlite extension: {query_history_path}")
    
    if query_history_path.startswith(DB_FOLDER):
        print("File is in the correct database folder ✓")
    else:
        print(f"Warning: File is not in the expected database folder: {DB_FOLDER}")
    print("--- END VERIFICATION ---\n")
    
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 

#local testing
# if __name__ == '__main__':
#     # port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host='localhost', port=8000) s