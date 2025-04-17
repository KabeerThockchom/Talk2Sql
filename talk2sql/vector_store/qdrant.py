import hashlib
import uuid
from typing import List, Tuple
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .base import Talk2SQLBase

class QdrantVectorStore(Talk2SQLBase):
    """Vector store implementation using Qdrant for semantic search and storage."""
    
    def __init__(self, config=None):
        """
        Initialize Qdrant vector store.
        
        Args:
            config: Configuration dictionary with options:
                - qdrant_url: Qdrant server URL
                - qdrant_api_key: API key for Qdrant cloud
                - embedding_size: Size of embedding vectors
                - questions_collection: Name for questions collection
                - schema_collection: Name for schema collection
                - docs_collection: Name for documentation collection
                - n_results: Number of results to return (default: 5)
        """
        super().__init__(config)
        
        # Get configuration values
        self.embedding_size = config.get("embedding_size", 1536)  # Default size for OpenAI embeddings
        self.n_results = config.get("n_results", 5)
        
        # Collection names
        self.questions_collection = config.get("questions_collection", "Talk2SQL_questions")
        self.schema_collection = config.get("schema_collection", "Talk2SQL_schema")
        self.docs_collection = config.get("docs_collection", "Talk2SQL_docs")
        
        # Initialize Qdrant client with cloud configuration
        # Default to Talk2SQL's Qdrant cloud instance if not specified
        default_url = "https://d960d7c1-5c26-4a91-8e7a-fb70954d24c1.eu-west-1-0.aws.cloud.qdrant.io:6333"
        default_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.NQSukS5fheodhJDs5AxgxpOxHJCG9ROszaR2Jr6o1BU"
        
        self.url = config.get("qdrant_url", default_url)
        self.api_key = config.get("qdrant_api_key", default_api_key)
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        
        # Only setup collections if this class is being used directly, not through inheritance
        if self.__class__.__name__ == "QdrantVectorStore":
            self._setup_collections()
        
    def _setup_collections(self):
        """Create collections if they don't exist."""
        # Questions collection
        if not self.client.collection_exists(self.questions_collection):
            self.client.create_collection(
                collection_name=self.questions_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            
        # Schema collection
        if not self.client.collection_exists(self.schema_collection):
            self.client.create_collection(
                collection_name=self.schema_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            
        # Documentation collection
        if not self.client.collection_exists(self.docs_collection):
            self.client.create_collection(
                collection_name=self.docs_collection,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
    
    def _generate_deterministic_id(self, content: str) -> str:
        """Generate a deterministic ID from content."""
        content_bytes = content.encode('utf-8')
        hash_hex = hashlib.sha256(content_bytes).hexdigest()
        namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
        content_uuid = str(uuid.uuid5(namespace, hash_hex))
        return content_uuid
    
    def add_question_sql(self, question: str, sql: str) -> str:
        """
        Add question-SQL pair to vector store.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            
        Returns:
            ID of the stored entry
        """
        # Create a composite representation
        content = f"Question: {question}\nSQL: {sql}"
        
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(content)
        
        # Get embedding
        embedding = self.generate_embedding(question)
        
        # Insert into questions collection
        self.client.upsert(
            collection_name=self.questions_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question": question,
                        "sql": sql
                    }
                )
            ]
        )
        
        return f"{point_id}-q"
    
    def add_schema(self, schema: str) -> str:
        """
        Add database schema to vector store.
        
        Args:
            schema: Database schema (DDL)
            
        Returns:
            ID of the stored entry
        """
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(schema)
        
        # Get embedding
        embedding = self.generate_embedding(schema)
        
        # Insert into schema collection
        self.client.upsert(
            collection_name=self.schema_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "schema": schema
                    }
                )
            ]
        )
        
        return f"{point_id}-s"
    
    def add_documentation(self, documentation: str) -> str:
        """
        Add documentation to vector store.
        
        Args:
            documentation: Documentation text
            
        Returns:
            ID of the stored entry
        """
        # Generate deterministic ID for deduplication
        point_id = self._generate_deterministic_id(documentation)
        
        # Get embedding
        embedding = self.generate_embedding(documentation)
        
        # Insert into documentation collection
        self.client.upsert(
            collection_name=self.docs_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "documentation": documentation
                    }
                )
            ]
        )
        
        return f"{point_id}-d"
    
    def get_similar_questions(self, question: str) -> list:
        """
        Get similar questions with their SQL from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of question-SQL pairs
        """
        embedding = self.generate_embedding(question)
        
        results = self.client.search(
            collection_name=self.questions_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload for point in results]
    
    def get_related_schema(self, question: str) -> list:
        """
        Get related schema information from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of schema strings
        """
        embedding = self.generate_embedding(question)
        
        results = self.client.search(
            collection_name=self.schema_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload["schema"] for point in results]
    
    def get_related_documentation(self, question: str) -> list:
        """
        Get related documentation from vector store.
        
        Args:
            question: Natural language question
            
        Returns:
            List of documentation strings
        """
        embedding = self.generate_embedding(question)
        
        results = self.client.search(
            collection_name=self.docs_collection,
            query_vector=embedding,
            limit=self.n_results
        )
        
        return [point.payload["documentation"] for point in results]
    
    def get_all_training_data(self) -> pd.DataFrame:
        """
        Get all training data from the questions collection.
        
        Returns:
            DataFrame with question, SQL, and ID
        """
        # Scroll through all points in the questions collection
        points = []
        offset = None
        limit = 100
        
        while True:
            # Get batch of points
            batch = self.client.scroll(
                collection_name=self.questions_collection,
                limit=limit,
                offset=offset
            )
            
            # Add to points list
            points.extend(batch[0])
            
            # Update offset for next batch or break if done
            offset = batch[1]
            if offset is None or len(batch[0]) < limit:
                break
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                "id": f"{point.id}-q",
                "question": point.payload.get("question", ""),
                "sql": point.payload.get("sql", "")
            }
            for point in points
        ])
        
        return df
    
    def remove_training_data(self, id: str) -> bool:
        """
        Remove training data entry by ID.
        
        Args:
            id: Entry ID (with -q suffix)
            
        Returns:
            Success flag
        """
        # Extract the pure ID (remove the -q suffix)
        if id.endswith("-q"):
            point_id = id[:-2]
        else:
            point_id = id
        
        try:
            # Delete point
            self.client.delete(
                collection_name=self.questions_collection,
                points_selector=[point_id]
            )
            return True
        except Exception as e:
            print(f"Error removing training data: {e}")
            return False
    
    def reset_collection(self, collection_type: str = "all") -> bool:
        """
        Reset (delete and recreate) a collection or all collections.
        
        Args:
            collection_type: Collection type ("questions", "schema", "docs", or "all")
            
        Returns:
            Success flag
        """
        try:
            # Determine collections to reset
            collections = []
            if collection_type == "questions" or collection_type == "all":
                collections.append(self.questions_collection)
            if collection_type == "schema" or collection_type == "all":
                collections.append(self.schema_collection)
            if collection_type == "docs" or collection_type == "all":
                collections.append(self.docs_collection)
            
            # Reset each collection
            for collection in collections:
                if self.client.collection_exists(collection):
                    self.client.delete_collection(collection)
            
            # Re-create collections
            self._setup_collections()
            
            return True
        except Exception as e:
            print(f"Error resetting collections: {e}")
            return False