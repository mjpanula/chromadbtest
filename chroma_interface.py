"""
ChromaDB Interface with GPU Acceleration
Provides methods for storing and searching text paragraphs using ChromaDB
with GPU-accelerated embeddings via sentence-transformers.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional
import uuid


class ChromaDBInterface:
    """Interface for ChromaDB operations with GPU acceleration."""
    
    def __init__(self, collection_name: str = "text_paragraphs", 
                 persist_directory: str = "./chroma_data",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB interface with GPU-accelerated embeddings.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            model_name: Sentence-transformers model name for embeddings
        """
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the embedding model with GPU support
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Collection '{collection_name}' initialized with {self.collection.count()} documents")
    
    def add_paragraph(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a text paragraph to the database.
        
        Args:
            text: The text paragraph to store
            metadata: Optional metadata dictionary
            
        Returns:
            The ID of the added document
        """
        # Generate embedding using GPU
        embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        return doc_id
    
    def add_paragraphs(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add multiple text paragraphs to the database in batch.
        
        Args:
            texts: List of text paragraphs to store
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of IDs for the added documents
        """
        # Generate embeddings using GPU (batched for efficiency)
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False).tolist()
        
        # Generate unique IDs
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        return doc_ids
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar paragraphs using a query string.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing documents, distances, metadatas, and ids
        """
        # Generate query embedding using GPU
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
    def compare_text(self, text: str, n_results: int = 5) -> Dict:
        """
        Compare a text paragraph with existing ones in the database.
        
        Args:
            text: The text to compare
            n_results: Number of similar results to return
            
        Returns:
            Dictionary containing similar documents and their similarity scores
        """
        return self.search(text, n_results)
    
    def get_all_documents(self) -> Dict:
        """
        Retrieve all documents from the collection.
        
        Returns:
            Dictionary containing all documents with their metadata
        """
        results = self.collection.get()
        return {
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'ids': results['ids']
        }
    
    def delete_document(self, doc_id: str):
        """
        Delete a document by ID.
        
        Args:
            doc_id: The ID of the document to delete
        """
        self.collection.delete(ids=[doc_id])
    
    def count_documents(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def reset_collection(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection reset successfully")
