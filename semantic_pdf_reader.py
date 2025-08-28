"""Semantic PDF reader using ChromaDB and PyPDFLoader for better search capabilities."""

import os
# Disable ChromaDB telemetry before any imports to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SemanticPatriotManualReader:
    """Semantic search-enabled reader for the Jeep Patriot manual."""
    
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.documents = []
        
    def load_and_index_manual(self) -> bool:
        """Load PDF and create vector index."""
        try:
            logger.info("Loading PDF with PyPDFLoader...")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            self.documents = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(self.documents)} chunks")
            
            # Always create fresh vector store to ensure proper indexing
            logger.info("Creating ChromaDB index...")
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="patriot_manual"
            )
            logger.info("ChromaDB index created and persisted")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading and indexing manual: {e}")
            return False
    
    def semantic_search(self, query: str, k: int = 5) -> List[str]:
        """Perform semantic search on the manual."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Extract content from documents
            results = []
            for doc in docs:
                content = doc.page_content.strip()
                if content:
                    # Add page metadata if available
                    page_info = ""
                    if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                        page_info = f" (Page {doc.metadata['page'] + 1})"
                    
                    results.append(f"{content}{page_info}")
            
            logger.info(f"Found {len(results)} relevant sections for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict]:
        """Perform semantic search with relevance scores."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Perform similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold and format results
            results = []
            for doc, score in docs_with_scores:
                if score <= score_threshold:  # Lower scores are better in some implementations
                    content = doc.page_content.strip()
                    if content:
                        page_info = ""
                        if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                            page_info = f" (Page {doc.metadata['page'] + 1})"
                        
                        results.append({
                            'content': f"{content}{page_info}",
                            'score': score,
                            'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                        })
            
            logger.info(f"Found {len(results)} relevant sections above threshold for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during semantic search with scores: {e}")
            return []
    
