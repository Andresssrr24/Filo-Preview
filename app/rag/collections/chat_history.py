from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import HierarchicalNodeParser, SemanticSplitterNodeParser 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from app import settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

vecstore_global_settings = settings.VECSTORE_GLOBAL_SETTINGS
chat_collection_settings = settings.CHAT_HISTORY_COLLECTION_SETTINGS

class ChatVecstore:
    def __init__(self):
        self.ef = vecstore_global_settings["embedding_model"]
        self.chat_memory = []
        self.vecstore = None

    def create_get_collection(self):
        persist_dir = vecstore_global_settings["persist_directory"]
        collection_name = chat_collection_settings["collection_name"]
        collection_metadata = chat_collection_settings["collection_metadata"]

        try:
            self.vecstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.ef,
                collection_name=collection_name,
                collection_metadata=collection_metadata
            )
            return self.vecstore
        
        except Exception as e:
            logger.error(f"Error while getting chat history collection: {e}")
            raise

    def add_conversation_turn(self, user_message: dict, assistant_response: dict) -> list:
        """Add a turn of conversation to the chat history collection and in-memory buffer"""
        try:
            if not self.vecstore:
                logger.info("Getting chat history vector store...")
                self.create_get_collection()

            conversation = f"User: {user_message['content']}\nAssistant: {assistant_response['content']}"
            doc = Document(
                page_content=conversation,
                metadata={
                    "timestamp": datetime.now().isoformat(), 
                    "type": "conversation_turn", 
                    "user_message": user_message['content'], 
                    "assistant_response": assistant_response['content']
                }
            )
            self.vecstore.add_documents([doc])
            logger.info("Message added to chat history.")
            self.chat_memory.append(conversation)
            if len(self.chat_memory) > chat_collection_settings["chat_memory_size"]:
                self.chat_memory.pop(0)  # Maintain fixed size

            return self.chat_memory

        except Exception as e:
            logger.error(f"Error adding message to chat history: {e}")
            raise
        
    def get_buffer_memory(self) -> list:
        if self.chat_memory:
            return self.chat_memory
        else:
            return []

    # TODO: Improve chat memory retriever
    # TODO: This should go on rag pipeline file
    def get_k_messages(self, k: int = chat_collection_settings["similarity_search_size"]) -> list:
        try:
            if not self.vecstore:
                logger.info("Getting chat history vector store...")
                self.create_get_collection()
            
            retriever = self.vecstore.as_retriever(search_type=chat_collection_settings["search_type"], search_kwargs={"k": k})
            docs = retriever.get_relevant_documents("")  # Empty query to get top k by recency
            
            return docs
        
        except Exception as e:
            logger.error(f"Error retrieving chat messages: {e}")
            raise