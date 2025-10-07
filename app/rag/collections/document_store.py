from llama_index.readers.file.docs.base import PDFReader
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, SemanticSplitterNodeParser, SimpleNodeParser # TODO: Change SimpleNodeParser for a own customized parser 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from pathlib import Path
from app import settings
import logging

logger = logging.getLogger(__name__)

vecstore_global_settings = settings.VECSTORE_GLOBAL_SETTINGS
collection_settings = settings.DOCUMENT_COLLECTION_SETTINGS

class DocIndexer:
    # Extract text
    # TODO: Dinamically changes parameters depending doc size
    # todo: by now, this implementation is made for large and complex docs
    def pdf_to_nodes(self, 
                     path: str = vecstore_global_settings["pdf_test"],  #TODO: Place path
                     chunk_sizes: int = collection_settings["chunk_sizes"], 
                     chunk_overlap: int =collection_settings["chunk_overlap"]) -> list:
        reader = PDFReader()
        docs = reader.load_data(file=path)
        splitter = SentenceSplitter(chunk_size=chunk_sizes[0],
                                    chunk_overlap=chunk_overlap,
                                    include_metadata=True
                                    )
        nodes = splitter.get_nodes_from_documents(docs)
        logger.info(f"Indexed {len(nodes)} nodes")
        # TODO: Implement this method
        """base_parser = SimpleNodeParser.from_defaults() # TODO: Change for own customized
        parser_map = {"simple": base_parser}

        hierarch_parser = HierarchicalNodeParser(
            chunk_size=chunk_sizes, 
            chunk_overlap=chunk_overlap, 
            include_metadata=True,
            node_parser_map=parser_map,
            node_parser_ids=['simple']
            )
        hi_nodes = hierarch_parser.get_nodes_from_documents(docs)

        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=vecstore_global_settings["embedding_model"],
        )"""
        
        """nodes = []
        for node in hi_nodes:
            split_nodes = semantic_splitter.get_nodes_from_documents([node])
            nodes.extend(split_nodes)"""

        return nodes
    
class DocVecStore:
    def __init__(self, persist_dir=vecstore_global_settings["persist_directory"]):
        self.persist_dir = persist_dir
        self.ef = HuggingFaceEmbeddings(
            model_name=vecstore_global_settings["embedding_model"],
        )
        self.vecstore = None

    def create_get_collection(self):
        """Get collection if exists, create it otherwise"""
        collection_name = collection_settings["collection_name"]
        collection_metadata = collection_settings["collection_metadata"]
        
        try:
            self.vecstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.ef,
                collection_name=collection_name,
                collection_metadata=collection_metadata
            )
            if self.vecstore:
                logger.info(f"Vector store {collection_name} loaded.")
            return self.vecstore
        except Exception as e:
            logger.error(f"Error while getting collection: {e}")
            raise

    def populate_collection(self, nodes: list) -> None:
        try:
            if not self.vecstore:
                logger.info("Getting vector store...")
                self.create_get_collection()
            
            documents = []
            for node in nodes:
                documents.append(Document(
                    page_content=node.text,
                    metadata=node.metadata
                ))
            
            self.vecstore.add_documents(documents)
            logger.info(f"All {len(nodes)} nodes were indexed")
        
        except Exception as e:
            logger.error(f"Error indexing document chunks: {e}")
            raise