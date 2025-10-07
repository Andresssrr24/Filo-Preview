from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_tavily import TavilySearch
import logging
import hashlib
import os
from app import settings
from dotenv import load_dotenv
from app.utils.retriever_utils import EmbeddingExtractionHelper
from app.llms.llms import llms

load_dotenv()
logger = logging.getLogger(__name__)

retriever_tools_settings = settings.RETRIEVER_TOOLS_SETTINGS

models = llms()
emb_extractor_help = EmbeddingExtractionHelper()

class RetrieverTools:
    """Retriever tools for llm"""
    def __init__(self, vecstore, query_generator_llm=models["query_generator"], tavily_api_key=os.getenv("TAVILY_API_KEY")):
        self.vecstore = vecstore
        self.base_retriever = self.vecstore.as_retriever(
            search_type=retriever_tools_settings["base_retriever_search_type"],
            search_kwargs={"k": retriever_tools_settings["base_retriever_k"]}
        )
        self.multi_q_retriever = MultiQueryRetriever.from_llm(
            include_original=True, 
            retriever=self.vecstore.as_retriever(
                search_type=retriever_tools_settings["multi_query_search_type"], search_kwargs={"k": retriever_tools_settings["multi_query_k"]}
                ), 
            llm=query_generator_llm
        )
        self.web_search = TavilySearch(
            api_key=tavily_api_key, 
            max_results=retriever_tools_settings["web_search_k"], search_depth=retriever_tools_settings["search_depth"]
        )
    def _format_output(self, docs: list, query: str, include_embeddings: bool = False) -> dict:
        """Output formatter that returns structured data from retrieved docs"""
        form_docs = []
        embedding_data = None

        # get embeddings if requested
        if include_embeddings and query:
            embedding_data = emb_extractor_help.get_doc_embeddings(docs=docs, query=query, vecstore=self.vecstore)

            # add similarity scores to doc metadat
            if embedding_data and embedding_data.get("similarity_scores"):
                for i, doc in enumerate(docs):
                    if i < len(embedding_data["similarity_scores"]):
                        doc.metadata["similarity_scores"] = embedding_data["similarity_scores"][i]

        # format docs
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page_label', 'N/A')
            source = doc.metadata.get('file_name', 'N/A')
            similarity_score = doc.metadata.get('similarity_score', 'N/A')

            # format sim score
            sim = f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else similarity_score

            form_docs.append(
                f"Document {i}:\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Similarity: {similarity_score}\n"
                f"Content: {doc.page_content}"
            )
        
        # build result
        result = {
            "formatted_text": "\n\n".join(form_docs),
            "document_count": len(docs),
            "query": query
        }

        # add embeddings if available        
        if include_embeddings and embedding_data:
            result["embeddings"] = embedding_data

        return result
    
    def _append_doc_id(self, docs: list) -> list:
        """Generate id for each retrieved doc and append relevant data"""
        for i, doc in enumerate(docs):
            # Ensure metadata exists
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            # Generate unique ID using content hash and index
            content_preview = doc.page_content[:100]  # First 50 chars for hash
            content_hash = hashlib.md5(content_preview.encode()).hexdigest()[:6]
            
            doc.metadata.update({
                "document_id": f"retrieved_{content_hash}_{i}",
                "retrieval_rank": i + 1,
                "content_length": len(doc.page_content)
            })
        
        return docs

    def check_vector_store_content(self):
        # ! TEST FUNCTION
        """Check if vector store has documents"""
        try:
            # Try to get all documents (if supported by your vector store)
            if hasattr(self.base_retriever, 'get_relevant_documents'):
                # Get some test queries
                test_docs = self.base_retriever.invoke("test")
                logger.info(f"Test query returned {len(test_docs)} docs")
                
            # Check vector store statistics
            if hasattr(self.vecstore, '_collection'):
                collection = self.vecstore._collection
                count = collection.count()
                logger.info(f"Vector store has {count} documents")
                
        except Exception as e:
            logger.error(f"Error checking vector store: {e}")

    # Base retriever tool
    def base_retriever_tool(self, query: str, include_embeddings: bool = False) -> str:
        """Basic search using base retriever """
        try:
            docs = self.base_retriever.invoke(query)
          
            if not docs:
                return "No relevant docs found."
            
            return self._format_output(docs, query, include_embeddings)
        
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    # Multi-Query retriever tool
    # TODO: Fallback must be executed programmatically to reduce api calls
    def multi_query_retriever_tool(self, query: str, include_embeddings: bool = False) -> str: # TODO: optimize main llm token usage
        """Retrieve documents by generating multiple synthetic queries with an llm """
        try:
            docs = self.multi_q_retriever.invoke(query)
            
            if not docs:
                logger.warning("No docs found, changing to base_retriever")
                return self.base_retriever_tool(query)
            
            return self._format_output(docs, query, include_embeddings)
        except:
            logger.warning("Couldn't use multi-query retriever, changing to base_retriever")
            return self.base_retriever_tool(query)
        
    def web_search_tool(self, query: str) -> str: 
        """Search the web for information related to the query"""
        try:
            results = self.web_search.invoke(query)

            if not results:
                return f"No web results found for {query}."

            """formatted = "\n".join(
                f"- {r['title']}: {r['content']} (source: {r['url']})"
                for r in results
            )"""
            return f"Web search results for '{query}':\n{results}" 
        
        except Exception as e:
            return f"Error during web search: {str(e)}"