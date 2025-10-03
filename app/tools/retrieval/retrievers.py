from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_tavily import TavilySearch
from app import settings
import logging
import os
from dotenv import load_dotenv
from app.llms.llms import llms

load_dotenv()
logger = logging.getLogger(__name__)

retriever_tools_settings = settings.RETRIEVER_TOOLS_SETTINGS
models = llms()

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

    def _format_output(self, docs: list) -> str:
        form_docs = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page_label', 'N/A')
            source = doc.metadata.get('file_name', 'N/A')

            form_docs.append(
                f"Document {i}:\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {doc.page_content}"
            )

        return "\n\n".join(form_docs)

    def check_vector_store_content(self):
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

    '''def _get_docs_embeddings(self, docs: list, query: str = None) -> dict:
        """Retrieve doc embeddings"""
        try:
            if hasattr(self.vecstore, '_collection'):
                collection = self.vecstore._collection

                doc_ids'''

    # Base retriever tool
    def base_retriever_tool(self, query: str) -> str:
        """Basic search using base retriever """
        try:
            docs = self.base_retriever.invoke(query)
          
            if not docs:
                return "No relevant docs found."
            
            return self._format_output(docs)
        
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    # Multi-Query retriever tool
    # TODO: Fallback must be executed programmatically to reduce api calls
    def multi_query_retriever_tool(self, query: str) -> str: # TODO: optimize main llm token usage
        """Retrieve documents by generating multiple synthetic queries with an llm """
        try:
            docs = self.multi_q_retriever.invoke(query)
            
            if not docs:
                logger.info("No docs found, changing to base_retriever")
                return self.base_retriever_tool(query)
            
            return self._format_output(docs)
        except:
            logger.info("Couldn't use multi-query retriever, changing to base_retriever")
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