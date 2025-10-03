from app.tools.retrieval.retrievers import RetrieverTools
from app.rag.collections.document_store import DocVecStore, DocIndexer
from app.rag.post_retrieval.chunk_clusterer import Clusterer
from dotenv import load_dotenv
import os
import logging
from app import settings
logger = logging.getLogger(__name__)

load_dotenv()

def test_rag_pipeline():
    vecdb = DocVecStore()
    vecstore = vecdb.create_get_collection()
    indexer = DocIndexer()
    nodes = indexer.pdf_to_nodes(path=os.getenv("PDF_PATH"))
    if len(nodes) > 2:
        logger.info('Nodes indexed')
    vecdb.populate_collection(nodes)

    ret_tools = RetrieverTools(vecstore=vecstore)
    ret_tools.check_vector_store_content()
    query = "In an act of justice"
    docs = ret_tools.base_retriever_tool(query)
    #rsp = ret_tools.multi_query_retriever_tool(query)
    logger.info("Docs retrieved")
    print(docs)

    '''clusterer = Clusterer()
    clusterer.cluster_chunks(docs, )'''


test_rag_pipeline()
retriever_tools_settings = settings.RETRIEVER_TOOLS_SETTINGS