# Auxiliar functions for retriever pipeline
from sklearn.metrics.pairwise import cosine_similarity
import logging
from app import settings

logger = logging.getLogger(__name__)

token_limits = settings.TOKEN_LIMITS["retrievers"]
emb_dimensionality = settings.VECSTORE_GLOBAL_SETTINGS

class EmbeddingExtractionHelper:
    """
    This class contains all auxiliar functions that
    contribute to embedding extraction at retrieval layer,
    right after documents are retrieved
    """
    def __init__(self):
        self.docs_w_ids: list = []
        self.vecstore = None

    def _get_stored_embedings(self) -> list:
        """Try to retrieve stored embeddings from vector store"""
        try:
            # This works just with chroma
            if hasattr(self.vecstore, '_collection'):
                collection = self.vecstore._collection

                # Get doc ids
                doc_ids = [doc.metadata['document_id'] for doc in self.docs]
                results = collection.get(
                    ids=doc_ids,
                    include=["embeddings", "metadatas"]
                )

                if results and results.get("embeddings"):
                    logger.info(f"Retrieved {len(results['embeeddings'])} stored embeddings")
                    return results["embeddings"]
                
        except Exception as e:
            logger.warning(f"Failed to retrieve stored embeddings: {e}")
            return None

    def _regenerate_embeddings(self) -> list:
        """Fallback: Re-embed document content"""
        embeddings = []

        for doc in self.docs:
            try:
                # Truncate very long docs
                content = doc.page_content
                if len(content) > token_limits["regenerate_embeddings_limit"]:
                    content = content[:token_limits["regenerate_embeddings_limit"]]
                    logger.debug(f"Truncated document for embedding: {doc.metadata['document_id']}")

                    embedding = self.vecstore.embedding_function.embed_query(content)
                    embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {doc.metadata['document_id']}: {e}")

                # add zero vector as placeholder, based on emb model dimensionality
                zero_emb = [0.0] * emb_dimensionality
                embeddings.append(zero_emb)
        
        return embeddings
    
    def _similarity_scores(self, query_emb: list, doc_embs: list) -> list:
        """Calculate cosine similarity between query and retrieved docs"""
        sim_scores = []
         
        for doc_emb in doc_embs:
            try:
                sim = cosine_similarity(query_emb, doc_emb)
                if sim:
                    sim_scores.append(float(sim))

            except Exception as e:
                logger.warning(f"Error calculating similarity: {e}")
                sim_scores.append(0.0)

        return sim_scores
                
    def get_doc_embeddings(self, docs_w_ids: list, query: str = None, vecstore = None) -> dict:
        """Get embeddings for retrieved docs"""
        if not vecstore:
            logger.error("Could not locate vecstore.")

        self.vecstore = vecstore
        self.docs_w_ids = docs_w_ids

        try:
            embedding_data = {
                "document_embeddings": [],
                "document_ids": [doc.metadata["document_id"] for doc in docs_w_ids],
                "query_embedding": None,
                "similarity_scores": [],
                "retrieval_metadata": {}
            }

            # Get query embeddings
            if query:
                try:
                    # ? embedding_function or ef?
                    query_embedding = vecstore.embedding_function.embed_query(query)
                    embedding_data["query_embedding"] = query_embedding
                    embedding_data["query_embedding_dim"] = len(query_embedding)

                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {e}")
            
            # Strategy 1: Try to get embeddings from vecstore
            stored_embeddings = self._get_stored_embeddings(docs_w_ids)
            if stored_embeddings:
                embedding_data["document_embedddings"] = stored_embeddings
                embedding_data["embedding_source"] = "vector_store"
            else:
                # Strategy 2: Fallback - regenerate embeddings
                regenerated_embeddings = self._regenerate_embeddings(docs_w_ids)
                embedding_data["document_embeddings"] = regenerated_embeddings
                embedding_data["embedding_source"] = "regenerated"
            
            # Calculate similarity scores if have both query and document embeddings
            if embedding_data["query_embedding"] and embedding_data["document_embeddings"]:
                similarity_scores = self._calculate_similarities(
                    embedding_data["query_embedding"], 
                    embedding_data["document_embeddings"]
                )
                embedding_data["similarity_scores"] = similarity_scores

                # Add similarity scores to document metadata
                for i, (doc, similarity) in enumerate(zip(docs_w_ids, similarity_scores)):
                    doc.metadata["similarity_score"] = float(similarity)
                    doc.metadata["similarity_rank"] = i + 1

            # Add metadata
            if embedding_data["document_embeddings"]:
                embedding_data["retrieval_metadata"].update({
                    "embeddinng_dimensions": len(embedding_data["document_embeddings"][0]),
                    "total_embeddings": len(embedding_data["document_embeddings"]),
                })

            return embedding_data
         
        except Exception as e:
            logger.error(f"Error while getting retrieved docs embeddings: {e}")
            return {
                "error": str(e),
                "document_embeddings": [],
                "document_ids": [],
                "query_embedding": None,
                "similarity_scores": []
            }