'''Implement HDBSCAN over a bunch of retrived chunks, useful for:
- Any base retriever (except for quick answer)
- Academic essay generation
- Creative explanation
- Complex analytical step
- Literary analysis
- Concept tracking
- Multi-domain query
- Detailed analysis
- Low confidence
'''
import settings
import logging
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

cluster_parameters = settings.CHUNK_CLUSTERING_PARAMS['hdbscan_params']
sim_threshold = settings.CHUNK_CLUSTERING_PARAMS['sim_threshold']

logger = logging.getLogger(__name__)

class Clusterer:
    def __init__(self):
        self.min_cluster_size = cluster_parameters['min_cluster_size']
        self.max_cluster_size = cluster_parameters['max_cluster_size']
        self.metric = cluster_parameters['metric']
        self.allow_single_cluster = cluster_parameters['allow_single_cluster']
        self.docs = None
        self.embeddings = None

    def _compute_cluster_coherence(self, cluster_data: dict) -> dict:
        """Compute thematically coherence per cluster"""
        embeddings = cluster_data["embeddings"]
        if len(embeddings) <= 1:
            return 1.0 if len(embeddings) == 1 else 0.0
        
        centroid = cluster_data['centroid']
        similarities = cosine_similarity(embeddings, [centroid]).flatten()
        cluster_data['coherence'] = float(np.mean(similarities))
        return cluster_data
        
        return cluster_data

    def cluster_chunks(self, docs: list, embeddings: np.ndarray) -> np.ndarray:
        '''Cluster embedded chunks'''
        if not docs or not embeddings.size == 0:
            logger.warning("Empty docs or embeddings provided")
            return np.array([])
        
        if len(docs) != len(embeddings):
            logger.error("Docs and embeddings must have same length")
            raise ValueError("Docs and embeddings must have same length")

        self.docs = docs
        self.embeddings = embeddings

        # Adaptative cluster min size for small document sets
        min_size = min(self.min_cluster_size, len(docs))
        if min_size < 2:
            logger.warning(f"Dataset too small for clustering with min_cluster_size={self.min_cluster_size}")
            return np.full(len(docs), -1) # return as noise points if not 
         
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_size, 
                metric=self.metric, 
                max_cluster_size=self.max_cluster_size,
                allow_single_cluster=self.allow_single_cluster,
            )

            labels = clusterer.fit_predict(embeddings)
            logger.info(f"Clustering completed: {len(set(labels)) - (1 if -1 in labels else 0)} clusters found")
            return labels
        
        except Exception as e:
            logger.error(f"Chunk clustering failed: {e}")
            return np.full(len(docs), -1) # return all points as noise if not meet minimumm cluster siz

    def group_docs(self, labels: np.ndarray) -> dict:
        '''Group docs by cluster'''
        if labels is None or len(labels) == 0:
            return {}
    
        clusters = {}
        for i, (label, doc) in enumerate(zip(labels, self.docs)):
            if label == -1:
                continue

            if label not in clusters:
                clusters[label] = {
                    'document': [],
                    'size': 0,
                    'indices': []
                }
            clusters[label]['documents'].append(doc)
            clusters[label]['indices'].append(i)
            clusters[label]['size'] += 1

        cluster_data = {}
        for cluster_id, data in clusters.items():
            cluster_embeddings = self.embeddings[data['indices']]
            centroid = np.mean(cluster_embeddings, axis=0)
        
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            cluster_data[cluster_id] = {
                'documents': data['documents'],
                'size': len(data['documents']),
                'indices': data['indices'],
                'centroid': centroid,  # Real, usable centroid vector
                'centroid_norm': centroid_norm,
                'embeddings': cluster_embeddings
            }

            logger.info(
                f"Grouped into {len(cluster_data)} clusters" 
            )

        return cluster_data

    def find_relevant_clusters(
            self, 
            query_embedding: np.ndarray,  
            clusters: dict,
            sim_threshold: float = sim_threshold) -> list:
        """Find clusters moost relevant to a query"""
        relevant_clusters = []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logger.warning("Zero-norm query embedding")
            return []
        
        query_emb_norm = query_embedding / query_norm

        for cluster_id, cluster_data in clusters.items():
            centroid = cluster_data['centroid']
            centroid_norm = cluster_data['centroid_norm']

            if centroid_norm == 0:
                continue

            similarity = np.dot(query_emb_norm, centroid)

            if similarity >= sim_threshold:
                relevant_clusters.append(cluster_id)

        return sorted(relevant_clusters, key=lambda x: clusters[x]['size'], reverse=True)
    