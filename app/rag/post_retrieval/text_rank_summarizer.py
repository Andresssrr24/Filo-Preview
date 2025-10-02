"""Implement TextRank summarizer with fallbacks, augmentation, and reliability metrics
Used when:
- Base retriever (OCR)
- Academic essay generation
- Creative explanation
- Multi-domain query
- Multi-source
"""
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from app import settings
from heuristic_summarizer import HeuristicSummarizer
import logging

logger = logging.getLogger(__name__)

thresholds = settings.TEXTRANK_SUMMARIZER_PARAMS["confidence_thresholds"]
heuristic_summary_length = settings.TEXTRANK_SUMMARIZER_PARAMS["heuristic_summary_length"]
cluster_adequancy = settings.TEXTRANK_SUMMARIZER_PARAMS["cluster_adequancy"]
penalties = settings.TEXTRANK_SUMMARIZER_PARAMS["penalties"]

class TextRankSummarizer:
    '''TextRank summarizer for RAG retrieved text, wont receive less than 3 clusters'''
    def __init__(self):
        self.thresholds = thresholds
        self.text: str = ""
        self.num_sentences: int = 0
        self.summaries = {}
        self.is_augmented = False
    
    def _augment_summary(self, cluster_data) -> dict:
        """Merge summary with heuristic summarizer to deliver both perspectives"""
        original_text = cluster_data['text']
        heuristic_summary = HeuristicSummarizer.summarize_text(
            text=original_text, 
            summary_length=0.5
        )

        self.summaries['heuristic_summary'] = heuristic_summary

        return self.summaries

    def _add_summaries(self, cluster_data: dict, summaries: dict):
        """Add summary (or summaries) to cluster data"""
        final_summary = summaries["textrank_summary"]
        self.is_augmented = False

        if 'heuristic_summary' in summaries.keys():
            sum2 = summaries["heuristic_summary"]
            final_summary = f"Perspective 1: {final_summary}\n + Perspective 2: {sum2}"

            self.is_augmented = True
            cluster_data["is_augmented"] = self.is_augmented
        
        cluster_data["summary"] = final_summary

        return cluster_data

    def summary_reliability(self, cluster_data: dict, summary: str) -> dict:
        """Compute summary reliability per cluster"""
        reliability = 1.0

        # Use cluster coherence as factor just when it wasn't augmented
        if not self.is_augmented:
            # Factor 1: Cluster coherence
            coherence = cluster_data['coherence']

        # Factor 2: Cluster size addequacy
        num_chunks = len(cluster_data)
        if num_chunks < cluster_adequancy["optimal_min"]:
            # heavy penalization below optimal
            adequancy_score = penalties["growth_rate_below_optimal"] * num_chunks
        elif num_chunks > cluster_adequancy["optimal_max"]:
            # moderate penalization above optimal
            adequancy_score = penalties["growth_rate_above_optimal"] * num_chunks
        else:
            adequancy_score = 1.0

        # Factor 3: Summary coverage

        pass

    def summary_strategy(self) -> dict:
        """Choose type of summary retrieval depending on its reliability"""    
        pass

    def textrank(self) -> dict:
        """Performs textrank and saves summary in a dict"""
        pass
