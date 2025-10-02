'''Summarizes reetrieved text using a programmatic approach, with heuristics
This method may be called when: Evidence gathering needed, Multiple domains detected, or as part of any multi-query retriever call.'''
import re
from app import settings
import logging

logger = logging.getLogger(__name__)

stop_words =  settings.HEURISTIC_SUMMARIZER_PARAMS["stop_words"]
summary_length = settings.HEURISTIC_SUMMARIZER_PARAMS["summary_length"]

class HeuristicSummarizer:
    def __init__(self):
        self.text: str = ""
    
    def _get_sentences(self) -> list:
        """Split text into sentences using common punctuation"""
        sentences = re.split(r'([.?!]+)', self.text)

        # combine sentence fragment with its trailing punctuation
        full_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if not sentence:
                continue

            # add punctuation delimeter back if exists
            if i + 1 < len(sentences):
                full_sentences.append(sentence + sentences[i+1])
            else:
                full_sentences.append(sentence)
        
        return [s.strip() for s in full_sentences if s.strip()]

    def _get_words(self, sentence: str) -> list:
        """Cleans a sentence and returns a list of lowecase words"""
        return [word.lower() for word in re.findall(r'\b\w+\b', sentence)]

    def summarize_text(self, 
                       text: str,
                       summary_length: float = summary_length,
                       stop_words: set = stop_words
                    ) -> str:
        """
        Generate extractive summary using frequency-based scoring method
        Steps:
        1. Tokenize text into sentences
        2. Count frequency of words (excluding stop)
        3. Score each sentence based on total freq. of its important words
        4. Select top N sentences and reorder based on their appearance in original text
        """
        self.text = text
        
        sentences: list = self._get_sentences()
        # frecuency count
        word_freq: dict = {}
        for s in sentences:
            words = self._get_words(s)
            for w in words:
                # ignore stop and too short words
                if w not in stop_words and len(w) > 2:
                    word_freq[w] = word_freq.get(w, 0) + 1
        
        # sentence scoring
        scores = [] # this will store (text, score, index)

        for i, sentence in enumerate(sentences):
            words = self._get_words(sentence)
            score = 0.0

            for w in words:
                score += word_freq.get(w, 0)

            # normalize score by sentence length to favor coonsice sentence
            norm_score = score / (len(words) or 1)

            scores.append((sentence, norm_score, i))

        # top N selection
        scores.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Number of sentences in original text: {len(sentences)}")

        selected_sentences = scores[:int(float(summary_length) * float(len(sentences)))]
        logger.info(f"Summarized {summary_length*100:.1f}% of original text, resulting in {int(float(summary_length) * float(len(sentences)))} sentences.")
        
        # ordering by original index
        selected_sentences.sort(key=lambda x: x[2])

        # output
        summary = " ".join([text for text, score, index in selected_sentences])

        return summary