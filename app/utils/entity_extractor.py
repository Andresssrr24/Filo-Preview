import spacy
from functools import lru_cache
from app import settings

model_name = settings.ENTITY_EXTRACTION_SETTINGS

@lru_cache(maxsize=1)
def load_model():
    return spacy.load(model_name)

class EntityExtractor:
    def __init__(self):
        self.model = load_model()
        self.texts: list = []

    def _format_entities(self, doc):
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'score', 1.0)
            }
            for ent in doc.ents
        ]

    # TODO: Review batch size
    def extract(self, texts: list, batch_size: int = 100) -> list:
        self.texts = texts

        docs  = self.model.pipe(texts, batch_size=batch_size)
        return [self._format_entities(doc) for doc in docs]
    




