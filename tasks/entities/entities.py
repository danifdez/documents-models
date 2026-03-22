import logging
from typing import Dict, List, Any
import spacy
from utils.job_registry import job_handler
from utils.device import HAS_CUDA, get_spacy_model

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model to avoid crash at import time if not installed
nlp = None

def _get_nlp():
    global nlp
    if nlp is None:
        if HAS_CUDA:
            spacy.prefer_gpu()
        from services.model_config import get_task_config
        task_config = get_task_config("entity-extraction")
        model_name = task_config.get("model") or get_spacy_model()
        nlp = spacy.load(model_name)
        logger.info("spaCy loaded model: %s", model_name)
    return nlp

@job_handler("entity-extraction")
def entities(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract named entities from text using spaCy's en_core_web_trf model.
    
    This function processes texts to identify and extract named entities such as
    persons, organizations, locations, dates, etc. using spaCy's transformer-based
    model which provides high accuracy entity recognition.
    
    Parameters:
    payload (Dict[str, Any]): Dictionary containing 'texts' key with list of texts
                              to process. Each text can be a string or dict with 'text' key.
    
    Returns:
    Dict[str, List[Dict[str, str]]]: Dictionary with 'entities' key containing
                                     list of unique entities with 'word' and 'entity' keys.
    """
    texts = payload["texts"]
    parse_result = []
    
    # Process texts in batches for better performance
    # spaCy's nlp.pipe is more efficient than processing texts individually
    text_strings = []
    for item in texts:
        if isinstance(item, dict) and 'text' in item:
            text_strings.append(item['text'])
        else:
            text_strings.append(str(item))
    
    # Entity types to ignore (typically numerical or temporal values)
    ignored_entity_types = {
        'CARDINAL',  # Numerals that do not fall under another type
        'DATE',      # Absolute or relative dates or periods
        'MONEY',     # Monetary values, including unit
        'ORDINAL',   # "first", "second", etc.
        'PERCENT',   # Percentage, including "%"
        'QUANTITY',  # Measurements, as of weight or distance
        'TIME'       # Times smaller than a day
    }
    
    # Use spaCy's pipe method for efficient batch processing
    docs = _get_nlp().pipe(text_strings, batch_size=32)
    
    for doc in docs:
        for ent in doc.ents:
            # Filter out ignored entity types and very short entities
            if (len(ent.text.strip()) > 1 and 
                ent.label_ not in ignored_entity_types):
                parse_result.append({
                    'word': ent.text.strip(),
                    'entity': ent.label_,
                })
    
    # Remove duplicates while preserving order
    unique_result = []
    seen = set()
    for ent in parse_result:
        key = (ent['word'], ent['entity'])
        if key not in seen:
            seen.add(key)
            unique_result.append(ent)
    
    return {"entities": unique_result}
