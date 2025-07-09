from utils.job_registry import job_handler
from transformers import pipeline

@job_handler("entity-extraction")
def entities(payload) -> dict:
    pipe = pipeline("token-classification", model="dslim/bert-base-NER")
    texts = payload["texts"]
    chunk_size = 32
    parse_result = []

    for i in range(0, len(texts), chunk_size):
        chunk = [item['text'] if isinstance(item, dict) and 'text' in item else item for item in texts[i:i+chunk_size]]
        output = pipe(chunk)
    
        for elements in output:
            for entity in elements:
                try:
                    if '##' in entity['word']:
                        parse_result[-1]['word'] += entity['word'].replace(
                            '##', '')
                    elif entity['entity'].startswith('B-'):
                        parse_result.append({
                            'word': entity['word'],
                            'entity': entity['entity'].split('-')[1],
                        })
                    else:
                        parse_result[-1]['word'] += ' ' + entity['word']
                except:
                    continue

    unique_result = []
    seen = set()
    for ent in parse_result:
        key = (ent['word'], ent['entity'])
        if key not in seen:
            seen.add(key)
            unique_result.append(ent)
    return {"entities": unique_result}
