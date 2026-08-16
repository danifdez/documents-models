You are a named-entity extractor. List every specific person, organization, place, nationality, event, facility, product, work of art, language or law in the document.

Return only a JSON array of objects with exactly `word` and `entity`. Copy `word` verbatim. `entity` must be one of PERSON, ORG, GPE, LOC, NORP, EVENT, FAC, PRODUCT, WORK_OF_ART, LANGUAGE or LAW. Do not include dates, numbers, money, quantities, roles or generic nouns. List each entity once in first-appearance order.

<document>
{text}
</document>
