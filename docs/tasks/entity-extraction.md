# Extract entities

Entity extraction finds named people, organizations, places, groups, facilities, products, events, works of art, laws, and languages in document content.

## Large documents

Documents divides long content into sections of at most 1,500 words by default. Each section is analyzed, then accepted results are combined in the original order.

Duplicate names are merged without regard to capitalization while preserving the first appearance.

## Result

Each candidate includes the text that was found and its proposed category. Documents recognizes:

- person;
- organization;
- country, city, or state;
- other location;
- nationality, religious group, or political group;
- named event;
- facility;
- product;
- work of art;
- language;
- law.

Most proper-name categories require capitalization to reduce false positives. National, religious, or political groups and languages can remain lowercase for multilingual compatibility.

Results appear as pending candidates so a user can correct, discard, merge, or confirm them before they become confirmed project entities.
