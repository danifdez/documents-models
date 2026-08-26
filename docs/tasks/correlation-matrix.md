# Correlation matrix

A correlation matrix compares every pair in a set of numeric fields using Pearson correlation.

## Requirements

Select at least two numeric fields. If no fields are selected explicitly, Documents uses all numeric fields in the dataset.

## Result

The result is a square heatmap-ready matrix with a correlation coefficient and significance value for every pair. The diagonal is always 1 because each field is perfectly correlated with itself.

Documents also lists up to ten strongest pairs whose absolute correlation is at least 0.5, ordered from strongest to weakest.

A correlation near 1 means the fields tend to increase together, a value near -1 means one tends to decrease as the other increases, and a value near 0 indicates little linear relationship. Correlation does not establish causation.
