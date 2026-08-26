# Correlation between two fields

This analysis measures the linear relationship between two numeric dataset fields using Pearson correlation.

## Requirements

Choose two numeric fields with at least three valid paired data points.

## Result

Documents prepares a scatter plot and a linear regression line. The accompanying statistics include:

- the Pearson correlation from -1 to 1;
- the significance value;
- the proportion of variation described by the fitted line;
- the number of valid pairs;
- the slope and intercept of the line.

A value close to 1 or -1 indicates a strong linear relationship, while a value near 0 indicates little linear relationship. The result does not prove that one field causes the other.

If fewer than three valid pairs remain, the analysis fails instead of calculating an unreliable result.
