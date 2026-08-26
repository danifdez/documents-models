# Detect outliers

Outlier analysis identifies unusual values in a numeric field using the interquartile range method.

## Requirements

Choose one numeric field with at least four valid values.

## How boundaries are calculated

Documents calculates the lower and upper quartiles and their difference, then uses:

- lower boundary: lower quartile minus 1.5 times the interquartile range;
- upper boundary: upper quartile plus 1.5 times the interquartile range.

Values beyond either boundary are marked as outliers.

## Result

The result contains box-plot values, both boundaries, every detected outlier, and the original record identity for plotted points. Statistics include total and outlier counts, outlier percentage, mean, standard deviation, and the number of values more than three standard deviations from the mean.

An outlier is unusual under this statistical rule; it is not automatically an error and should be reviewed in context.
