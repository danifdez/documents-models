# Create chart data

This analysis prepares dataset values for bar, line, pie, or scatter charts.

## Choices

Users choose:

- the chart type;
- a field for the horizontal axis;
- an optional numeric field for the vertical axis;
- filters to apply first;
- how values are aggregated: count, mean, sum, minimum, maximum, or median;
- whether to sort by label or value and in which direction;
- the maximum number of categories, which defaults to 20.

If no vertical value field is selected, Documents counts occurrences.

## Result

Bar, line, and pie charts group the records by the horizontal field and apply the chosen aggregation. The result includes labels, values, record and category statistics, chart title, axis labels, and equivalent table data.

Scatter charts do not group values. They return the raw numeric coordinate pairs and the total number of points.
