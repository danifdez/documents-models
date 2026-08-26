# Query datasets

Dataset queries can filter, select, group, aggregate, and prepare records for tables or charts. They can use one dataset or combine several datasets through a shared field.

## Available operations

- filter records by equality, inequality, comparison, or text containment before analysis;
- select which fields are returned;
- group by a field;
- calculate count, mean, sum, minimum, maximum, or median;
- suggest a bar, line, or pie chart for grouped results;
- combine datasets using a selected joining field.

The first selected dataset is treated as the primary dataset when several are combined.

## Result

With grouping, Documents returns group labels, aggregated values, group count, total records, and matching table data.

Without grouping, Documents returns the selected rows and columns, together with total and returned record counts. This result does not suggest a chart.
