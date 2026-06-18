"""Step 2: take a first look at the match data."""

import pandas as pd  # the "spreadsheet for code" library

# Tell pandas: don't hide any columns, and use the full width before wrapping.
# Without these, pandas collapses middle columns into "..." to save space.
pd.set_option("display.max_columns", None)  # show every column
pd.set_option("display.width", None)        # don't squeeze to a fixed width

# Read the CSV file into a "DataFrame" - pandas' name for a table (rows + columns).
matches = pd.read_csv("data/results.csv")

# .shape tells us the size of the table as (number of rows, number of columns).
print("Size (rows, columns):", matches.shape)

# .columns lists the column names - what info we have about each match.
print("\nColumns:", list(matches.columns))

# .head() shows the first 5 rows so we can see what real data looks like.
print("\nFirst 5 matches:")
print(matches.head())

# .tail() shows the last 5 rows - the most recent matches in the file.
print("\nMost recent 5 matches:")
print(matches.tail())
