import pandas as pd

# Load the patterns data from Excel
patterns_df = pd.read_excel('/Users/paola_amigo/Desktop/Thesis/JazzSolos/data/Patterns/pattern_output_pitch_5_10_5_1_db.xlsx', engine='openpyxl')  

# Check the column names to verify if 'id' is present
print(patterns_df.columns)

# Split the 'id' column to get 'performer' and 'title'
if 'id' in patterns_df.columns:
    # Assuming the format is 'PerformerName_SongTitle_FINAL.sv'
    patterns_df[['performer', 'title']] = patterns_df['id'].str.extract(r'([^_]+)_(.*?)_FINAL.sv')

    # Drop the original 'id' column if it's no longer needed
    patterns_df = patterns_df.drop(columns=['id'])

    # Save the cleaned patterns data as a CSV file
    patterns_df.to_csv('/Users/paola_amigo/Desktop/Thesis/JazzSolos/data/patterns/patterns_processed.csv', index=False)
else:
    print("Column 'id' not found. Please check the Excel file.")