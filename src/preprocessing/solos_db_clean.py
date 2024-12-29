import pandas as pd
import os
import re

# Define relative paths for input and output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
input_file = os.path.join(data_dir, 'solos_db_with_durations.csv')
output_file = os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv')

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Initial dataset size: {df.shape}")

# Step 1: Remove solos with missing 'releasedate'
df = df.dropna(subset=['releasedate'])
print(f"After removing missing 'releasedate': {df.shape}")

# Step 2: Handle duplicates by 'performer' and 'title' based on `melid` (keep the one with the fewest notes)
df['num_notes'] = df.groupby('melid')['eventid'].transform('count')
duplicates_by_performer = df[df.duplicated(subset=['performer', 'title'], keep=False)]
melids_to_keep = duplicates_by_performer.loc[
    duplicates_by_performer.groupby(['performer', 'title'])['num_notes'].idxmin()
]['melid'].unique()
filtered_performer_duplicates = df[df['melid'].isin(melids_to_keep)]
df = df[~df['melid'].isin(duplicates_by_performer['melid'])]
df = pd.concat([df, filtered_performer_duplicates])
print(f"After handling duplicates by performer and title: {df.shape}")

# Step 3: Handle repeated solos by 'title' but different 'performers' (keep only the oldest performance)
df['releasedate'] = pd.to_datetime(df['releasedate'], errors='coerce')
repeated_titles = df[df.duplicated(subset=['title'], keep=False)]
melids_to_keep_oldest = repeated_titles.loc[
    repeated_titles.groupby('title')['releasedate'].idxmin()
]['melid'].unique()
filtered_oldest_performances = df[df['melid'].isin(melids_to_keep_oldest)]
df = df[~df['melid'].isin(repeated_titles['melid'])]
df = pd.concat([df, filtered_oldest_performances])
print(f"After handling repeated solos by title: {df.shape}")

# Step 4: Remove rows where 'key' is missing or empty
df = df[df['key'].notna() & (df['key'] != '')]
print(f"After removing rows with missing or empty 'key': {df.shape}")

# Step 5: Separate rows with empty 'composer' for later re-integration
no_composer_df = df[df['composer'].isna() | (df['composer'].str.strip() == '')]
df_with_composer = df[~df.index.isin(no_composer_df.index)]
print(f"Rows without composer: {no_composer_df.shape[0]}")
print(f"Rows with composer: {df_with_composer.shape[0]}")

# Step 6: Standardise 'performer' and 'composer' for comparison
def standardise_name(name):
    return (
        str(name).lower()  # Convert to lowercase
        .strip()  # Remove leading/trailing whitespace
        .replace('/', '')  # Remove special characters like '/'
        .replace('.', '')  # Remove periods
    )

df_with_composer['performer_cleaned'] = df_with_composer['performer'].apply(standardise_name)
df_with_composer['composer_cleaned'] = df_with_composer['composer'].apply(standardise_name)

# Step 7: Apply composer match logic
def composer_match(row):
    # Split composer into individual names and clean them
    composer_names = [standardise_name(name) for name in re.split(r'[,/]', row['composer_cleaned'])]
    performer_name = row['performer_cleaned']
    
    # Check if the performer's name matches any of the cleaned composer names
    return any(performer_name in name or name in performer_name for name in composer_names)

df_with_composer['match_status'] = df_with_composer.apply(composer_match, axis=1)
print(f"Number of rows retained after composer match: {df_with_composer['match_status'].sum()}")

df_with_composer = df_with_composer[df_with_composer['match_status']]

# Step 8: Combine rows with matched composer and rows without composer
df_final = pd.concat([df_with_composer, no_composer_df])

# Step 9: Add the 'metrical_position' column for validation
df_final.loc[:, 'metrical_position'] = df_final[['period', 'division', 'bar', 'beat', 'tatum']].astype(str).agg('.'.join, axis=1)
print("Metrical positions have been added to the cleaned dataset.")

# Reset the index for cleanliness
df_final.reset_index(drop=True, inplace=True)

# Drop the temporary standardised columns
df_final.drop(columns=['performer_cleaned', 'composer_cleaned', 'match_status'], inplace=True)

# Save the cleaned dataset to a new CSV file
df_final.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")