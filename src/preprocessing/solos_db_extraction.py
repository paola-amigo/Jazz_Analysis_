import pandas as pd
import sqlite3
import os

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
db_file = os.path.join(data_dir, 'wjazzd.db')
output_file_path = os.path.join(data_dir, 'solos_db.csv')  # Final output path

# Specify the fields to extract from each table
fields_to_extract = {
    # Other tables remain unchanged
    "melody": [
        "eventid",
        "melid",
        "onset",
        "pitch",
        "duration",
        "period",
        "division",
        "bar",
        "beat",
        "tatum",
        "subtatum",
        "num",
        "denom",
        "beatprops",
        "beatdur",
        "tatumprops"
    ],
    "solo_info": [
        "melid",
        "trackid", 
        "compid", 
        "recordid", 
        "performer", 
        "title", 
        "solopart", 
        "instrument", 
        "style", 
        "avgtempo", 
        "tempoclass", 
        "rhythmfeel", 
        "key", 
        "signature", 
        "chord_changes", 
        "chorus_count"
    ],
    "record_info": ["recordid", "releasedate"],
    "composition_info": ["compid","composer", "form", "template", "tonalitytype", "genre"],
    "track_info": ["trackid","lineup"] 
}

def connect_to_database(db_path):
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        exit(1)

def extract_table(conn, table_name, fields=None):
    """Extract data from a specific table."""
    query = f"SELECT {', '.join(fields) if fields else '*'} FROM {table_name}"
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Successfully extracted {len(df)} rows from '{table_name}' table.")
        return df
    except sqlite3.Error as e:
        print(f"Error reading table '{table_name}': {e}")
        return None

# Connect to the database
conn = connect_to_database(db_file)

# Extract the relevant tables with selected fields
extracted_tables = {
    table: extract_table(conn, table, fields)
    for table, fields in fields_to_extract.items()
}

# Close the connection
conn.close()

# Merge the tables
# Step 1: Merge melody with solo_info
merged_df_1 = extracted_tables['melody'].merge(extracted_tables['solo_info'], on='melid', how='left')

# Step 2: Merge with record_info
merged_df_2 = merged_df_1.merge(extracted_tables['record_info'], on='recordid', how='left')

# Step 3: Merge with composition_info
merged_df_3 = merged_df_2.merge(extracted_tables['composition_info'], on='compid', how='left')

# (Optional) Step 4: Merge with track_info if needed
if 'track_info' in extracted_tables:
    merged_df_final = merged_df_3.merge(extracted_tables['track_info'], on='trackid', how='left')
else:
    merged_df_final = merged_df_3

# Drop duplicated columns if they exist
merged_df_final = merged_df_final.loc[:, ~merged_df_final.columns.duplicated()]

# Save the final DataFrame to CSV
merged_df_final.to_csv(output_file_path, index=False)
print(f"Final merged data saved to {output_file_path}")