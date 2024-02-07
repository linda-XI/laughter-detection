import pandas as pd
import argparse
import analysis.utils as utils
import portion as P
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--extra_laugh_file', type=str, required=True)
parser.add_argument('--new_laugh_index', type=str, required=True)
parser.add_argument('--old_laugh_index', type=str, required=True)
args = parser.parse_args()

csv_file_path = args.extra_laugh_file
new_laugh_index_path = args.new_laugh_index
old_laugh_index_path = args.old_laugh_index

with open(new_laugh_index_path, "rb") as f:
        new_index = pickle.load(f)
with open(old_laugh_index_path, "rb") as f:
        old_index = pickle.load(f)
new_laugh_index =  new_index['laugh']
old_laugh_index =  old_index['laugh']

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Print the number of rows
num_rows = len(df)
print(f"Number of rows: {num_rows}")

# Calculate the total length of laughter
total_laugh_length = df[df['type'] == 'laugh']['length'].sum()
print(f"Total length of laughter(with overlap): {total_laugh_length} seconds")

laugh_no_overlap = P.empty()
meeting_groups = df.groupby(["meeting_id"])
for meeting_id, meeting_df in meeting_groups:
        part_groups = meeting_df.sort_values("start").groupby(["part_id"])
        for part_id, part_df in part_groups:
            difference = new_laugh_index[meeting_id][part_id] - old_laugh_index[meeting_id][part_id]
            laugh_no_overlap = laugh_no_overlap | difference
            
laugh_length_no_overlap = utils.to_sec(utils.p_len(laugh_no_overlap))
print(f"Total length of laughter(without overlap): {laugh_length_no_overlap} seconds")
                