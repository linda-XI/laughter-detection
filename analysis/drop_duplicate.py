import pandas as pd
import os

file_path = 'sample/extraLaughSample/extraLaughSample/'
new_path = 'sample/extraLaughSample/noduplicate/'
for filename in os.listdir(file_path):
    full_path = os.path.join(file_path, filename)
    df = pd.read_csv(full_path)
    df = df.drop_duplicates()
    df.to_csv(os.path.join(new_path, filename), index=False)