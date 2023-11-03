from analysis.transcript_parsing import parse
import pandas as pd

parse.laugh_only_df.to_csv('./laugh_only_df.csv', index=False)
