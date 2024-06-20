## This file combines all the embedded abstracts into a single file
## This is to help milvus bulkwriter to write the embeddings to the milvus database.
import pandas as pd
from glob import glob

# Gather the files
print("Gathering all the files...")
files = glob('arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5/*.parquet', recursive=True)
files.sort()

## Combine all the files
print("Combining all the files...")
df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

# Save the combined file
print("Saving the combined file...")
df.to_parquet('combined.parquet')

print("Done!")
