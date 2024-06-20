## Convert the json file to parquet file
import pandas as pd

# Load the data
data_file = 'kaggle/arxiv-metadata-oai-snapshot.json'
arxiv_metadata = pd.read_json(data_file, lines=True)

# ## Only save columns id, abstract
# arxiv_metadata = arxiv_metadata[['id', 'abstract']]

# Save the data to parquet file
arxiv_metadata.to_parquet('arxiv_metadata.parquet')

