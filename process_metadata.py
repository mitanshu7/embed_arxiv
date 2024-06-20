## Convert the json file to parquet file
import pandas as pd

# Load the data
print('Loading the data...')
data_file = 'kaggle/arxiv-metadata-oai-snapshot.json'
arxiv_metadata = pd.read_json(data_file, lines=True)

# ## Only save columns id, abstract
# print('Selecting columns...')
# arxiv_metadata = arxiv_metadata[['id', 'abstract']]
# print('Saving the data...')
# arxiv_metadata.to_parquet('kaggle/arxiv_metadata_id_abstract.parquet')

# Save the data to parquet file
print('Saving the data...')
arxiv_metadata.to_parquet('kaggle/arxiv_metadata.parquet')

