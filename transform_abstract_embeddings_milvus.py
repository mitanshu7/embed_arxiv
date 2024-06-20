## This script transforms the abstract embeddings into a format that can be ingested into Milvus
from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
import pandas as pd
from time import time

# Track the time
start = time()

# You need to work out a collection schema out of your dataset.
print("Creating the schema...")
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True
)

# Add the fields to the schema
print("Adding fields to the schema...")
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

# Verify the schema
print("Verifying the schema...")
print(schema.verify())

# Create a writer
print("Creating a writer...")
writer = LocalBulkWriter(
    schema=schema,
    local_path='.',
    segment_size=256 * 1024 * 1024, 
    file_type=BulkFileType.PARQUET
)

# Read the combined file
print("Reading the combined file...")
df = pd.read_parquet('arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5/combined.parquet')

# Loop through the dataframe
print("Looping through the dataframe...")
for i in range(len(df)):

    # Create an empty row (Dictionary) to store the data
    row = {}

    # Get the row from the dataframe
    df_row = df.iloc[i].to_dict()

    # Add the data to the row
    row["vector"] = df_row["abstract_embedding"]
    row["article_id"] = df_row["id"]

    # Add the row to the writer
    writer.append_row(row)

# Commit the writer    
print("Committing the writer...")
writer.commit()

# Print the data path
print("Data path:")
print(writer.data_path)

# Track the time
end = time()
print(f"Time taken: {end - start} seconds")

print("Done!")