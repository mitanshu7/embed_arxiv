## This file contains the code to embed the text data using 
## the pre-trained Alibaba-NLP/gte-base-en-v1.5 embedding model.
## We will use the multiprocessing library to parallelize the process.
## We embed the abstracts of the arxiv metadata files split by year.

# Importing the required libraries
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from glob import glob
import os
import torch.multiprocessing as mp

############################################################################################################
## Using GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

## Specify number of processes to use
## Choose according to your vram and number of cores.
## I am using 4 processes on a machine with 24 cores and 6GB vram.
## On morderate machines, you are vram bound.
num_processes = 4

## Gather all the files
print("Gathering all the files...")
split_dir = "arxiv_metadata_by_year"

files = glob(f"{split_dir}/*.parquet")
files.sort()
print(f"Number of files: {len(files)}")

############################################################################################################

## Load the model and tokenizer

### Load the model
print("Loading the model...")
model_path = "Alibaba-NLP/gte-base-en-v1.5"

model = AutoModel.from_pretrained(
    model_path, 
    trust_remote_code=True, # This is to trust the model code, if you can't use this,
    # then see https://huggingface.co/Alibaba-NLP/new-impl/discussions/2#662b08d04d8c3d0a09c88fa3
    unpad_inputs=True, # This is to unpad the inputs.
    use_memory_efficient_attention=True, # This is to use memory-efficient attention (from xformers library)
  )

# Move the model to the GPU device
print("Moving the model to the device...")
model.to(device)

## Load the tokenizer
print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

## Create embeddings directory
print("Creating embeddings directory...")
model_name = model_path.replace("/", "_")
embeddings_dir = f"arxiv_embeddings_{model_name}"
os.makedirs(embeddings_dir, exist_ok=True)

############################################################################################################
## Function to get embeddings
def get_embedding(text):
  
  """Passes text through the model and returns embeddings for text."""

  ## Pass text through the model, and extract embeddings
  with torch.autocast(device_type=device.type, dtype=torch.float16):  # or bfloat16
    with torch.inference_mode(): # This is to set the model to inference mode

      ## Tokenize the text
      encoded_text = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

      ## Move the encoded text to the device
      encoded_text = encoded_text.to(device)

      ## Pass the encoded text through the model
      model_output = model(**encoded_text)

      # Extract embedding vector from model output 
      embedding = model_output.last_hidden_state[:, 0]  

  return embedding.squeeze().cpu().numpy() # Squeeze the embedding and move it to cpu and convert to numpy array

############################################################################################################
## Function to process the file
def process_file(file):

  ## Track time
  start_file = pd.Timestamp.now()

  ## Skip if the file is already processed
  if os.path.exists(f"{embeddings_dir}/{file.split('/')[-1]}"):
    print(f"File: {file} already processed.")
    return

  ## Read the data
  data = pd.read_parquet(file)
  print(f"File: {file}")
  print(f"Data shape: {data.shape}")

  ## Create a column for embeddings
  data["abstract_embedding"] = data["abstract"].apply(get_embedding)
  print(f"Embeddings created for file: {file}")

  ## Save the data
  file_name = file.split("/")[-1]
  data.to_parquet(f"{embeddings_dir}/{file_name}")
  print(f"Saved: {embeddings_dir}/{file_name}")

  ## Track time
  end_file = pd.Timestamp.now()
  print(f"Time taken for FILE={file}: {end_file - start_file}")

############################################################################################################

## Process the files
if __name__ == '__main__':
    
    ## Track time
    start = pd.Timestamp.now()

    ## forkserver is not available on windows, use spawn instead
    # mp.set_start_method('spawn')

    ## Using forkserver for linux
    mp.set_start_method('forkserver')
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_file, files)

    ## Track time
    end = pd.Timestamp.now()

    print("Done!")

    ## Print the time taken
    print(f"TOTAL Time taken: {end - start}")

############################################################################################################