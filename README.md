# Embed Arxiv

## Overview

The Embed Arxiv is a tool designed to suggest relevant scientific articles based on the user's interests. This project involves downloading metadata from ArXiv, generating vector embeddings for the articles using an embedding model, and employing cosine similarity to recommend similar articles.

## Features

- Metadata Download: Collect metadata for scientific articles from ArXiv.
- Embedding Generation: Use a pre-trained embedding model to generate vector representations for the articles.
- Cosine Similarity Calculation: Compute cosine similarity between article vectors to find and recommend relevant articles using [Milvus](https://milvus.io/) as the vector databse. 

## Installation

### Disclaimer
Tested on Linux with CUDA 12, and the requirements/environment file follow the same.

### Prerequisites

Satisfy prerequisites by issuing the command for your choice of package manager
`pip install -r requirements.txt`
or
`conda env create -f environment.yml`

## Usage

Run the files in the following order:
1. `download_arxiv_metadata.py` to download the metadata from kaggle
2. `process_metadata.py` to convert json to parquet file for efficient computations.
3. `split_metadata_by_year.py` to split the metadata by year for batched computations and multiprocessing speedup.
4. `embed_abstract.py` to embed the abstract of the papers using the `Alibaba-NLP/gte-base-en-v1.5` model.
5. `combine_abstract_embeddings.py` to join all the split files into one.
6. `transform_abstract_embeddings_milvus.py` to generate files that milvus understands.

### Extras
1. `embed_all_mxbai_embed_large_v1.py` Embeds Title, Abstract, and Full-text article.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## TO-DO
- [ ] Embed title and full text articles created from [mitanshu7/scientific_dataset_arxiv](https://github.com/mitanshu7/scientific_dataset_arxiv) using [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (Max Tokens=8192, Embedding dimensions=768).

- [ ] Embed title, abstract, full text articles created from [mitanshu7/scientific_dataset_arxiv](https://github.com/mitanshu7/scientific_dataset_arxiv) using [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) (Max Tokens=8192, Embedding dimensions=1024) for its larger embedding dimensions.

- [ ] Create a website for recommending scientific articles using these embeddings. 

## Results:
1. Find **Abstract embedded dataset** till mid-2024 here: [bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5](https://huggingface.co/datasets/bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5) using model [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (Max Tokens=8192, Embedding dimensions=768).
2. Find **Title, Abstract, Full-text article embedded dataset** here: [bluuebunny/embedded_arxiv_dataset_by_year_mxbai-embed-large-v1](https://huggingface.co/datasets/bluuebunny/embedded_arxiv_dataset_by_year_mxbai-embed-large-v1) using model [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (Max Tokens=512, Embedding dimensions=1024).