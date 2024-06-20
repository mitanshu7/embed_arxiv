## This script loads the full arxiv_metadata and splits them by year
import pandas as pd
import os

## Load the dataset
print("Loading the dataset...")
dataset = pd.read_parquet("kaggle/arxiv_metadata.parquet")

## Group the dataset by first 2 characters of the id column
print("Grouping the dataset by year...")
dataset["year"] = dataset["id"].str[:2]

## Create a directory to store the split metadata
print("Creating a directory to store the split metadata...")
split_dir = "arxiv_metadata_by_year"
os.makedirs(split_dir, exist_ok=True)

## Split the dataset by year
print("Splitting the dataset by year...")
for year, group in dataset.groupby("year"):

    ## if the year does not convert to int, then skip
    try:
        year = int(year)
    except:
        continue

    ## Make the year a string
    year = str(year)

    ## Pad a 0 in front of the year if the year is less than 10
    if len(year) == 1:
        year = "0" + year

    ## Save the group to a parquet file
    print(f"Saving the group for 20{year}...")
    group.to_parquet(f"{split_dir}/20{year}.parquet")

## Done
print("Done!")