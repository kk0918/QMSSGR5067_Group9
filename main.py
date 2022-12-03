#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:07:08 2022

@author: Group 7
"""
import os
from utils import *
import pandas as pd 
import numpy as np
import time

def preprocess(df_in, num_split):    
    processed_df = df_in.copy()
    # lowercase headers
    processed_df.columns = [header.lower() for header in processed_df.columns]
    
    # Remove words between quotes
    processed_df["cleaned_review"] = processed_df.review.apply(remove_words_between_quotes)
    # Remove title 
    processed_df["cleaned_review"] = processed_df.apply(lambda x: remove_title(x.cleaned_review, x.movie), axis=1)
    # Clean text
    processed_df["cleaned_review"] = processed_df.cleaned_review.apply(clean_text)
    # Remove stopwords
    processed_df["cleaned_review"] = processed_df.cleaned_review.apply(rem_sw)
    
    # Drop columns not needed for analysis
    processed_df = processed_df.drop('date', axis=1)
    processed_df = processed_df.drop('publish', axis=1)

    # Split data into chunks
    data_split = np.array_split(processed_df, num_split)

    for i in range(num_split):
        start = time.time()
        write_pickle(data_split[i], out_path, f'preprocessed_{i}')
        end = time.time()
    
    return processed_df

if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + '/raw_datasets/rottentomatoes-400k.csv'
    top_critics_dataset_path = os.getcwd() + '/raw_datasets/rt_top_critics.csv'
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    reviews_df = read_csv(rotten_tomatoes_dataset_path)
    top_critics = read_csv(top_critics_dataset_path)
    
    """
        Set variables here to define whether we want to read or write new pickles
    """
    WRITE_NEW_PREPROCESSED_PICKLES = False
    WRITE_NEW_SENTIMENT_PICKLES = False
    NUM_OF_PROCESSES = 8

    if(WRITE_NEW_PREPROCESSED_PICKLES):
        preprocessed_df = preprocess(reviews_df)
    preprocessed_df = merge_pickle_dfs('preprocessed_', NUM_OF_PROCESSES, out_path)
    
    if(WRITE_NEW_SENTIMENT_PICKLES):
        sentiment_df= parallelize_write_sentiment_pickles(preprocessed_df, "cleaned_review", "vader_sentiment", out_path, sent_fun, NUM_OF_PROCESSES)
    sentiment_df = merge_pickle_dfs('sentiment_', NUM_OF_PROCESSES, out_path)



