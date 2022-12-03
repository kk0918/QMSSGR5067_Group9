#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:07:08 2022

@author: Group 7
"""
from utils import *
import pandas as pd 
import numpy as np
import time
import os


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
        These should be set to False unless you added new preprocessing steps 
        which then we would need to generate new preprocessed and sentiment pickles
    """
    WRITE_NEW_PREPROCESSED_PICKLES = False
    WRITE_NEW_SENTIMENT_PICKLES = False
    NUM_OF_PROCESSES = 8

    if(WRITE_NEW_PREPROCESSED_PICKLES):
        preprocessed_df = preprocess(reviews_df, out_path, NUM_OF_PROCESSES)
    preprocessed_df = merge_pickle_dfs('preprocessed_', NUM_OF_PROCESSES, out_path)
    
    if(WRITE_NEW_SENTIMENT_PICKLES):
        sentiment_df= parallelize_write_sentiment_pickles(preprocessed_df, "cleaned_review", "vader_sentiment", out_path, sent_fun, NUM_OF_PROCESSES)
    sentiment_df = merge_pickle_dfs('sentiment_', NUM_OF_PROCESSES, out_path)



