#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:07:08 2022

@author: Group 9
"""
from utils import *
import pandas as pd 
import numpy as np
import time
import os

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + '/raw_datasets/rottentomatoes-400k.csv'
    top_critics_dataset_path = os.getcwd() + '/raw_datasets/rt_top_critics.csv'
    domestic_box_office_dataset_path = os.getcwd() + '/raw_datasets/domestic_box_office_2000-2021_reduced.csv'
    rt_scores_path = os.getcwd() + '/raw_datasets/rotten_tomatoes_movies_score.csv'
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Dfs from CSVs 
    reviews_df = read_csv(rotten_tomatoes_dataset_path)
    top_critics = read_csv(top_critics_dataset_path)
    domestic_box_office = read_csv(domestic_box_office_dataset_path)
    rt_scores = read_csv(os.getcwd() + '/raw_datasets/rotten_tomatoes_movies_score.csv')

    """
        Set variables here to define whether we want to read or write new pickles
        These should be set to False unless you added new preprocessing steps 
        which then we would need to generate new preprocessed and sentiment pickles
    """
    WRITE_NEW_PREPROCESSED_PICKLES = False
    WRITE_NEW_SENTIMENT_PICKLES = False
    WRITE_NEW_MERGED_BOX_OFFICE_PICKLES = False
    WRITE_NEW_RT_SCORES_PICKLES = False
    NUM_OF_PROCESSES = 8

    if(WRITE_NEW_PREPROCESSED_PICKLES):
        preprocessed_df = preprocess_sentiment_df(reviews_df, out_path, NUM_OF_PROCESSES)
    preprocessed_df = merge_pickle_dfs('preprocessed_', NUM_OF_PROCESSES, out_path)
    
    if(WRITE_NEW_SENTIMENT_PICKLES):
        sentiment_df= parallelize_write_sentiment_pickles(preprocessed_df, "cleaned_review", "vader_sentiment", out_path, sent_fun, NUM_OF_PROCESSES)
    sentiment_df = merge_pickle_dfs('sentiment_', NUM_OF_PROCESSES, out_path)
    
    if(WRITE_NEW_MERGED_BOX_OFFICE_PICKLES):
        preprocess_box_office_df = preprocess_headers_and_movie_df(domestic_box_office, out_path, 'movie', 'box_office')
    preprocess_box_office_df = read_pickle(out_path, 'box_office')
    
    if(WRITE_NEW_RT_SCORES_PICKLES):
        rt_scores_df = preprocess_headers_and_movie_df(rt_scores, out_path, 'movie_title', 'rt_scores')
    rt_scores_df = read_pickle(out_path, 'rt_scores')
    rt_scores_df = rt_scores_df[['movie', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 'audience_count']]
    
    merged_box_office_and_original_df = sentiment_df.merge(preprocess_box_office_df, how='inner', on='movie')

    # this DF contains sentiment, box office data, and RT scores
    final_df = merged_box_office_and_original_df.merge(rt_scores_df, how='inner', on='movie')
    print("Unique movies: ",len( final_df.movie.unique() ))
