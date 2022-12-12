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

"""
    DF Key:
        preprocessed_df:
            dataframe consisting of original rotten_tomatoes 
            dataset with critics reviews and critic score. Preprocessed so the
            reviews have words between quotes removed, titles from review removed, 
            text cleaned, stopwords removed, and title lowercased. 
        sentiment_df:
            exact same as preprocessed_df except with a vader_sentiment
            column attached. Only reason this is separate was because calculating
            sentiment was extremely slow and didn't want it to be super slow 
            if we needed to change other unrelated columns.
        preprocess_box_office_df:
            dataframe from box office file. intermediate 
            df necessary to merge with sentiment_df. preprocessed to just 
            lowercase column names and lowercase movie title.
        rt_scores_df:
            dataframe from rotten tomatoes scores file. intermediate df 
            necessary to merge to create final df. preprocessed to just lowercase
            column names and lowercase movie title.
        final_rt_df:
            Final dataframe consisting of sentiment_df merged with preprocess_box_office_df
            and rt_scores_df.
"""

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
        preprocessed_box_office_df = preprocess_box_office_df(domestic_box_office, out_path, 'box_office')
    preprocessed_box_office_df = read_pickle(out_path, 'box_office')
    
    if(WRITE_NEW_RT_SCORES_PICKLES):
        rt_scores_df = preprocess_rt_scores_df(rt_scores, out_path, 'rt_scores')
    rt_scores_df = read_pickle(out_path, 'rt_scores')
    rt_scores_df = rt_scores_df[['movie', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 'audience_count']]
    
    merged_box_office_and_original_df = sentiment_df.merge(preprocessed_box_office_df, how='inner', on='movie')

    # this DF contains sentiment, box office data, and RT scores
    final_rt_df = merged_box_office_and_original_df.merge(rt_scores_df, how='inner', on='movie')
    print("Unique movies: ",len( final_rt_df.movie.unique() ))
