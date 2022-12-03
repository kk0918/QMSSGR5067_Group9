#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:07:08 2022

@author: Group 7
"""
import os
from utils import *
import pandas as pd 

def preprocess(df_in, read_pickle_flag=True):
    out_path = os.getcwd() + "/pickles/"
    
    # if read pickle flag enabled and directory for pickles exists
    if(read_pickle_flag and os.path.exists(out_path)):
        reviews_df = read_pickle(out_path, "preprocessed_reviews")
        return reviews_df
    
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
    # Create a new pickle for when we add additional preprocessing steps
    # and if pickles folder doesn't exist, make one
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    write_pickle(processed_df, out_path, "preprocessed_reviews")
    
    return processed_df

rotten_tomatoes_dataset_path = os.getcwd() + '/raw_datasets/rottentomatoes-400k.csv'
top_critics_dataset_path = os.getcwd() + '/raw_datasets/rt_top_critics.csv'

reviews_df = read_csv(rotten_tomatoes_dataset_path)
top_critics = read_csv(top_critics_dataset_path)

# Read pickle to save processing time, make sure to change flags after creating pickle
# cannot save pickles to git because the files are too large 
preprocessed_df = preprocess(reviews_df, False)




