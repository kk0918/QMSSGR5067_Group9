#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:07:08 2022

@author: Group 7
"""
import os
from utils import *
import pandas as pd 

# Custom helper functions not available from utils
def read_csv(file_path):
    df = pd.read_table(file_path, sep=",")
    return df

# Add preprocessing steps here
def preprocess(df_in, read_pickle_flag=False, write_pickle_flag=True):
    out_path = os.getcwd() + "/pickles/"

    if(read_pickle_flag):
        reviews_df = read_pickle(out_path, "preprocessed_reviews")
        return reviews_df
    
    # lowercase headers
    df_in.columns = [header.lower() for header in df_in.columns]
    # clean text and remove stopwords
    df_in["cleaned_review"] = df_in.review.apply(clean_text).apply(rem_sw)

    # Create a new pickle if we add additional preprocessing steps
    if (write_pickle_flag):
        write_pickle(df_in, out_path, "preprocessed_reviews")
    return df_in

def getVaderSentiment(df_in, col_name):
    df_in["vader_score"] = df_in[col_name].apply(sent_fun)
    return df_in

rotten_tomatoes_dataset_path = os.getcwd() + '/rottentomatoes-400k.csv'
top_critics_dataset_path = os.getcwd() + '/rt_top_critics.csv'

reviews_df = read_csv(rotten_tomatoes_dataset_path)
top_critics = read_csv(top_critics_dataset_path)

# Read pickle to save processing time, make sure to change flags after creating pickle
# cannot save pickles to git because the files are too large 
preprocessed_df = preprocess(reviews_df, False, True)




