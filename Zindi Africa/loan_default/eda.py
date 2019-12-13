import pandas as pd
import numpy as np
import matplotlib as plt


df = pd.read_csv('trainperf.csv')

cols_drop = ['customerid', 'systemloanid', 'referredby']

df = df.drop(cols_drop,axis=1)

df['approveddate'] = df['approveddate'].str.split(' ', expand=True)[0]
df['approveddate'] = pd.to_datetime(df['approveddate'], infer_datetime_format=True)
df['approveddate'] = df['approveddate'].dt.dayofweek

df['creationdate'] = df['creationdate'].str.split(' ',expand=True)[0]
df['creationdate'] = pd.to_datetime(df['creationdate'],infer_datetime_format=True)
df['creationdate'] = df['creationdate'].dt.dayofweek


df['good_bad_flag'] = pd.Categorical(df['good_bad_flag'])
good_bad_categories = df.good_bad_flag.cat.categories 
df['good_bad_flag'] = df.good_bad_flag.cat.codes


df.to_csv('trainperfsafi.csv')
