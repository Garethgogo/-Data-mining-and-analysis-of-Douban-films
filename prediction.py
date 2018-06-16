import os, sys, pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler


import lightgbm as lgb
dfoff = pd.read_csv('ccf_offline_stage1_train.csv')
dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv')

dfon = pd.read_csv('ccf_online_stage1_train.csv')

def getDiscountType(row):
    row = str(row)
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    """Convert discount to rate"""

    if row != row:
        return 1.0
    row = str(row)
    if ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getReach(row):
    row = str(row)
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getMinus(row):
    row = str(row)
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def replaceDistance(row):

    if row != row:

        return -1.0
    else:
        return row


def processData(df):
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_boundary'] = df['Discount_rate'].apply(getReach)
    df['discount_minus'] = df['Discount_rate'].apply(getMinus)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['distance'] = df['Distance'].apply(replaceDistance)
    print(df['discount_rate'].unique())
    print(df['distance'].unique())
    return df


dfoff = processData(dfoff)
dftest = processData(dftest)