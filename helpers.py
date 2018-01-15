
import os
import datetime
from importlib import reload
import pickle
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import *

# functions used throughout notebook
def load_data(filename, prefix):
    """Combine all three years of data for a file. Places prefix for each column"""
    years = ['s2014','s2015','s2016']
    mid_path = '/data/'

    for year in years:
        path = year + mid_path + filename + '.csv'
        df = pd.read_csv(path,parse_dates = ['Date'], infer_datetime_format=True, index_col = 0, encoding='latin1')
        if year == 's2014':
            df_m = df
        else:
            df_m = df_m.append(df)

    df_m.columns = prefix + "__" + np.array(df_m.columns)
    df_m.reset_index(drop = True, inplace = True)

    # drop all % signs from dataframe
    for col in [col for col in df_m.columns if col.endswith('%')]:
        df_m[col] = df_m[col].str.replace("%","").astype(float)
    for col in [col for col in df_m.columns if col.endswith('HR/FB')]:
        df_m[col] = df_m[col].str.replace("%","").astype(float)

    # drop fake index col
    cols = [col for col in df_m.columns if not col.endswith("#")]
    df_m = df_m[cols]

    cols = [prefix + "__" + x for x in ['Date','Team','Name']]
    cols = [col for col in df_m.columns if col in cols]
    df_m.drop_duplicates(subset = cols, inplace = True)
    return df_m

def target_percent(df):
    """Calculate percent of set that results in ten selections per day

    Args:
        df: Full set dataframe

    Returns:
        Threshold value to be passed to np.percentile
    """
    game_count = df.date.unique().shape[0]
    at_bats = df.shape[0]
    pct = 10 / (at_bats / game_count)

    return 100 - round(pct * 100, 2)

def load_analysis_set(version):
    """Load analysis data set. Processes if necessary.
    Args:
        version: int. checkpointed version of data

    Returns:
        df: Analysis dataframe
    """
    file = 'data/analysis_' + str(version) + "_.csv"

    df = pd.read_csv(file, parse_dates=['date', 'date_m1'], encoding = 'latin1')

    return df

def load_meta_data(version):
    """Load meta data dictionary. Processes if necessary.
    Args:
        version: int. checkpointed version of metadata

    Returns:
        df: Analysis dataframe
    """
    file = 'models/meta_' + str(version) + "_.pickle"
    with open(file, 'rb') as f:
        meta = pickle.load(f)
    return meta

def random_predictions(df):
    """Returns array of random ones and zeros consistent with hit rate
    Args:
        df: analysis dataframe
    """
    rate = df[df.model_set == 'train'].got_hit.mean()
    dev_size = (df.model_set == 'dev').sum()
    t = [1] * int(dev_size * rate)
    t += ([0] * (dev_size - len(t)))
    t = shuffle(np.array(t))
    return t

def rate_histogram(df, features):
    """Plot precision by percentile for a set of features in facetgrid

    Args:
        df: model dataframe. must have features and model_set column
        features: names of columns to plot
    """
    # get percentile groupings of train set for selected features
    df = df[df.model_set == 'train']
    df_long = pd.DataFrame(columns=['percentile', 'feature', 'got_hit'])
    labels = [x for x in range(1,11)]
    for feature in features:
        bins = pd.cut(df[feature], bins=10, labels=labels)
        df_temp = pd.DataFrame({'percentile': bins,
                                'feature': [feature] * len(bins),
                                'got_hit': df.got_hit.tolist()})
        df_long = pd.concat([df_long, df_temp])

    # get average hit rate
    df_long['got_hit'] = pd.to_numeric(df_long.got_hit)
    df_long['percentile'] = pd.to_numeric(df_long.percentile)
    df_long = df_long.groupby(['feature', 'percentile']).mean().reset_index()

    # plot as facet grid
    g = sns.FacetGrid(df_long, col="feature", col_wrap=3)
    g = g.map(plt.plot, "percentile", "got_hit", marker = ".")
    g.set_titles("{col_name}")


def impute_na(df, strategy = 'mean'):
    """Fills na with mean of numeric fields
    Args:
        df: analysis dataframe
        strategy: how to fill na. see sklearn imputer documentation

    Returns:
        filled dataframe
    """
    imp = Imputer()
    numeric = ['int64', 'float64']
    col_num = [x for x,y in df.dtypes.iteritems() if y in numeric]
    col_other = [x for x,y in df.dtypes.iteritems() if y not in numeric]
    df_filled = df[col_num]
    df_filled = pd.DataFrame(imp.fit_transform(df_filled), columns = df_filled.columns, index = df.index)
    df_filled = pd.concat([df[col_other], df_filled], axis =1)
    return df_filled

def train_model(df, model, features):
    """Trains an sklearn model and returns predictions
    Args:
        df: main analysis dataframe
        model: sklearn model
        features: column names of features
    Returns:
        Tuple of probability preds, fitted model
    """
    m = df.model_set == 'train'
    model.fit(df.loc[m,features], df[m]['got_hit'])
    m = df.model_set == 'dev'
    dev_probs = model.predict_proba(df.loc[m,features])[:,1]
    return dev_probs, model

def pick_features(df,
                  prefix,
                  meta_file,
                  n_features = 2,
                  threshold = 75,
                  interactions = True):
    """
    Picks features from a set that result in best logistic regression precision for each player.

    Args:
        df: analysis dataset
        prefix: string prefix of feature set column names.
        meta_file: dict. model metadata dictionary where feature selections are stored
        n_features: int. number of features to include from the feature set
        threshold: int 0-100. probability threshold in which to calculate precision
        interactions: bool. should interaction effects be considered when calculating precision.
    """
    choices = [col for col in df.columns if col.startswith(prefix)]
    feature_combos = list(combinations(choices, n_features))
    for i, player_id in enumerate(meta_file.keys()):
        print("Calculating Model {}".format(i), end="")
        best_score = 0

        # select rows for player
        c1 = df.player_id == player_id
        c2 = df.model_set == 'train'
        df_train = df[c1 & c2]
        df_train = df_train.copy()
        c2 = df.model_set == 'dev'
        df_dev = df[c1 & c2]
        df_dev = df_dev.copy()

        # iterate through feature combinations
        for features in feature_combos:

            # select features
            if interactions:
                poly = PolynomialFeatures(interaction_only=True)
                df_train_f = poly.fit_transform(df_train[list(features)])
                df_train_f = pd.DataFrame(df_train_f)
                df_dev_f = poly.fit_transform(df_dev[list(features)])
                df_dev_f = pd.DataFrame(df_dev_f)
                df_train_f['got_hit'] = list(df_train['got_hit'])
                df_dev_f['got_hit'] = list(df_dev['got_hit'])
            else:
                df_train_f = df_train[list(features) + ['got_hit']].copy()
                df_dev_f = df_dev[list(features) + ['got_hit']].copy()

            # fit logistic regression and get prediction probabilities
            feat_w_interact = [col for col in df_train_f.columns if col != 'got_hit']
            lgr = LogisticRegression()
            lgr.fit(df_train_f[feat_w_interact], df_train_f['got_hit'])
            dev_probs = lgr.predict_proba(df_dev_f[feat_w_interact])[:,1]

            # get precision
            thresh = np.percentile(dev_probs, threshold)
            y_true = df_dev_f.got_hit
            y_pred = dev_probs >= thresh
            precision = metrics.precision_score(y_true = y_true,
                                                y_pred = y_pred)

            if precision > best_score:
                best_score = precision
                best_features = list(features)

        print("," + str(best_features))
        meta_file[player_id]['features'] += best_features

    return meta_file

def construct_models(df, meta_file, threshold = 80, export_path = None):
    """Builds ensemble model for each player based on features in meta file

    Args:
        df: analysis dataset
        meta_file: dict. model metadata dictionary where feature selections are stored.
        threshold: int 0-100. probability threshold in which to calculate precision
        export_path: folder where models should be exported

    Returns:
        tuple of analysis df with predictions concatenated, and metafile
    """
    df_w_probs = pd.DataFrame(columns=(list(df.columns) + ['probs', 'model_precision']))
    for i, player_id in enumerate(meta_file.keys()):

        # filter by player
        p_name = meta_file[player_id]['name']
        print("Calculating Model {} - {}".format(i, p_name), end="")
        c1 = df.player_id == player_id
        df_player = df[c1].copy()

        # setup voting classifier model
        lgr = LogisticRegression()
        rfc = RandomForestClassifier(n_estimators=75, max_depth=4, max_features=.7)
        nn = MLPClassifier(activation='tanh', hidden_layer_sizes=(100, 2))
        vc = VotingClassifier(estimators=[('lgr',lgr), ('rfc',rfc), ('nn',nn)],
                              voting='soft', n_jobs=-1)

        # fit and get predictions
        m = df_player.model_set == 'train'
        features = meta_file[player_id]['features']
        vc.fit(df_player.loc[m,features], df_player[m]['got_hit'])
        probs_all = vc.predict_proba(df_player.loc[:,features])[:,1]

        # standardize probs according to percentile
        # concat to df
        probs_all = pd.Series(probs_all, index=df_player.index)
        c1 = df_player.model_set == 'train'
        c2 = df_player.model_set == 'dev'
        probs_for_percentile = probs_all[(c1 | c2)].tolist()
        meta_file[player_id]['probs'] = probs_for_percentile
        probs = [percentileofscore(probs_for_percentile, x) for x in probs_all]
        df_player['probs'] = probs


        # get precision score
        dev_probs = probs_all[c2]
        thresh = np.percentile(dev_probs, threshold)
        y_true = df_player.loc[c2, 'got_hit']
        y_pred = dev_probs >= thresh
        precision = metrics.precision_score(y_true = y_true,
                                            y_pred = y_pred)
        print(", Precision = {}".format(str(precision)))
        df_player['model_precision'] = precision

        # concat to export df
        df_w_probs = pd.concat([df_w_probs, df_player])

        # export sklearn vc
        if export_path:
            fname = export_path + str(player_id) + '.pickle'
            with open(fname , 'wb') as f:
                pickle.dump(vc, f)
    return df_w_probs, meta_file

not_features = ['date','home','matchup','opp_pitcher','opp_pitcher_lefty',
                'opp_team','own_pitcher','own_pitcher_lefty','team', 'date_m1',
                'team_fg','opp_team_fg','got_hit','name','model_set','fs_id',
                'player_id', 'hit_pct_meta']
