import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from multiprocessing import Pool as ProcessPool
from itertools import groupby
import gc
import json

def to_pickle( df , path ):
    f = open( path , 'wb' )
    pickle.dump(df , f)
    f.close()
def load_pickle( path ):
    f = open( path , 'rb' )
    df = pickle.load( f )
    f.close()
    return df

def fill_launch_seq( df ):
    def gen_launch_seq(row):
        seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
        seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
        end = row.end_date
        seq = [seq_map.get(x, 0) for x in range(end-63, end+1)]
        return seq
    df["launch_seq"] = df.apply(gen_launch_seq, axis=1)
    return df

def df_split( df , length = 10000 ):
    ls = []
    for i in tqdm( range( int(len(df) / length) + 1 ) ):
        lf = i*length
        rt = (i+1)*length
        ls.append( df.iloc[lf:rt] )
    return ls

def modifylist( df ):
    def modify( row ):
        newrow = {}
        if str(row.date_list) == 'nan':
            newrow['playtime_list'] = np.nan #row.playtime_list
            newrow['item_seq'] = np.nan
            newrow['duration_list'] = np.nan
            newrow['date_list'] = np.nan
            return newrow
        date_list = row.date_list
        newrow['playtime_list'] = row.playtime_list[ date_list < row.end_date ]
        newrow['item_seq'] = row.item_seq[ date_list < row.end_date ]
        newrow['duration_list'] = row.duration_list[ date_list < row.end_date ]
        newrow['date_list'] = row.date_list[ date_list < row.end_date ]
        return pd.Series( newrow )
    df = df.apply( modify , axis=1 )
    return df

def get_playtime( df ):
    def get_playtime_seq(row):
        try:
            seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
            seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
            seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
            seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-63, row.end_date+1)]
            return seq
        except:
            return np.nan
    df["playtime_seq"] =df.apply(get_playtime_seq, axis=1)
    return df

def get_duration( df ):
    def get_duration_prefer(duration_list):
        try:
            drn_list = sorted(duration_list.split(";"))
            drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
            if drn_map:
                max_ = max(drn_map.values())
                res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
                return res
            else:
                return np.nan
        except:
            return np.nan
    
    df["duration_prefer"] = df.duration_list.apply(get_duration_prefer)
    return df

def get_overrate( df ):
    item_time_dic  = load_pickle('item_time_dic.pkl')
    def process_row( row ):
        if isinstance( row.item_seq , float ) or isinstance( row.item_seq[0] , float ):
        #if str( row.item_seq ) == 'nan' or str( row.item_seq[0] ) == 'nan':
            overrate = np.nan
        else:
            overrate = row.playtime_list / np.array( [ item_time_dic[i] for i in row.item_seq ] )
        return overrate
    df['overrate'] = df.apply( lambda x : process_row( x ) , axis=1 )
    return df

def get_label_list(df):
    def func(row):
        uid = row.user_id
        value = ( row.launch_date , row.launch_type )
        date = []
        score = []
        if len( value[0] ) != 0:
            start = np.min( value[0] )
            
        else:
            start = row.end_date - 7 - 63
        final = row.end_date - 7
        for i in range( start , final + 1 ):
            if i + 7 > row.end_date :
                break
            date.append( i )
            end = i + 8
            score.append(  sum([1 for x in set(value[0]) if i < x < end])  )
        if len( score ) <= 64:
            row['label_date'] = [ 0 for i in range( 64 - len(date) ) ] + date
            row['label_list'] = [ 0 for i in range( 64 - len(score) ) ] + score
        else:
            row['label_date'] = date[-64:]
            row['label_list'] = score[-64:]
        return row# [ date , score ]
    df = df.apply( lambda x : func(x) , axis=1 )
    return df

def get_launch_seq( df ):
    return df.launch_seq.apply( pd.Series )

def get_playback_seq( df ):
    return df.playtime_seq.apply( 
        lambda x: json.loads(str([0]*64)) if isinstance(x,float) else x ).apply( pd.Series )

def fill_inter_seq( df ):
    def gen_launch_seq(row):
        if isinstance( row.interact_type , float ):
            seq_sort = sorted(zip([0], [0]), key=lambda x: x[1])
        else:
            seq_sort = sorted(zip(row.interact_type, row.date_inter_list), key=lambda x: x[1])
        seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
        end = row.end_date
        seq = [seq_map.get(x, 0) for x in range(end-63, end+1)]
        return seq
    df["inter_seq"] = df.apply(gen_launch_seq, axis=1)
    return df
def get_inter_seq( df ):
    return df.inter_seq.apply( pd.Series )

def seqmodifylist( df ):
    def modify( row ):
        newrow = {}
        if str(row.date_list) == 'nan':
            newrow['playtime_list'] = np.nan #row.playtime_list
            newrow['item_seq'] = np.nan
            newrow['duration_list'] = np.nan
            newrow['date_list'] = np.nan
            return newrow
        date_list = np.array( row.date_list )
        newrow['playtime_list'] = np.array(row.playtime_list)[ date_list < row.end_date ]
        newrow['item_seq'] = np.array(row.item_seq)[ date_list < row.end_date ]
        newrow['duration_list'] = np.array(row.duration_list)[ date_list < row.end_date ]
        newrow['date_list'] = np.array(row.date_list)[ date_list < row.end_date ]
        return pd.Series( newrow )
    df = df.apply( modify , axis=1 )
    return df
def seqget_playtime( df ):
    def get_playtime_seq(row):
        try:
            seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
            seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
            seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
            seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-63, row.end_date+1)]
            return seq
        except:
            return np.nan
    df["playtime_seq"] =df.apply(get_playtime_seq, axis=1)
    return df

def new_seq( df ):
    def func( row ):
        row['list'] = list( np.concatenate( 
            [ np.array( row.launch_seq ) , np.array(row.playtime_seq) , np.array( row.label_list ) ] , axis=None ) )
        return row
    df = df.apply( lambda x : func(x) , axis=1 )
    return df