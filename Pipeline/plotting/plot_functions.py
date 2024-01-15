"""File containing all functions to plot graphs"""

import matplotlib.pyplot as plt
from matplotlib import colors
from shapely import wkt
import geopandas as gpd
import pandas as pd
import numpy as np



def gemeente_lader(path ='..\\..\\Support\\gemeentegrenzen.csv'):
    
    """
    Function loads in csv with the geographical coordinates of every municipality in the netherlands
    and converts it to the right datatype. 
    """
    
    # Loading in the geographical data
    gemeentes = pd.read_csv(path)
    gemeentes['geometry'] = gemeentes['geometry'].apply(wkt.loads)
    #gpd_gem = gpd.GeoDataFrame(gemeentes, crs='epsg:4326')
    
    return gemeentes


def map_plotter(df_real, df_synth, gemeentes, frame, column='<indicate data column>', zip_code='<zip_code column>'):
    
    """
    Function plots 2 graphs of the netherlands using the zipcode 4 in the data + a given additional column
    
    df_real: Pandas dataframe with the trainings data
    df_synth: Pandas dataframe with the synthetic data or holdout data
    gemeentes: Pandas dataframe with municipality names and coordinates 
    frame: Name of the synthetic data or holdout data used to name the plot.
    column: Column which mean gets plotted on the map.  
    zip_code = name of the column which contains the zip4
    """
    
       
    df_real = df_real[df_real[zip_code].map(df_real[zip_code].value_counts()) > 25]
    df_real = df_real.groupby([zip_code], as_index=False)[column].mean()
    df_plot = df_real.merge(gemeentes, how='left', left_on=zip_code, right_on='PC6')
    df_plot = gpd.GeoDataFrame(df_plot, crs='epsg:4326')
    
    df_synth = df_synth[df_synth[zip_code].map(df_synth[zip_code].value_counts()) > 25]
    df_synth = df_synth.groupby([zip_code], as_index=False)[column].mean()
    df_synth_plot = df_synth.merge(gemeentes, how='left', left_on=zip_code, right_on='PC6')
    df_synth_plot = gpd.GeoDataFrame(df_synth_plot, crs='epsg:4326')
    
    tick_max = df_real[column].max()
    tick_min = df_real[column].min()
    tick_center = df_real[column].mean()
    divnorm = colors.TwoSlopeNorm(vmin=tick_min, vcenter=tick_center , vmax=tick_max)
    c_mapc = 'RdBu'
    
    
    cbar = plt.cm.ScalarMappable(norm=divnorm, cmap=c_mapc)
    
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('real')
    ax2.title.set_text(frame)
    
    df_plot.plot(column=column, 
                figsize= (10,8),
                aspect=1,
                legend=True,
                norm=divnorm,
                cmap=c_mapc, 
                ax=ax1)
    
    df_synth_plot.plot(column=column, 
                figsize= (10,8),
                aspect=1,
                legend=True,
                norm=divnorm,
                cmap=c_mapc, 
                ax=ax2)
    
    plt.close()
    return fig


def distribution_comparison(df1, df2, column, name, step_split=4):
    """
    Function shows a comparsion of the distribution of two columns. 
    df1, df2: 2 dataframes of the same shape
    column: the column of the dataframes that is going to be compared.
    """
    steps = int(df1[column].nunique() /  step_split)
    bins = np.linspace(df1[column].min(), df1[column].max(), steps)
    
    #print(df1[column].min(), df1[column].max(), df1[column].nunique())

    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('real')
    ax2.title.set_text(name)
    ax3.title.set_text('both')

    ax1.hist(df1[column], bins, color='b', edgecolor='k', linewidth=1)
    ax2.hist(df2[column], bins, color='b', edgecolor='k', linewidth=1)
    ax3.hist(df1[column], bins, alpha=0.5, label='real')
    ax3.hist(df2[column], bins, alpha=0.5, label='synth')

    fig.suptitle('Comparison of distributions')
    
    plt.close()
    return fig