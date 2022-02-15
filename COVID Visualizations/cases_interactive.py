#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:46:51 2021

@author: mattroth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go 
import plotly.express as px
import plotly.io as pio


# Import raw data from NYT Github and set date to report thru
url = 'https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv'
df = pd.read_csv(url, parse_dates=['date'])
thru_date = df['date'].loc[df['date'].dt.day.isin([30,31,28])].max() # Report thru last complete month in dataset

# Set up DF of counties, states, populations to filter whole dataset to
counties = [
    'New York City',
    'Los Angeles',
    'Cook',
    'Harris',
    'Maricopa', 
    'Philadelphia',
    'Bexar',
    'San Diego',
    'Dallas',
    'Santa Clara',
    'Travis',
    'Duval',
    'Tarrant',
    'Franklin',
    'Mecklenburg',
    'San Francisco',
    'Marion',
    'King',
    'Denver',
    'District of Columbia',
    'Suffolk',
    'El Paso',
    'Davidson',
    'Wayne',
    'Oklahoma',
    'Multnomah',
    'Clark',
    'Shelby',
    'Jefferson',
    'Baltimore city'
]
populations = [
    8336817,
    10039107,
    5150233,
    4713325,
    4485414,
    1584064,
    2003554,
    3338000,
    2635516,
    1928000,
    1273954,
    957755,
    2102515,
    1316756,
    1110356,
    881549,
    964582,
    2252782,
    727211,
    705749,
    803907,
    839238,
    694144,
    1749343,
    797434,
    812855,
    2266715,
    937166,
    766757,
    593490
    ]

states = [
    'New York',
    'California',
    'Illinois',
    'Texas',
    'Arizona',
    'Pennsylvania',
    'Texas',
    'California',
    'Texas',
    'California',
    'Texas',
    'Florida',
    'Texas',
    'Ohio',
    'North Carolina',
    'California',
    'Indiana',
    'Washington',
    'Colorado',
    'District of Columbia',
    'Massachusetts',
    'Texas',
    'Tennessee',
    'Michigan',
    'Oklahoma',
    'Oregon',
    'Nevada',
    'Tennessee',
    'Kentucky',
    'Maryland'
]

places = pd.DataFrame(
        {'county':counties,
         'state':states,
         'county_pop':populations},
        columns=['county','state','county_pop']
    )

# Merge places with df to filter on county, state, date
df = pd.merge(df, places, on=['county','state'])
df = df[ (df['date'] <= thru_date) & (df['date'] >= '2020-03-01') ]

# Replace county names with the city of interest within the county
names = [
    ('Los Angeles', 'LA'),
    ('New York City', 'NYC'),
    ('Harris', 'HOU'),
    ('Cook', 'CHI'),
    ('Maricopa', 'PHX'),
    ('Tarrant', 'Ft. Worth'),
    ('Philadelphia', 'PHL'),
    ('Baltimore city', "Baltimore"),
    ('Davidson', 'NASH'),
    ('Clark', 'LV'),
    ('Bexar', 'San Antonio'),
    ('Suffolk', 'BOS'),
    ('Franklin', 'Columbus'),
    ('Travis', 'Austin'),
    ('Jefferson', 'LVILLE'),
    ('Wayne', 'DET'),
    ('Shelby', 'MEM'),
    ('King', 'SEA'),
    ('Oklahoma', 'OKC'),
    ('Duval', 'JAX'),
    ('Marion', 'INDY'),
    ('District of Columbia', 'DC'),
    ('Multnomah', 'PDX'),
    ('Mecklenburg', 'CHAR'),
    ('El Paso', 'ELP'),
    ('Dallas', 'DAL'),
    ('Denver', 'DEN'),
    ('San Diego', 'SD'),
    ('Santa Clara', 'SJ'),
    ('San Francisco', 'SF')
    ]

for name in names:
    original, replacement = name
    if original in ['Baltimore city', 'District of Columbia']:
        df['county'] = df['county'].replace(original, replacement + ' (' + replacement + ')')
    else:
        df['county'] = df['county'].replace(original, original + ' (' + replacement + ')')

# Replace cases and deaths with per 100k residents
df['cases'] = (df['cases'] / df['county_pop']) * 100000
df['deaths'] = (df['deaths'] / df['county_pop']) * 100000

# Add month & year columns and remake df with just last day of each month's data
import calendar 
df['month'] = df['date'].dt.month.apply(lambda x: calendar.month_abbr[x])
df['year'] = df['date'].dt.year

df = pd.DataFrame(df.groupby(['county','month','year'])[['date','cases', 'deaths']].max())\
    .sort_values(['county','date'], ascending=[True, True])\
    .reset_index()


# Make index order as most cases per capita all time ascending
index_order = df.groupby('county')['deaths'].max().sort_values(ascending=True).index

df_stat_agg = pd.DataFrame()

# Iterate monthly from start of pandemic to make stat table and plots
for curr_month in list(np.sort(df['date'].unique())):
    
    # Create df of current month in for loop
    df_curr = df[df['date'] <= curr_month]

    # Calc monthly change
    df_curr['new_cases'] = (df_curr['cases'] - df_curr.groupby('county')['cases'].shift()).fillna(df_curr['cases'])
    df_curr['new_deaths'] = (df_curr['deaths'] - df_curr.groupby('county')['deaths'].shift()).fillna(df_curr['deaths'])
    df_curr['prev_month'] = df_curr.groupby('county')['new_deaths'].shift(1).fillna(0)
    df_curr['prev_month2'] = df_curr.groupby('county')['new_deaths'].shift(2).fillna(0)


    # Make summary stats table with 25th, mean, 75th percentiles
    lower_bound_deaths = pd.DataFrame(df_curr.groupby(['county'])['new_deaths'].quantile(0.25))
    upper_bound_deaths = pd.DataFrame(df_curr.groupby(['county'])['new_deaths'].quantile(0.75))
    stats_deaths = pd.DataFrame(df_curr.groupby(['county'])['new_deaths'].agg([np.max, np.min, np.median, np.mean]))
    current_deaths = df_curr.query('date == @curr_month')[['county','new_deaths']].set_index('county')
    prev_month = df_curr.query('date == @curr_month')[['county','prev_month']].set_index('county')
    prev_month2 = df_curr.query('date == @curr_month')[['county','prev_month2']].set_index('county')
    
    df_stat = pd.merge(lower_bound_deaths, upper_bound_deaths, left_index=True, right_index=True, suffixes=('low','mean'))\
                .merge(current_deaths, left_index=True, right_index=True)\
                .merge(prev_month, left_index=True, right_index=True)\
                .merge(stats_deaths, left_index=True, right_index=True)\
                .merge(prev_month2, left_index=True, right_index=True)\

    df_stat = df_stat.rename(columns={
                    'new_deathslow':'25th Percentile',
                    'new_deathsmean':'75th Percentile',
                    'new_deaths':'Last Month',
                    'prev_month':'2 Months Ago',
                    'prev_month2':'3 Months Ago',
                    'mean':'Mean New Deaths',
                    'median':'Median New Deaths',
                    'amin':'Min New Deaths',
                    'amax':'Max New Deaths'
                    }
                ).reindex(index_order)   

    # New column to compare last month to midway point between median and 75th percentile (rough measure for medium-bad status)
    # df_stat['last_month_to_med'] = (df_stat['Last Month'] - ((df_stat['Median New Cases'] + df_stat['75th Percentile']) / 2)) / ((df_stat['Median New Cases'] + df_stat['75th Percentile']) / 2)


    # New column to compare last month to median
    df_stat['last_month_to_med'] = (df_stat['Last Month'] - df_stat['Median New Deaths']) / df_stat['Median New Deaths']

    # Append curr_month to df_stat_agg
    df_stat['month'] = curr_month

    df_stat_agg = df_stat_agg.append(df_stat)

df_stat_agg['color'] = np.where(df_stat_agg['Last Month'] >= df_stat_agg['2 Months Ago'], 'darkred', 'darkgreen')


def make_graph_objects(m):
    max_dot = go.Scatter(
            x=df_stat_agg[df_stat_agg['month'] == m].index,
            y=df_stat_agg[df_stat_agg['month'] == m]['Max New Deaths'],
            mode='markers',   
            marker=dict(symbol='line-ew-open', color='red', size=10),
            name='Max Monthly Deaths'
        )
    max_line = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['Max New Deaths'],
                mode='lines',   
                line_shape='hvh',
                line=dict(color='red', width=0.25),
                opacity=0.6,        
                showlegend=False
            )
        
    iqr_h = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['75th Percentile'],
                mode='lines',
                line_shape="hvh",   
                line=dict(color='darkorange', width=1),     
                name='75th Percentile'
            )
    iqr_l = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['25th Percentile'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(142,142,142, 0.2)',        
                line_shape="hvh",
                line=dict(color='green', width=1),
                name='25th Percentile'
            )
        
    median = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['Median New Deaths'],
                mode='markers',
                line_shape="vh",        
                marker=dict(symbol='line-ew-open', color='blue', size=10),
                name='Median Monthly Deaths'
            )    

    curr_month = pd.to_datetime(m)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    months[(curr_month.month - 1) - 2]

    last_month = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['Last Month'],
                mode='text',     
                text=calendar.month_abbr[months[(curr_month.month - 1)]],        
                textfont=dict(size=8),        
                # marker=dict(
                #     size=12,     
                #     line=dict(width=10),       
                #     symbol='line-ew-open',
                #     color=df_stat_agg[df_stat_agg['month'] == m]['last_month_to_med'],
                #     colorscale='rdylgn_r',             
                #     opacity=0.2
                # ),
                showlegend=False
            )


    prev_month = go.Scatter(
                x=df_stat_agg[df_stat_agg['month'] == m].index,
                y=df_stat_agg[df_stat_agg['month'] == m]['2 Months Ago'],
                mode='text',     
                text=calendar.month_abbr[months[(curr_month.month - 2)]],
                textfont=dict(size=8),   
                opacity=0.5, 
                showlegend=False    
            )
    
    bars = go.Bar(
            x=df_stat_agg[df_stat_agg['month'] == m].index,
            y=df_stat_agg[df_stat_agg['month'] == m]['Last Month'] - df_stat_agg[df_stat_agg['month'] == m]['2 Months Ago'],
            base=df_stat_agg[df_stat_agg['month'] == m]['2 Months Ago'],
            marker_color=df_stat_agg[df_stat_agg['month'] == m]['color'],
            opacity=0.15,
            width=0.6,
            showlegend=False            
        )

    return max_dot,max_line,iqr_h,iqr_l,median,last_month,prev_month,bars


import plotly.io as pio
pio.renderers.default = "browser"

min_month = df_stat_agg['month'].min()

# fig = go.Figure()
initial = make_graph_objects(min_month)

# create a layout with out title 
layout = go.Layout(title_text="Title")

# combine the graph_objects into a figure
fig = go.Figure(data=[initial[0],initial[1],initial[2],initial[3],initial[4],initial[5], initial[6], initial[7]])

fig.data[0].visible = True

# create a list of frames
frames = []
# create a frame for every line y
for m in list(df_stat_agg['month'].sort_values().unique()):
    
    new_month = make_graph_objects(m)
    
    # create the button
    button = {
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 800}}],
            }
        ],
    }
    # add the button to the layout and update the 
    # title to show the gradient descent step
    layout = go.Layout(updatemenus=[button], 
                       title_text=f"Month {m}")
    # create a frame object
    frame = go.Frame(
        data=[new_month[0],new_month[1],new_month[2],new_month[3],new_month[4],new_month[5],new_month[6],new_month[7]], 
        layout=go.Layout(title_text=f"Gradient Descent Step {m}")
    )
# add the frame object to the frames list
    frames.append(frame)

# combine the graph_objects into a figure
fig = go.Figure(data=[new_month[0],new_month[1],new_month[2],new_month[3],new_month[4],new_month[5],new_month[6],new_month[7]], 
                frames=frames,
                layout = layout)

fig.update_layout(    
    template='plotly_white',
    height=800, 
    width=1200,
    yaxis_range=[0,125]
    # legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01
    # )
)    
fig.show()

