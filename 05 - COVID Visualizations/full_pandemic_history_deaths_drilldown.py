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


    # Set up plot 
    fig, (a0, a1) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), gridspec_kw={'width_ratios': [9, 1]})
    for ax in (a0, a1):
        if ax == a1:
            d = pd.DataFrame(df_stat.loc['King (SEA)']).transpose()
            d.loc['King (SEA) '] = d.loc['King (SEA)']
        else:
            d = df_stat
        
        ax.scatter(
            d.index,
            d['Max New Deaths'],
            marker='_', 
            # linestyle='dashed',
            # label='Max Monthly Cases',
            color='r',
            s=400,
            alpha=0.5
            )


        ax.plot(
            d.index,
            d['Max New Deaths'],
            alpha=0.19,
            color='r',
            linestyle='--',
            drawstyle='steps-mid',
            label='Max Monthly Deaths'
            )

        ax.step(
            d.index,
            d['75th Percentile'],
            alpha=0.3,
            color='darkorange',
            label='75th Percentile',
            where='mid')

        if ax == a0:
            ax.scatter(
                d.index,
                d['Median New Deaths'],
                marker='_', 
                # linestyle='dashed',
                label='Median Monthly Deaths',
                # where='mid',
                s=750,
                alpha=0.3
                )
        else:
            ax.scatter(
                [0.5,0.5],
                d['Median New Deaths'],
                marker='_', 
                # linestyle='dashed',
                label='Median Monthly Deaths',
                # where='mid',
                s=1000,
                alpha=0.3
                )

        ax.step(
            d.index,
            d['25th Percentile'],
            alpha=0.3,
            color='green',
            label='25th Percentile',
            where='mid')

        ax.fill_between(d.index,
                        d['25th Percentile'], 
                        d['75th Percentile'],
                        step='mid',
                        color='blue',
                        alpha=0.04,
                        label='Interquartile Range')


        # plt.scatter(
        #     df_stat.index,
        #     df_stat['Min New Deaths'],
        #     marker='_', 
        #     # linestyle='dashed',
        #     label='Min Monthly Deaths',
        #     color='g',
        #     s=400,
        #     alpha=0.2
        #     )
        
        curr_month = pd.to_datetime(curr_month)
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        months[(curr_month.month - 1) - 2]

        if ax == a0:
            ax.scatter(
                data=d,
                x=d.index,
                y='Last Month',
                s=500,
                c=d['last_month_to_med'],
                # edgecolors='gray',
                vmin=-1,
                vmax=1,
                linewidths = 0.75,
                marker='$ ' + calendar.month_abbr[months[(curr_month.month - 1)]] + '$',
                label= None,
                cmap=plt.cm.get_cmap("RdYlGn_r", )
                )

            ax.scatter(
                data=d,
                x=d.index,
                y='2 Months Ago',
                marker='$ ' + calendar.month_abbr[months[(curr_month.month - 1) - 1]] + '$',
                label=None,
                color='gray',
                s=400,
                alpha=0.1
                )
        else:
            ax.scatter(
                data=d,
                x=[0.5, 0.5],
                y='Last Month',
                s=900,
                c=d['last_month_to_med'],
                # edgecolors='gray',
                vmin=-1,
                vmax=1,
                linewidths = 0.75,
                marker='$ ' + calendar.month_abbr[months[(curr_month.month - 1)]] + '$',
                label= None,
                cmap=plt.cm.get_cmap("RdYlGn_r", )
                )

            ax.scatter(
                data=d,
                x=[0.5, 0.5],
                y='2 Months Ago',
                marker='$ ' + calendar.month_abbr[months[(curr_month.month - 1) - 1]] + '$',
                label=None,
                color='gray',
                s=800,
                alpha=0.1
                )

        if ax == a0:
            for county in index_order:
                x = list(index_order).index(county)
                if d.loc[county, 'Last Month'] > d.loc[county, '2 Months Ago']:
                    mod = 1.5
                elif d.loc[county, 'Last Month'] < d.loc[county, '2 Months Ago']:
                    mod = -1.5
                else:
                    mod = 0
                y = d.loc[county, '2 Months Ago'] 
                dy = (d.loc[county, 'Last Month'] - d.loc[county, '2 Months Ago']) 
                if dy > 0:
                    color = 'darkred'
                else:
                    color='green'
                ax.arrow(
                    x=x,
                    y=y + mod, 
                    dx=0,
                    dy= dy - (3 * mod),
                    width=0.05,
                    length_includes_head=True,
                    alpha=0.15,
                    color=color,
                    head_length=1.5,
                    head_width=0.3
                    )
        else:
            x = 0
            if d.loc['King (SEA)', 'Last Month'] > d.loc['King (SEA)', '2 Months Ago']:
                mod = 0.25
            elif d.loc['King (SEA)', 'Last Month'] < d.loc['King (SEA)', '2 Months Ago']:
                mod = -0.25
            else:
                mod = 0
            y = d.loc['King (SEA)', '2 Months Ago'] 
            dy = (d.loc['King (SEA)', 'Last Month'] - d.loc['King (SEA)', '2 Months Ago']) 
            if dy > 0:
                color = 'darkred'
            else:
                color='green'
            ax.arrow(
                x=0.5,
                y=y + mod, 
                dx=0,
                dy= dy - (1.5 * mod),
                width=0.025,
                length_includes_head=True,
                alpha=0.35,
                color=color,
                head_length=0.2,
                # head_width=0.00000001
                )
                    
        


        # ax.set_yscale('log')
        a0.set_ylim((0, 200))
        a1.set_ylim((0, 15))
        ax.locator_params(axis="y", nbins=10)        
        a0.tick_params(axis="y", left=True, right=True,labelleft=True, labelright=True)
        a1.tick_params(axis="y", left=True)
        ax.tick_params(axis="x", rotation=90, labelsize=14)
        ax.xaxis.grid(True, alpha=0.18)
        ax.grid(True, which='both', axis='y', alpha=0.3)
        ax.tick_params(axis='x', colors='lightslategray')   
        a1.get_xaxis().set_ticklabels([])     
        if ax == a0:
            ax.get_xticklabels()[d.reset_index().query('county == "King (SEA)" ').index[0]].set(color="teal", fontweight='bold')         
        a0.set_ylabel('Deaths per 100k County Residents\n', fontsize=16)
        a0.set_xlabel(None)
        a0.set_title('Monthly COVID-19 Deaths: Top 30 Counties Comparison \n Data Thru: ' + calendar.month_abbr[curr_month.month] + ' ' + str(curr_month.year), fontsize=24)    
        a1.set_title('King County\nDrilldown', fontsize=24)   
        a1.text(x=0.625, y=-3, s='King County\nDrilldown', color='teal', fontsize=14, rotation=90, ha='right', fontweight='bold')

        a0.legend(fontsize=14, loc=2) 
        
    # plt.legend(fontsize=14, bbox_to_anchor=(1.01,1), borderaxespad=0)
    # plt.legend(fontsize=14, loc=2)
    plt.show()  
