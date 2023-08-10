import pandas as pd
import numpy as np
from scrapy import Selector
import requests
import re
import dateutil.parser as dparser
from datetime import datetime, timedelta

# xpath that points to proper place in pages
xpath = '/html/body/div/div/div/div/div/ul[1]/li/div/span'
        
# Read in ccdi data and keywords
sel_ccdi = pd.read_csv('CCDI_Selected_Data.csv')
keywords = pd.read_csv('keywords.csv')

# Format dates in sel_ccdi
sel_ccdi['Date of Announcement'] = sel_ccdi['Date of Announcement'].apply(lambda x: dparser.parse(x, dayfirst=True, fuzzy=True).date())

# Rename columns for easier accessing in namedtuples
sel_ccdi.rename(columns={list(sel_ccdi)[1]: 'Name', list(sel_ccdi)[2]:'Date'}, inplace=True)

# Initialize empty list to store all results and counter i for ppl
completed_scrapes = []
i = 0

# Iterate through names in outer loop
for namedtuple in sel_ccdi.itertuples():
    
    # Extract name and date of announcement
    name = namedtuple.Name
    date_ccdi = namedtuple.Date

    # Initialize counter for kwds
    j = 0

    # Loop through keywords and run scrape
    for kwd in keywords['keywords']:
        
        # Initialize empty dict to store results from inner loop
        results = {}

        # Print status report
        print('Status Report: Scraping for person #' + str(i+1) + ' of ' + str(len(sel_ccdi)) + ', Keyword #' + str(j+1))

        # Construct url from parameters 
        url = 'https://search.caixin.com/newsearch/search?keyword=' +\
              name + '+AND+' + kwd +\
              '&x=0&y=0&channel=&time=&type=1&sort=1&' +\
              'startDate='+ '2000-01-01' +\
              '&endDate=' + (date_ccdi - timedelta(1)).strftime('%Y-%m-%d') +\
              '&special=false'

        # make get request to url and create selector object
        html = requests.get(url).content
        sel = Selector(text = html)

        # Go to xpath and extract relevant section
        sel_list = sel.xpath(xpath)
        
        # Only look for date if the url returns results
        if len(sel_list) != 0:
            # Convert to str and extract date
            d = re.search(r'\d{4}-\d{2}-\d{2}', str(sel_list))
            date_extract = datetime.strptime(d.group(), '%Y-%m-%d').date()
        else:
            date_extract = None

        # Add current name and keyword to results dict
        results['Name'] = name 
        results['Keyword'] = kwd 

        # Output to results dict based on logical test for date_extract being < date_ccdi and non-null
        if date_extract is None or date_extract >= date_ccdi:
            results['CCDI Date'] = date_ccdi.strftime('%Y-%m-%d')
            results['Caixin Date'] = None
            results['Caixin Earlier than CCDI'] = False
            results['N Days Earlier'] = None
            results['URL'] = None
        else:
            results['CCDI Date'] = date_ccdi.strftime('%Y-%m-%d')
            results['Caixin Date'] = date_extract.strftime('%Y-%m-%d')
            results['Caixin Earlier than CCDI'] = True
            results['N Days Earlier'] = (date_extract - date_ccdi).days
            results['URL'] = url

        # Append to completed_scrapes list and increment kwd counter
        completed_scrapes.append(results)
        j += 1
    
    # Increment ppl counter
    i += 1

# Create dataframe from completed scrapes
final_results = pd.DataFrame(completed_scrapes)

# Write to csv
final_results.to_csv('scraped_results.csv', index=False, encoding='utf_8_sig')

print('Done!')
print('Results CSV created: scraped_results.csv')  

