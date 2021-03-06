# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:41:18 2018

@author: Yangyang Fu
"""

'''
This script aims to read time series data and identify missing date/time
'''

import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib

# list the files in the data directory
data_dir = 'data/F1/minuteData/2017'
files = os.listdir(data_dir)

"""
Determine if the column names in each csv for the same building is consistent or not
 
"""
# determine if the channel names are consistent in all files: based on the column names in the first file

for i,fil in enumerate(files):
    currentFile = fil
    # read the file in, skipping initial rows, passing in the date/time column as an index and parsing dates.
    data = pd.read_csv(os.path.join(data_dir + os.sep + currentFile), skiprows=6, index_col = [0], parse_dates=True)
    currentColumns = data.axes[1]
    # reassign for next iteration
    if i < 1:
        baseColumns = currentColumns
        baseFile = currentFile
    
    if i > 1:
        index = []
        equal = baseColumns==currentColumns
        if not all(equal):
            index = [j for j, x in enumerate(equal) if not x]
            print ('Column names are inconsistent: '+ baseFile + ' and '+ currentFile)
            print ('Index of inconsistent columns: '+str(index+1))
            print ('Columns in '+baseFile + ' are: ' + baseColumns)
            print ('Columns in '+currentFile + ' are: ' + currentColumns)
            
"""
Determine if the samples in each period is complete - minutely data

"""       
## no need to read existing output csv file. 
outputPath = 'data/F1/minuteData/2017/'
outputName = '2017.csv'
try:
    os.remove(outputPath+outputName)
except OSError:
    pass
           
# read all scattered csv files into one csv file
dfList = []
df = []
for fil in files:
    currentFile = fil
    # read the file in, skipping initial rows, passing in the date/time column as an index and parsing dates.
    data = pd.read_csv(os.path.join(data_dir + os.sep + currentFile), skiprows=6, index_col = [0], parse_dates=True)
    dfList.append(data)
df = pd.concat(dfList)

## read tslib time stamp from data frame
dt1 = df.index[0]
dt2 = df.index[-1]
# total days in the datetime periods
days = dt2.dayofyear - dt1.dayofyear + 1

## duplicate rows full of nan
df = df.drop_duplicates()

# check if minutely data is complete
if df.shape[0] < days*24*60:
    print ("Warning: Mising data!" )
    # reindex the data frame with complete time stamp
    
    # reindex the csv file using every minute
    reindex = pd.date_range(dt1,dt2,freq="60S")   
    # the missing value is filled with NaN, altough there are some other methods available  
    fullDF = df.reindex_axis(reindex) 
    
elif df.shape[0] > days*24*60:
    print("Warning: Redundant data")
    #
    # do some thing later
    #
    fullDF = df    
else:
    fullDF = df    
# outout a big csv file
fullDF.to_csv(outputPath+outputName,encoding='utf-8')
## output a pickle file
joblib.dump(fullDF,'2017-raw.pkl')    
