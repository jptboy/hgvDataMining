#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:46:45 2018

@author: yangyangfu

Usage: this script is used to generate training and testing data for ANN model. Different prediction purposes probably need different preprocessing.
    this is for PV generation prediction.
"""

from sklearn.externals import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

workDir ='C:/github/hgvDataMining/'


os.chdir(workDir)

# -------------------------------------------------------------------------------
# before doing anything, it's better to plot some figures to see the data quality
# load the solar data
solarFilePath = workDir+'data/solar/minuteData/2017/'
solarFileName = '2017.csv'
#joblib.load()

solarDF = pd.read_csv(solarFilePath+solarFileName,skiprows=0,index_col=[0],parse_dates=True)

# load raw PV generation
pvFilePath = workDir + 'data/F1/minuteData/2017/'
pvFileName = '2017.csv'
pvGenDF = -pd.read_csv(pvFilePath+pvFileName, skiprows=0, index_col=[0],parse_dates=True,skipinitialspace=True,usecols=[0,2])

### ------------------------------------------------------------------------------------
# Need consider daylight saving time in different data recorder system. For example, 
# in SiteSage (power system), the power is recorded using daylight saving time. 
# But in Razon (solar data), they use normal local time. Therefore, in summer, 
# the power is always one hour ahead of solar data starting the daylight saving time.
# To fix the time zone gap, we need move the power in pvGenDF one hour backward during 
# daylight saving time (3/12/2017 2:00 AM to 11/5/2017 1:00 AM)
""" Shift data according to time zone information is done mannually in csv file """
### -------------- ------------------------------------------------------------------

# better to plot the figure
pvGenDF.plot()
        
# recover negative value by using its absolute value
pvGenDFRec = pvGenDF.abs()

# # combine x and y
xy = pd.concat([solarDF,pvGenDFRec],axis=1)

# find the negative generation
neg = pvGenDF.index[pvGenDF[pvGenDF.columns[0]]<0].tolist()

# plot the negative generation with solar data for a day
#preDay = pd.Timestamp('2017-01-01')
#for i in neg:
#    day = i.date()
#    if day!=preDay: 
#        startPlot=pd.Timestamp(day)
#        endPlot=startPlot+pd.DateOffset(days=1)
#        minuteRange = pd.date_range(startPlot,endPlot,freq='1min')
#        xyPlot = xy.loc[minuteRange]
#        xyPlot[['IrrGlobal (W/m2)','IrrDiffuse (W/m2)','IrrDirect (W/m2)','CH3-Solar Input (F1)']].plot()
#        plt.savefig(str(day)+'.png')



      
###         Training and Testing data
# use data from June and July
#
trainPeriodStart = pd.Timestamp('2017-06-01')
trainPeriodEnd = pd.Timestamp('2017-07-01')
June = xy.loc[trainPeriodStart:trainPeriodEnd][:-1] # slice the data frame by date range and drop the laste row

testPeriodStart = pd.Timestamp('2017-07-01')
testPeriodEnd = pd.Timestamp('2017-08-01')
July = xy.loc[testPeriodStart:testPeriodEnd][:-1] # slice the data frame by date range and drop the laste row

# drop NAN
June = June.dropna(axis=0,how='any')
July = July.dropna(axis=0,how='any')

joblib.dump(June,'Train-June.pkl')
joblib.dump(July,'Test-July.pkl')