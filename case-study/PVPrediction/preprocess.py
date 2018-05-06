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

workDir ='/Users/yangyangfu/github/hgvDataMining/'


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


# better to plot the figure
pvGenDF.plot()

# find the negative generation
neg = pvGenDF.index[pvGenDF[pvGenDF.columns[0]]<0].tolist()


# combine x and y
xy = pd.concat([solarDF,pvGenDF],axis=1)


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

# Need consider daylight saving time in different data recorder system. For example, 
# in SiteSage (power system), the power is recorded using daylight saving time. 
# But in Razon (solar data), they use normal local time. Therefore, in summer, 
# the power is always one hour ahead of solar data starting the daylight saving time.
# To fix the time zone gap, we need move the power in pvGenDF one hour backward during 
# daylight saving time (3/12/2017 2:00 AM to 11/5/2017 1:00 AM)

# slice dataframe to get training data and testing data
