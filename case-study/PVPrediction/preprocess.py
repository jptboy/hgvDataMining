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
startPlot=pd.Timestamp(neg[1814].date())
endPlot=startPlot+pd.DateOffset(days=1)
minuteRange = pd.date_range(startPlot,endPlot,freq='1min')
xyPlot = xy.loc[minuteRange]
xyPlot[['IrrGlobal (W/m2)','IrrDiffuse (W/m2)','IrrDirect (W/m2)','CH3-Solar Input (F1)']].plot()


# slice dataframe to get training data and testing data
