# -*- coding: utf-8 -*-
"""
Created on Fri May 04 14:21:03 2018

@author: Administrator
"""
import os

workDir ='C:\\Github\\hgvDataMining\\'
data_dir = workDir+'data\\solar\\minuteData\\2017\\'

os.chdir(workDir)

import cleanTimeSeries as cts

skiprows=0
index_col=[0]
parse_dates=True
y = cts.checkColumnNames(data_dir,skiprows=skiprows,index_col=index_col,parse_dates=parse_dates)

outputPath = data_dir
outputName = '2017'
df = cts.checkMissMinuteData(data_dir,outputPath,outputName,skiprows=skiprows,index_col=index_col,parse_dates=parse_dates)   




