import pandas as pd
import os
data_dir = './data/raw-data'
testfiles = os.listdir(data_dir)
realfiles=[]
'''
The for loop below this comment is used to get rid of lock files in the data directory
'''
for x in testfiles:
    if (x[len(x)-4:len(x)]=='.csv'):
        realfiles.append(x)

fDframes=[]
dFrames=[]
#test=[]
for i in range(0,len(realfiles)):
    new_fulldf= pd.read_csv(os.path.join(data_dir+'/'+realfiles[i]), skiprows=6, index_col=[0],parse_dates=True)
    
    new_stackeddf=pd.DataFrame(new_fulldf.stack(),columns=['power'])#creating a new data frame that uses the stacked first data frame with the stacked column being power
    new_stackeddf.index.names=['date/time','channel']

    idx=pd.IndexSlice
    channel_name='CH1-Bldg F1'

    new_data_fullbldg_df= new_stackeddf.loc[idx[:,[channel_name]], :]
    new_data_fullbldg_df.reset_index(level = 'channel', inplace=True)
    new_data_fullbldg_df.reset_index(level = 'date/time', inplace=True)
    #test.append( new_data_fullbldg_df)
    #newtestdf=pd.Data

    new_good_df = pd.DataFrame({'date/time' : new_data_fullbldg_df.iloc[:,0], 'power_watts' : new_data_fullbldg_df.iloc[:,2] })#makes new db with columns of these and omits the channel name
    new_good_df.set_index('date/time', inplace=True)#sets index back to date

    fDframes.append(new_stackeddf)
    dFrames.append( new_good_df)
for x in range(0,len(fDframes)):
    print(fDframes[x].head(10))
    print('----------')
    print(dFrames[x].head(10))
