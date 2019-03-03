#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:05:22 2019

@author: flo
"""

#%%

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import ConsumptionModelX as cm

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from datetime import datetime as dt



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold
from astral import Astral



a = Astral()
city = a['Madrid']
city.elevation = 0
#%%

co = pd.read_csv('/Users/flo/Documents/IE/Data Science/Consumptions.csv', sep=';')
tmp = pd.read_csv('/Users/flo/Documents/IE/Data Science/Temperatures.csv', sep=';')
tmp2 = pd.read_csv('/Users/flo/Documents/IE/Data Science/Temperatures2.csv', sep=';')
cts = pd.read_csv('/Users/flo/Documents/IE/Data Science/nCustomers.csv', sep=';')


cts.rename(columns={'datetime':'Date'}, inplace=True)

ctmp = pd.merge(cts, tmp2, how='outer', on=['Date'])


ctmp = ctmp.loc[np.repeat(ctmp.index.values, 24)]
ctmp = ctmp.reset_index()
ctmp = ctmp.drop('index',1)

def sethours():
    n = 1
    for i, row in ctmp.iterrows():
        value = n
        ctmp.at[i,'Hour'] = value
        n +=1
        if n == 25:
            n = 1
sethours() 



ctmp['Hour'] = ctmp['Hour'].astype(int)


def calcLightHours(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour + sun['sunset'].minute / 60) , (sun['sunrise'].hour + sun['sunrise'].minute / 60)
ctmp['LightHours'] = ctmp['Date'].map(calcLightHours)

def calcLightHours_Sunrise(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunrise'].hour)
ctmp['Sunrise'] = ctmp['Date'].map(calcLightHours_Sunrise)


def calcLightHours_Sunset(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour)
ctmp['Sunset'] = ctmp['Date'].map(calcLightHours_Sunset)

ctmp['Light'] = np.where((ctmp['Hour']>=ctmp['Sunrise']) &( ctmp['Hour']<ctmp['Sunset']), 1, 0)


ctmp['Hour'] = ctmp['Hour'].astype(str)
          

ctmp['Weekday'] = pd.DatetimeIndex(ctmp['Date']).weekday
ctmp['Weekend'] = ctmp['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#ctmp['Weekday'] = ctmp['Weekday'].astype(str)

ctmp.rename(columns={'nCUPS':'CUPS'}, inplace=True)




phour1 = ctmp.loc[ctmp['Hour'] == '1']
phour2 = ctmp.loc[ctmp['Hour'] == '2']
phour3 = ctmp.loc[ctmp['Hour'] == '3']
phour4 = ctmp.loc[ctmp['Hour'] == '4']
phour5 = ctmp.loc[ctmp['Hour'] == '5']
phour6 = ctmp.loc[ctmp['Hour'] == '6']
phour7 = ctmp.loc[ctmp['Hour'] == '7']
phour8 = ctmp.loc[ctmp['Hour'] == '8']
phour9 = ctmp.loc[ctmp['Hour'] == '9']
phour10 = ctmp.loc[ctmp['Hour'] == '10']
phour11= ctmp.loc[ctmp['Hour'] == '11']
phour12 = ctmp.loc[ctmp['Hour'] == '12']
phour13 = ctmp.loc[ctmp['Hour'] == '13']
phour14 = ctmp.loc[ctmp['Hour'] == '14']
phour15 = ctmp.loc[ctmp['Hour'] == '15']
phour16 = ctmp.loc[ctmp['Hour'] == '16']
phour17 = ctmp.loc[ctmp['Hour'] == '17']
phour18 = ctmp.loc[ctmp['Hour'] == '18']
phour19 = ctmp.loc[ctmp['Hour'] == '19']
phour20 = ctmp.loc[ctmp['Hour'] == '20']
phour21 = ctmp.loc[ctmp['Hour'] == '21']
phour22 = ctmp.loc[ctmp['Hour'] == '22']
phour23 = ctmp.loc[ctmp['Hour'] == '23']
phour24 = ctmp.loc[ctmp['Hour'] == '24']

#%%


ctmp[['Date', 'CUPS']].groupby('Date').agg('mean').plot()



#Time to predict

#• Main inputs: Consumptions.csv and Temperatures.csv (download from campus.ie.edu)
#• Goal: give an advice of how much electricity the commercial company has to buy in the market
#• Days to predict:
#• 2017-08-10 to 2017-08-20 // nC & temp
#• 2017-09-10 to 2017-09-20 // nC & temp
#• 2017-11-10 to 2017-11-20 // nC & temp
#• 2018-02-10 to 2018-02-20 // nC & temp
#• Additional data (download from campus.ie.edu):
#• nCustomers.csv: number of customers for each day from 2017-07-01 to 2018-03-01
#• Temperatures2.csv: temperatures from 2017-07-01 to 2018-03-01
#%%
def predictor_algorithm(hourdata):
#        si = ('si', SimpleImputer(missing_values=np.NaN, strategy='median'))
#        ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
#        pipec = Pipeline([si, ohe])
        
        #Xc1 = hourdata[['Weekday']] 
        #Xct1 = pd.DataFrame(pipec.fit_transform(Xc1))
         
        imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
        #scl = ('scl', StandardScaler())
        pipen = Pipeline([imp])
        
        Xn1 = hourdata[['CUPS','tMean','Light','Weekend']] 
        Xnt1 = pd.DataFrame(pipen.fit_transform(Xn1))
        
        #new_X = pd.concat([Xct1, Xnt1], axis=1, sort=False)
        return Xnt1
    
def predictor_algorithm_full(data):
        si = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
        ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        pipec = Pipeline([si, ohe])
        
        #Xc = X.select_dtypes(exclude=[np.number])
        Xc1 = data[['Hour']] 
        Xct1 = pd.DataFrame(pipec.fit_transform(Xc1))
         
        imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
        #scl = ('scl', StandardScaler())
        pipen = Pipeline([imp])
        
        Xn1 = data[['CUPS','tMean','Light','Weekend']] 
        Xnt1 = pd.DataFrame(pipen.fit_transform(Xn1))
        
        new_X = pd.concat([Xct1, Xnt1], axis=1, sort=False)
        return new_X    

#%%
        
#model = RandomForestRegressor(n_estimators=100, n_jobs=-1)  
#model= linear_model.ElasticNet(alpha=0.23, max_iter=1000)
#model= linear_model.ElasticNet(alpha=0.01, max_iter=1000)
model= svm.SVR(kernel='linear', C=2000)
#model= linear_model.Ridge(alpha=0.1, max_iter=1000)    
#model= svm.SVR(kernel='linear', C=49)  
#
#
#
#scaler = StandardScaler()  
#
#test = cups_co
#predict = ctmp
#XTEST = cm.getX_full(test)
#yTEST = cm.getY(test)
#XTEST = scaler.fit_transform(XTEST)
#model.fit(XTEST, yTEST)
#XPREDICT = predictor_algorithm_full(predict)
#XPREDICT = scaler.transform(XPREDICT)
#yPREDICT = model.predict(XPREDICT)        
##    
#predicts = pd.DataFrame(yPREDICT,columns=['Predictions'],dtype=float)
#result = pd.concat([ctmp,predicts],axis=1,sort=False).reset_index()
#result['Date'] = pd.to_datetime(result.Date)
   
    
    
#%%    
        


scaler1 = StandardScaler()  
scaler2 = StandardScaler()  
scaler3 = StandardScaler()  
scaler4 = StandardScaler()  
scaler5 = StandardScaler()  
scaler6 = StandardScaler()  
scaler7 = StandardScaler()  
scaler8 = StandardScaler()  
scaler9 = StandardScaler()  
scaler10 = StandardScaler()  
scaler11 = StandardScaler()  
scaler12 = StandardScaler()  
scaler13 = StandardScaler()  
scaler14 = StandardScaler()  
scaler15 = StandardScaler()  
scaler16 = StandardScaler()  
scaler17 = StandardScaler()  
scaler18 = StandardScaler()  
scaler19 = StandardScaler()  
scaler20 = StandardScaler()  
scaler21 = StandardScaler() 
scaler22 = StandardScaler()  
scaler23 = StandardScaler()
scaler24 = StandardScaler()  


test1 = hour1
predict1 = phour1
X1 = cm.getX(test1)
y1 = cm.getY(test1)
X1 = scaler1.fit_transform(X1)
model.fit(X1, y1)
Xnew1 = predictor_algorithm(predict1)
Xnew1 = scaler1.transform(Xnew1)
ynew1 = model.predict(Xnew1)

#%%


test2 = hour2
predict2 = phour2
X2 = cm.getX(test2)
y2 = cm.getY(test2)
X2 = scaler2.fit_transform(X2)
model.fit(X2, y2)
Xnew2 = predictor_algorithm(predict2)
Xnew2 = scaler2.transform(Xnew2)

ynew2 = model.predict(Xnew2)

#%%
test3 = hour3
predict3 = phour3
X3 = cm.getX(test3)
y3 = cm.getY(test3)
X3 = scaler3.fit_transform(X3)
model.fit(X3, y3)
Xnew3 = predictor_algorithm(predict3)
Xnew3 = scaler3.transform(Xnew3)

ynew3 = model.predict(Xnew3)

#%%
test4 = hour4
predict4 = phour4
X4 = cm.getX(test4)
y4 = cm.getY(test4)
X4 = scaler4.fit_transform(X4)
model.fit(X4, y4)
Xnew4 = predictor_algorithm(predict4)
Xnew4 = scaler4.transform(Xnew4)

ynew4 = model.predict(Xnew4)

#%%
test5 = hour5
predict5 = phour5
X5 = cm.getX(test5)
y5 = cm.getY(test5)
X5 = scaler5.fit_transform(X5)
model.fit(X5, y5)
Xnew5 = predictor_algorithm(predict5)
Xnew5 = scaler5.transform(Xnew5)

ynew5 = model.predict(Xnew5)

#%%

test6 = hour6
predict6 = phour6
X6 = cm.getX(test6)
y6 = cm.getY(test6)
X6 = scaler6.fit_transform(X6)
model.fit(X6, y6)
Xnew6 = predictor_algorithm(predict6)
Xnew6 = scaler6.transform(Xnew6)

ynew6 = model.predict(Xnew6)

#%%
test7 = hour7
predict7 = phour7
X7 = cm.getX(test7)
y7 = cm.getY(test7)
X7 = scaler7.fit_transform(X7)

model.fit(X7, y7)
Xnew7 = predictor_algorithm(predict7)
Xnew7 = scaler7.transform(Xnew7)

ynew7 = model.predict(Xnew7)


#%%

test8 = hour8
predict8 = phour8
X8 = cm.getX(test8)
y8 = cm.getY(test8)
X8 = scaler8.fit_transform(X8)

model.fit(X8, y8)
Xnew8 = predictor_algorithm(predict8)
Xnew8 = scaler8.transform(Xnew8)

ynew8 = model.predict(Xnew8)


#%%


test9 = hour9
predict9 = phour9
X9 = cm.getX(test9)
y9 = cm.getY(test9)
X9 = scaler9.fit_transform(X9)

model.fit(X9, y9)
Xnew9 = predictor_algorithm(predict9)
Xnew9 = scaler9.transform(Xnew9)

ynew9 = model.predict(Xnew9)


#%%

test10 = hour10
predict10 = phour10
X10 = cm.getX(test10)
y10 = cm.getY(test10)
X10 = scaler10.fit_transform(X10)

model.fit(X10, y10)
Xnew10 = predictor_algorithm(predict10)
Xnew10 = scaler10.transform(Xnew10)

ynew10 = model.predict(Xnew10)


#%%



test11 = hour11
predict11 = phour11
X11 = cm.getX(test11)
y11 = cm.getY(test11)
X11 = scaler11.fit_transform(X11)

model.fit(X11, y11)
Xnew11 = predictor_algorithm(predict11)
Xnew11 = scaler11.transform(Xnew11)

ynew11 = model.predict(Xnew11)



#%%

test12 = hour12
predict12 = phour1
X12 = cm.getX(test12)
y12 = cm.getY(test12)
X12 = scaler12.fit_transform(X12)

model.fit(X12, y12)
Xnew12 = predictor_algorithm(predict12)
Xnew12 = scaler12.transform(Xnew12)

ynew12 = model.predict(Xnew12)


#%%
test13 = hour13
predict13 = phour13
X13 = cm.getX(test13)
y13 = cm.getY(test13)
X13 = scaler13.fit_transform(X13)

model.fit(X13, y13)
Xnew13 = predictor_algorithm(predict13)
Xnew13 = scaler13.transform(Xnew13)

ynew13 = model.predict(Xnew13)
#%%

test14 = hour14
predict14 = phour14
X14 = cm.getX(test14)
y14 = cm.getY(test14)
X14 = scaler14.fit_transform(X14)

model.fit(X14, y14)
Xnew14 = predictor_algorithm(predict14)
Xnew14 = scaler14.transform(Xnew14)

ynew14 = model.predict(Xnew14)


#%%
test15 = hour15
predict15 = phour15
X15 = cm.getX(test15)
y15 = cm.getY(test15)
X15 = scaler15.fit_transform(X15)

model.fit(X15, y15)
Xnew15 = predictor_algorithm(predict15)
Xnew15 = scaler15.transform(Xnew15)

ynew15 = model.predict(Xnew15)



#%%
test16 = hour16
predict16 = phour16
X16 = cm.getX(test16)
y16 = cm.getY(test16)
X16 = scaler16.fit_transform(X16)

model.fit(X16, y16)
Xnew16 = predictor_algorithm(predict16)
Xnew16 = scaler16.transform(Xnew16)

ynew16 = model.predict(Xnew16)



#%%
test17 = hour17
predict17 = phour17
X17 = cm.getX(test17)
y17 = cm.getY(test17)
X17 = scaler17.fit_transform(X17)

model.fit(X17, y17)
Xnew17 = predictor_algorithm(predict17)
Xnew17 = scaler17.transform(Xnew17)

ynew17 = model.predict(Xnew17)


#%%
test18 = hour18
predict18 = phour18
X18 = cm.getX(test18)
y18 = cm.getY(test18)
X18 = scaler18.fit_transform(X18)

model.fit(X18, y18)
Xnew18 = predictor_algorithm(predict18)
Xnew18 = scaler18.transform(Xnew18)

ynew18 = model.predict(Xnew18)

#%%

test19 = hour19
predict19 = phour19
X19 = cm.getX(test18)
y19 = cm.getY(test19)
X19 = scaler19.fit_transform(X19)

model.fit(X19, y19)
Xnew19 = predictor_algorithm(predict19)
Xnew19 = scaler19.transform(Xnew19)

ynew19 = model.predict(Xnew19)

#%%

test20 = hour20
predict20 = phour20
X20 = cm.getX(test20)
y20 = cm.getY(test20)
X20 = scaler20.fit_transform(X20)

model.fit(X20, y20)
Xnew20 = predictor_algorithm(predict20)
Xnew20 = scaler20.transform(Xnew20)

ynew20 = model.predict(Xnew20)

test21 = hour21
predict21 = phour21
X21 = cm.getX(test21)
y21 = cm.getY(test21)
X21 = scaler21.fit_transform(X21)

model.fit(X21, y21)
Xnew21 = predictor_algorithm(predict21)
Xnew21 = scaler21.transform(Xnew21)

ynew21 = model.predict(Xnew21)


#%%
test22 = hour22
predict22 = phour22
X22 = cm.getX(test22)
y22 = cm.getY(test22)
X22 = scaler22.fit_transform(X22)

model.fit(X22, y22)
Xnew22 = predictor_algorithm(predict22)
Xnew22 = scaler22.transform(Xnew22)

ynew22 = model.predict(Xnew22)




#%%
test23 = hour23
predict23 = phour23
X23 = cm.getX(test23)
y23 = cm.getY(test23)
X23 = scaler23.fit_transform(X23)

model.fit(X23, y23)
Xnew23 = predictor_algorithm(predict23)
Xnew23 = scaler23.transform(Xnew23)

ynew23 = model.predict(Xnew23)


#%%

test24 = hour24
predict24 = phour24
X24 = cm.getX(test24)
y24 = cm.getY(test24)
X24 = scaler24.fit_transform(X24)

model.fit(X24, y24)
Xnew24 = predictor_algorithm(predict24)
Xnew24 = scaler24.transform(Xnew24)

ynew24 = model.predict(Xnew24)



    
predicts1 = pd.DataFrame(ynew1,columns=['Predictions'],dtype=float)
predicts1['Hour'] = 1
raw_predicts = pd.concat([cts['Date'],predicts1],axis=1,sort=False).reset_index()  
  

predictionhours=[ynew2,ynew3,ynew4,ynew5,ynew6,ynew7,ynew8,ynew9,ynew10,ynew11,ynew12,ynew13,ynew14,
                 ynew15,ynew16,ynew17,ynew18,ynew19,ynew20,ynew21,ynew22,ynew23,ynew24]    
    
nnn=2
for i in predictionhours:
    predicti = pd.DataFrame(i,columns=['Predictions'],dtype=float)
    predicti['Hour'] = nnn
    predicti = pd.concat([cts['Date'],predicti],axis=1,sort=False).reset_index()
    raw_predicts = pd.concat([raw_predicts,predicti],axis=0,sort=False)
    nnn += 1

raw_predicts['Date'] = pd.to_datetime(raw_predicts.Date)
raw_predicts = raw_predicts.sort_values(by=['Date','Hour'])

ctmp['Date'] = pd.to_datetime(ctmp.Date)
raw_predicts['Hour'] = raw_predicts['Hour'].astype(str)
result = pd.merge(ctmp,raw_predicts, how='outer', on=['Date','Hour'])


#%%

result['avg'] = result['Predictions']/result['CUPS']

#%%

#cortmp.describe
avg_tmp_predict = result[['Date', 'avg', 'tMean']].groupby(['Date']).agg("mean").reset_index()
plt.scatter(avg_tmp_predict['tMean'], avg_tmp_predict['avg']) #  

result[['Date', 'Predictions']].groupby('Date').agg(sum).plot() #Sum of consumptions by date

#%%
result['Hour'] = result['Hour'].astype(int)



result[['Weekday', 'avg']].groupby('Weekday').agg('mean').plot() 

result[['Hour', 'avg']].groupby('Hour').agg("mean").plot() 
co[['Hour', 'Value']].groupby('Hour').agg("mean").plot() 
#%%
ctmp.rename(columns={'nCUPS':'CUPS'}, inplace=True)
cups_co['Date'] = pd.to_datetime(cups_co.Date)

totalcups = pd.concat([cups_co[['Date','CUPS','Value']],result[['Date','CUPS','Predictions']]],axis=0,sort=False).reset_index()
totalcups['Value'].fillna(0, inplace=True)
totalcups['Predictions'].fillna(0, inplace=True)

totalcups['Total'] = totalcups['Value'] + totalcups['Predictions']



#%%

totalcups[['Date', 'Total']].groupby('Date').agg(sum).plot() 




#%%

ctmp[['Date', 'CUPS']].groupby('Date').agg('mean').plot()
totalcups[['Date', 'CUPS']].groupby('Date').agg('mean').plot()


#%%
#• 2017-08-10 to 2017-08-20 // nC & temp
#• 2017-09-10 to 2017-09-20 // nC & temp
#• 2017-11-10 to 2017-11-20 // nC & temp
#• 2018-02-10 to 2018-02-20 

csvexport = result[['Date','Hour','Predictions']]

csvexport1= csvexport[csvexport["Date"].isin(pd.date_range("2017-08-10", "2017-08-20"))]
csvexport2= csvexport[csvexport["Date"].isin(pd.date_range("2017-09-10", "2017-09-20"))]
csvexport3= csvexport[csvexport["Date"].isin(pd.date_range("2017-11-10", "2017-11-20"))]
csvexport4= csvexport[csvexport["Date"].isin(pd.date_range("2018-02-10", "2018-02-20"))]

csvexport = pd.concat([csvexport1,csvexport2,csvexport3,csvexport4],axis=0,sort=False)



csvexport['Date'] = csvexport['Date'].dt.strftime('%Y-%m-%d')
csvexport['Hour'] = csvexport['Hour'].astype(int)

#export_csv = csvexport.to_csv (r'/Users/flo/Documents/IE/Data Science/ConsumptionPredictionsFlorian.csv', index = None, header=False)




