#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:06:30 2019

@author: Flo
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor

from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from datetime import datetime as dt
from astral import Astral
from sklearn.preprocessing import PolynomialFeatures
from astral import Astral

a = Astral()
city = a['Madrid']
city.elevation = 0


#%%

co = pd.read_csv('/Users/flo/Documents/IE/Data Science/Consumptions.csv', sep=';')
tmp = pd.read_csv('/Users/flo/Documents/IE/Data Science/Temperatures.csv', sep=';')

#tmp2 = pd.read_csv('/Users/Flo/Documents/IE/Temperatures2.csv', sep=';')
co = co[co['Rate'] != '3.0A']
#%%

# Analysing Consumption data
#co[['Date', 'Value']].groupby('Date').agg(sum).plot() #Sum of consumptions by date
#co[['Date', 'Value']].groupby('Date').agg("mean").plot() #average of consumption by date

#co[['Month', 'Value']].groupby('Month').agg("mean").plot() #average consumption for each month
#co[['Weekday', 'Value']].groupby('Weekday').agg("mean").plot() #average consumption for each weekday
#co[['Hour', 'Value']].groupby('Hour').agg("mean").plot() #average consumption for each hour

#%%

#Describing Relation between Temperature and Consumption
cor = co[['Date', 'Value']].groupby(['Date']).agg("mean").reset_index() #grouping by date, to get average consumption for each day, then re-indexing
cortmp = pd.merge(cor, tmp, how='outer', on=['Date']) #merging both datasets, comparing data for each date
#cortmp.describe
tmpvalue = cortmp[['Date', 'Value', 'tMean']].groupby(['Date']).agg("mean").reset_index()
plt.scatter(tmpvalue['tMean'], tmpvalue['Value']) #Consumption vs temperature --> You consume more when its cold or really hot (air con or heating)


#%%Merge of Datasets


tempsco_full = pd.merge(co,tmp,how='inner',on=['Date'])
#tempsco_full = tempsco_full.loc[tempsco_full['Rate'] == '2.0DHA']


new = tempsco_full[['Date', 'Hour', 'Value']].groupby(['Date','Hour']).agg('mean').reset_index()
cups = tempsco_full[['Date', 'CUPS']].groupby(['Date'])['CUPS'].nunique().reset_index()
avgco = tempsco_full[['Date', 'Hour', 'Value']].groupby(['Date','Hour']).agg("mean").reset_index()

#tempsco_new1 = tempsco_full.drop('Value', 1)
#tempsco_new2 = tempsco_new1.drop('Hour', 1)


new_co = pd.merge(new,tmp,how='inner',on=['Date'])
avg_co = pd.merge(avgco,tmp,how='inner',on=['Date'])
cups_co = pd.merge(new_co,cups,how='inner',on=['Date'])
cups_co['Weekday'] = pd.DatetimeIndex(cups_co['Date']).weekday
cups_co['Month'] = pd.DatetimeIndex(cups_co['Date']).month
cups_co['Weekend'] = cups_co['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)



new_co['Hour'] = new_co['Hour'].astype(str)
avg_co['Hour'] = new_co['Hour'].astype(str)
#cups_co['Hour'] = cups_co['Hour'].astype(str)
#cups_co['Weekday'] = cups_co['Weekday'].astype(str)
cups_co['Day'] = pd.DatetimeIndex(cups_co['Date']).day

cups_co[['Day', 'Value']].groupby('Day').agg("mean").plot()
#
#
#tempsco_full['Year'] = pd.DatetimeIndex(tempsco_full['Date']).year
#tempsco_full['Month'] = pd.DatetimeIndex(tempsco_full['Date']).month
#tempsco_full['Weekday'] = pd.DatetimeIndex(tempsco_full['Date']).weekday


#co[['Hour', 'Value']].groupby('Hour').agg(sum)

#Combining Temperature Datasets
#tmps = pd.concat([tmp,tmp2])

#Merging consumption and temperature into one dataset
tempsco = tempsco_full.drop('CUPS', 1)

#tempsco['Date']= tempsco['Date'].map(dt.datetime.toordinal)
#%%

def calcLightHours(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour + sun['sunset'].minute / 60) , (sun['sunrise'].hour + sun['sunrise'].minute / 60)
cups_co['LightHours'] = cups_co['Date'].map(calcLightHours)

def calcLightHours_Sunrise(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunrise'].hour)
cups_co['Sunrise'] = cups_co['Date'].map(calcLightHours_Sunrise)


def calcLightHours_Sunset(date):
    date = dt.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour)
cups_co['Sunset'] = cups_co['Date'].map(calcLightHours_Sunset)

cups_co['hourlight'] = cups_co['Sunset'] - cups_co['Sunrise']


#%%
cups_co['Light'] = np.where((cups_co['Hour']>=cups_co['Sunrise']) &( cups_co['Hour']<cups_co['Sunset']), 1, 0)

cups_co['Hour'] = cups_co['Hour'].astype(str)



#%%


#sns.heatmap(cups_co.corr())

#%%

hour1 = cups_co.loc[cups_co['Hour'] == '1']
hour2 = cups_co.loc[cups_co['Hour'] == '2']
hour3 = cups_co.loc[cups_co['Hour'] == '3']
hour4 = cups_co.loc[cups_co['Hour'] == '4']
hour5 = cups_co.loc[cups_co['Hour'] == '5']
hour6 = cups_co.loc[cups_co['Hour'] == '6']
hour7 = cups_co.loc[cups_co['Hour'] == '7']
hour8 = cups_co.loc[cups_co['Hour'] == '8']
hour9 = cups_co.loc[cups_co['Hour'] == '9']
hour10 = cups_co.loc[cups_co['Hour'] == '10']
hour11= cups_co.loc[cups_co['Hour'] == '11']
hour12 = cups_co.loc[cups_co['Hour'] == '12']
hour13 = cups_co.loc[cups_co['Hour'] == '13']
hour14 = cups_co.loc[cups_co['Hour'] == '14']
hour15 = cups_co.loc[cups_co['Hour'] == '15']
hour16 = cups_co.loc[cups_co['Hour'] == '16']
hour17 = cups_co.loc[cups_co['Hour'] == '17']
hour18 = cups_co.loc[cups_co['Hour'] == '18']
hour19 = cups_co.loc[cups_co['Hour'] == '19']
hour20 = cups_co.loc[cups_co['Hour'] == '20']
hour21 = cups_co.loc[cups_co['Hour'] == '21']
hour22 = cups_co.loc[cups_co['Hour'] == '22']
hour23 = cups_co.loc[cups_co['Hour'] == '23']
hour24 = cups_co.loc[cups_co['Hour'] == '24']

#%%

    
def houralgo2(hourdata, model, cvStrategy, scoring, name,dicti):
    #si = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
    #ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    #pipec = Pipeline([si, ohe])
    
    #Xc = X.select_dtypes(exclude=[np.number])
    #Xc = hourdata[['Weekday']] 
    #Xct = pd.DataFrame(pipec.fit_transform(Xc))
    #Xct['Weekend']= Xct[5]+Xct[6]
    #Xct = Xct.drop([0,1,2,3,4,5,6],axis=1)
 
    imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
    #scl = ('scl', StandardScaler())
    pipen = Pipeline([imp])
    
    #Xn = X.select_dtypes(include=[np.number])
    Xn = hourdata[['tMean','Light','Weekend']] 
    Xnt = pd.DataFrame(pipen.fit_transform(Xn))
    
    #X = pd.concat([Xct, Xnt], axis=1, sort=False)
    X = Xnt
    y = hourdata['Value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
    model.fit(X_train, y_train)
    #y_pred_lr = model.predict(X_test)
    
    scalar = StandardScaler()
    
    pipeline = Pipeline([('transformer', scalar), ('estimator', model)])
    y_predicted= cross_val_predict(pipeline, X, y, cv=10 )
    y_test = y_test.values.flatten()
    print(np.average(np.abs(y - y_predicted) / y)) 
    dicti[name] = (np.average(np.abs(y - y_predicted) / y)) 
    

    
#%%

def getX_full(data):
    si = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
    ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    pipec = Pipeline([si, ohe])
    
    #Xc = X.select_dtypes(exclude=[np.number])
    Xc = data[['Hour']] 
    Xct = pd.DataFrame(pipec.fit_transform(Xc))
 
    imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
    #scl = ('scl', StandardScaler())
    pipen = Pipeline([imp])
    
    #Xn = X.select_dtypes(include=[np.number])
    Xn = data[['CUPS','tMean','Light','Weekend']] 
    Xnt = pd.DataFrame(pipen.fit_transform(Xn))
    
    X = pd.concat([Xct, Xnt], axis=1, sort=False)
    return X

def getX(hourdata):
    #si = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
    #ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    #pipec = Pipeline([si, ohe])
    
    #Xc = X.select_dtypes(exclude=[np.number])
    #Xc = hourdata[['Weekday']] 
    #Xct = pd.DataFrame(pipec.fit_transform(Xc))
 
    imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
    #scl = ('scl', StandardScaler())
    pipen = Pipeline([imp])
    
    #Xn = X.select_dtypes(include=[np.number])
    Xn = hourdata[['CUPS','tMean','Light','Weekend']] 
    Xnt = pd.DataFrame(pipen.fit_transform(Xn))
    
    #X = pd.concat([Xct, Xnt], axis=1, sort=False)
    return Xnt

def getY(data):
    y = data['Value']
    return y

#%%Models

rs = 0
dicti_linear = {}
dicti_elastic = {}
dicti_lasso = {}
dicti_svm = {}
dicti_svm1 = {}
dicti_svm2 = {}
dicti_svm3 = {}


dictis = [dicti_linear,dicti_elastic,dicti_lasso,dicti_svm,dicti_svm1,dicti_svm2, dicti_svm3]

alphas = [0.001,0.01,0.1,1,10]
alphaX = 0.1

models = [
#     ['Linear', linear_model.Ridge(alpha= 10,fit_intercept=True),dicti_linear],
#     ['elastic',linear_model.ElasticNet(alpha= 0.3,max_iter= 1000),dicti_elastic],
     #['Forrest',RandomForestRegressor(n_estimators=100, n_jobs=-1)]
     #['Lasso',linear_model.Lasso(alpha=0.3, fit_intercept=True),dicti_lasso],
#     ['SVM30',svm.SVR(kernel='linear', C=1000),dicti_svm1],
     ['SVM40',RandomForestRegressor(n_estimators=100, n_jobs=-1),dicti_svm]
]

#'Random Forest', RandomForestRegressor(n_estimators=100, n_jobs=-1)
  # ['Support Vector Machine', svm.SVR(kernel='linear', C=1000)]
  
cvStrategies = [
    ['Shuffle Split', ShuffleSplit(n_splits=10, test_size=0.3, random_state=rs)],
    ['K-Fold', KFold(n_splits=6, shuffle=True, random_state=rs)]
] #Training and Splitting part

cvStrategy = cvStrategies[1]
scoring = 'r2' #Fd you will get a value between 0 and 1, the more the better
#scoring = 'neg_mean_squared_error'



def score_hours():
    hours = [hour1,hour2,hour3,hour4,hour5,hour6,hour7,hour8,hour9,hour10,hour11,hour12,hour13,hour14,hour15,hour16,hour17,hour18,hour19,hour20,hour21,hour22,hour23,hour24]
    nn = 1
    for hourdb in hours:
        print(nn)
        nn +=1
        for model in models:
            print(cvStrategy[0] + ' - ' + model[0])
            houralgo2(hourdb,model[1], cvStrategy[1], scoring,nn,model[2])
            
def evaluate_models():
    n = 1
    for dic in dictis: 
        score = 0
        for key in dic:
            score += dic[key]
        print (n)    
        print(score/24) 
        n += 1
        
        
          



