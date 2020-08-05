# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:20:12 2020

@author: Ugur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
 
figDir = "Figures/" 
#%%
from sklearn import preprocessing

def split(): print("\n____________________________________________________________________________________\n")

#Tüm featureler için korelasyon matrisi
def plotCorrelationMatrix(df, graphWidth):  
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for xd', fontsize=40)
    plt.show()
#Boxplot, gruplama ve korelasyonların hepsini analiz eden all-in-one fonksiyon
    plt.savefig(figDir+'CorrelationMatrix.png')
def intro(df,graph=True,splitPlots=True,EraseNullColumns=False,printCorrelations=True,corrThreshold=0.5):
    
    dataframe=df.copy()
    
    if(EraseNullColumns==True):  dataframe.dropna(axis=1,inplace=True)

    split()
    print(df)
    split()
    print(dataframe.head(5))
    split()
    
    print(dataframe.info())
    split()
    
    print(dataframe.describe())
    split()
    
#-------------------------------BOXPLOTFEATURES-----------------------------      
    
    
    if(graph):

        if(splitPlots==True):
            print("                         ___BOXPLOTFETURES")

            for column in dataframe.columns:
                if(dataframe[column].dtype==np.int or dataframe[column].dtype==np.float):
                    plt.figure()
                    dataframe.boxplot([column])
                    plt.savefig(figDir+'{}.png'.format(column))
                    
        else:
            dataframe.boxplot()
            
    #If unique values of columns is under 10, print unique values with considered column


#-------------------------------GROUPBY-----------------------------        

    print("                         _____GROUPBY____")

    for column in dataframe.columns:    
        unique_values=dataframe[column].unique()
        if(unique_values.size<=10):
            print(column,": ",unique_values)
            print("\nGrouped By: ",column,"\n\n",dataframe.groupby(column).mean())
            split()
            print("\n")
            
        
#-------------------------------CORRELATIONS-----------------------------        
    if(printCorrelations==True):
        print("                         ____CORRELATIONS____")
        corrByValues= dataframe.corr().copy()
        flag = False
        corr_matrix=abs(corrByValues>=corrThreshold)
        columns= corr_matrix.columns
        for i in range(columns.size):
            for j in range(i,columns.size):
                iIndex=columns[i]
                jIndex=columns[j] 
                if (i!=j and corr_matrix[iIndex][jIndex]==True and (len(df[iIndex].unique())!=1 and len(df[jIndex].unique())!=1 )):
                    sign = "Positive"
                    if(corrByValues[iIndex][jIndex]<0): sign="Negative"
                    split()
                    flag = True
                    print(iIndex.upper(), " has a " ,sign," correlation with ",jIndex.upper(),": {} \n".format(corrByValues[iIndex][jIndex]))
        
        plt.show()
        plotCorrelationMatrix(df,30)       
        
        split()
        if(not flag):
            print("No Correlation Found") 
    return dataframe

#KDE dağılımı ile featureları plotlar
def plotCols(df,time):
    
    for col in df.columns:
        if(df[col].dtype==np.int or df[col].dtype==np.float):
            if(len(df[col].unique())>1):
                fig = df.plot(x=time,y=col,kind="kde", title = "{}-{} KDE".format(time,col))    
                fig.get_figure().savefig(figDir+"{}-kde.png".format(time+"-"+col))
                plt.show()
            plt.plot(df[time],df[col]) 
            plt.title("{}-{}".format(time,col))
            plt.show()
            plt.savefig(figDir+'{}.png'.format(time+"-"+col))
        
#Verilen feature'ları scatter ile Y'ye göre karşılaştırır.        
def XCorrWithY(df, X, Y):
    for col in  X:
        print(col,"-",Y)
        plt.scatter(df[col],df[Y]) 
        plt.title("{}-{}".format(col,Y))
        plt.show()    
#Dataframeyi normalize eder. (Preprocessing)        
def normalizedf(df,offset=0):
    min_max_scaler = preprocessing.MinMaxScaler() 
    new = df.copy()
    cols = df.columns[offset:]
    new[cols] = (min_max_scaler.fit_transform(new[cols]) )  
    return new    
#%% Tüm Makinelerin verilerini import et
    
colnames = ["unit_num","time_in_cycles" ]
for i in range(3):
    colnames.append("operational_setting{}".format(i+1))
for i in range(21):
    colnames.append("s{}".format(i+1))
    
def getData(prefix,num,names=colnames):
        return pd.read_csv('{}_FD00{}.txt'.format(prefix,num), delim_whitespace=True, header = None, names= names )  
        
traindfs = []
testdfs = [] 
testYs = []

for i in range(4):
    traindfs.append(getData("train",i+1) ) 
      
for i in range(4): 
    testdfs.append(getData("test",i+1) )       
     
for i in range(4):
    testYs.append(getData("RUL",i+1,names=["Y"]))       

#%% Makinelerden birini seç
index = 3
traindf = traindfs[index]
testdf  = testdfs[index]
testY   = testYs[index]  
#           R^2   MAE     RMSE 
# Index 0  0.74  17.27   21.27 : LSTM
# Index 0  0.79  15.09   18.8 : LSTM, CROP 50
# Index 0  0.72  17.27   27.09 : STACK MLP+LASSO -> MLP LOOK_BACK: 31
# Index 0  0.73  16.31   21.65 : LSTM, PADDING 50
# Index 1  0.61  25.66   33.64 : LGMRegressor           LOOK_BACK: 21
# Index 2  0.51  20.49   29.01 : STACK MLP+LASSO -> MLP LOOK_BACK: 35
# Index 2  0.62  19.57   25.66 : STACK MLP+LASSO -> MLP LOOK_BACK: 50 PADDING
# Index 3 -> look back = 19
# Denenecekler: Sabit look_back ile bu look_back'in altında kalan veriler gözardı edilecek

#%% Veri ANALİZİ
intro(traindf)
plotCols(traindf,"time_in_cycles")
XCorrWithY(traindf,["s9"],"s14")
"""
Time in cycles: Sensor2,3,4,11,15,17 ile pozitif korelasyonu bulunuyor 
Sensor2'nin: Sensor 3,4,8,11,13,15,17
Sensor3'ün:  Sensor 4,8,9,11,13,15,17
Sensor4'ün: 8,11,13,15,17 
Sensor7'nin: 12,20,21
Sensor8'in: 11,13,15,17
Sensor9'un: 14(%96)  
Sensor11: 13,15,17
Sensor12: 20,21
Sensor13: 15,17
Sensor15: 17
Sensor20: 21


-Veride Null değer yok
-15 ve 17. sensörlerin çoğu ile korelasyonu vr 
-operationalssetting3,s1,s5,10,s16,s18 ve s19'un train datasında sadece 1 değeri var; s6-> 2 değer 
"""

#%% Preprocessing for 001 

#Sabit kalan feature'ları temizle
def removeSame(df,threshold=1):
    new = df.copy()
    willremove = [] 
    for col in new.columns:
        if(len(new[col].unique())<=threshold):
            del new[col] 
            willremove.append(col)
    return new,willremove

#Korelasyonu fazla olan feature'leri temizle
def get_train_columns(df,CorrThreshold):
    corrByValues= df.corr().copy()
    corrMat = abs(corrByValues)>=CorrThreshold
    print(corrMat)
    columnList = df.columns.to_list()
    length = len(columnList)
    features = columnList
    for i in range(length):
        for j in range(i+1,length):
            if(corrMat.iloc[i,j]==True and df.columns[j] in features):
                features.remove(df.columns[j])
    return features  

#Time-Series bir şekilde train ve test verilerini ayarla
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset.iloc[i:(i+look_back), :-1]
		dataX.append(a.to_numpy())
		dataY.append(dataset.iloc[i + look_back, -1]) 
	return np.array(dataX), np.array(dataY)  
trainX = traindf
testX = testdf
#Her devirde Sabit kalan değerleri çıkar 
trainX,deletedCols = removeSame(traindf.iloc[:,:],threshold=1)
testX = testdf.drop(columns=deletedCols)   

#Korelasyonu başka feature'lardan belirli seviteen fazla olan fetureları çıkar 
input_features = get_train_columns(trainX,0.75) 
trainX, testX = trainX[input_features], testX[input_features]
#Her unit için çalıştığı en fazla devir
trainYs = traindf.groupby(["unit_num"]).time_in_cycles.max() 

#%% Method 2
CROP = True
PADDING = True 
#Prepare train for all units without intercepting with each other     
look_back = 50 #min(trainYs.min(),testX.groupby(["unit_num"]).time_in_cycles.max().min())
assert look_back<=min(trainYs.min(),testX.groupby(["unit_num"]).time_in_cycles.max().min()) or CROP
unitnums = trainX.unit_num.unique() 
def Padding(tX):
    unitnums = tX.unit_num.unique()
    dfPerUnit = []
    for i in unitnums:
        maxCycle = int(tX.loc[tX.unit_num==i].time_in_cycles.max())
        tmp = tX.loc[tX.unit_num==i]
        if(maxCycle<look_back):
            for j in range(look_back-maxCycle):
                tmp = tmp.append(tmp.iloc[-1] , ignore_index= True)     
                tmp.iloc[-1].time_in_cycles = maxCycle + j + 1 
        dfPerUnit.append(tmp)
    return pd.concat(dfPerUnit)    
    
X = trainX.copy()
Y = trainYs.copy()
tX = testX.copy()
tY = testY.copy()
if(PADDING):
    tX = Padding(tX)
    X = Padding(X) 

X["time"] = X["time_in_cycles"]
tX["time"] = tX["time_in_cycles"]

X, tX = normalizedf(X,offset=2).fillna(0), normalizedf(tX,offset=2).fillna(0) 


X["Y"]=X.time_in_cycles
for i in range(len(unitnums)): 
    X["Y"].loc[X.unit_num==unitnums[i]] -= Y.iloc[i] 
    X["Y"]= abs(X["Y"]) 

       
    

 
#Padding: En arkadaki değere look_back-size kadar aynı değeri ekle, cycle değerlerini yeni eklenenlere göre düzenle  
unitnums = trainX.unit_num.unique()    
 
x, y = create_dataset(X.loc[X.unit_num==unitnums[0]].iloc[:,2:],look_back)
arrX = x
arrY = y
for i in range(1,len(unitnums)): 
    machineData = X.loc[X.unit_num==unitnums[i]]
    if(CROP and machineData.shape[0]>look_back):
        x, y = create_dataset(machineData.iloc[:,2:],look_back)
        arrX = np.vstack((arrX,x))
        arrY = np.append(arrY,y)
print(tX.shape,X.shape)

#Prepare test for all units without intercepting with each other    
    
testarrX = [tX[tX['unit_num']==id].values[-look_back:] for id in unitnums if (CROP and tX[tX['unit_num']==id].shape[0]>=look_back)]
print(tX.shape,X.shape)

if(not testarrX[-1].shape[0]):
    testarrX = testarrX[:-1]
testarrX = np.asarray(testarrX ).astype(np.float32)

tX = testarrX[:,:,2:] 
remainingIdsAfterCrop = testarrX[:,0,0].astype(int)
X = arrX
Y = arrY 
#edit tY for remainingIds after Crop. (-1 for mapping ids to indexes)
tY = testY.iloc[remainingIdsAfterCrop-1].Y
featureCount = X.shape[2] 
#%% Check Shapes
print("Train shape: ",X.shape," ", Y.shape,"\nTest shape:",tX.shape," ",tY.shape)
np.random.seed(42)
assert(X.shape[1:]==tX.shape[1:]) 
#%%
track = pd.DataFrame(data={"Machine Type":[],"batch_size":[],"modelSum":[],"look_back":[],"optimizer":[],"lr":lr,"epochs":[],"history":[],"RMSE":[],"r2":[]})

#%% LSTM 
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM, Dropout, LeakyReLU  

from keras.optimizers import Adam, SGD, Adamax,RMSprop  
import tensorflow as tf 
from keras.callbacks import EarlyStopping, ModelCheckpoint
"""batch_size = 1000
epochs = 200
lr = 0.001
optimizer = Adam 

model = Sequential()
model.add(LSTM(200, input_shape=(look_back,featureCount ),return_sequences=True ,activation="linear"))  
model.add(Dropout(0.2))
model.add(LSTM(100, input_shape=(look_back,featureCount ),return_sequences=False ,activation="linear"))  
model.add(Dropout(0.2)) 
model.add(Dense(1 ),activation="relu") 
model.compile(loss='mse', optimizer=optimizer(lr=lr) ,metrics=["mae"] )
history = model.fit(X, Y, validation_data= (tX,tY),epochs=epochs, batch_size=batch_size, verbose=2,shuffle=True,use_multiprocessing=True,
                    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min'),
                                  ModelCheckpoint("model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)]) 


""" 
model = Sequential()
model.add(LSTM(
         input_shape=(look_back, featureCount),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2)) 
model.add(Dense(units=1,activation="linear")) 
model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae'  ])

print(model.summary())

# fit the network
history = model.fit(X, Y, epochs=100, batch_size=200 ,  validation_data=(tX,tY), verbose=2,use_multiprocessing=True,
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min'),
                       ModelCheckpoint("modelCROP.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=1)]
          )

#Plot History
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("LSTM Training")
plt.show()
#%% ML Preprocesssing and testing functionss 
import math 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score 
def Reshape3D(X):
    return np.reshape(X,(X.shape[0], featureCount*look_back))  
def testModel(model,MtX,tY): 
    testPredict = model.predict(MtX)
    testPredict = np.reshape(testPredict,testPredict.shape[0]) 
    tY = tY.astype("float")
    testPredict = testPredict.astype("float")  
    testScore = (mean_absolute_error(tY, testPredict))
    root_mse = math.sqrt(mean_squared_error(tY,testPredict))
    r2score = r2_score(tY,testPredict)
    print(str(model)+'\nTest Score: %.2f MAE' % (testScore))
    print('Test Score: %.2f RMSE' % (root_mse))
    print('Test Score: %.2f r2' % (r2score)) 
    plt.plot(testPredict)
    plt.plot(tY)
    plt.title(str(model))
    plt.show()
    return testScore,r2score
#%% TEst for LSTM 
from keras.models import load_model
model = load_model("modelCROP.h5",compile=True)
testModel(model,tX,tY)
 #%% 
trackML = pd.DataFrame(data={"MachineType":[],"Model":[],"look_back":[],"RMSE":[],"r2":[]})

#%% Machine Learning Training and Test       
from sklearn.linear_model import LinearRegression, Lasso,Ridge, BayesianRidge 
import xgboost as xgb
from sklearn import svm
from sklearn import tree 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
models = [LGBMRegressor()]#LinearRegression(), xgb.XGBRegressor(),Lasso(),MLPRegressor(max_iter=500),LGBMRegressor(),CatBoostRegressor( ),Ridge(),BayesianRidge(),tree.DecisionTreeRegressor(),svm.SVR(),GradientBoostingRegressor()] 
MtX = Reshape3D(tX)  
MX = Reshape3D(X)
testScores = []
r2scores = []
for MLmodel in models:
    MLmodel = MLmodel.fit(MX,Y)
    testScore,r2score = testModel(MLmodel,MtX,tY)
    testScores.append(testScore)
    r2scores.append(r2score)
# make predictions 

#%%
for i in range(len(models)):
    
    trackML = trackML.append({"MachineType":index,"Model":models[i],"look_back":look_back,"RMSE":testScores[i],"r2":r2scores[i]},ignore_index=True)
    
#%% Stacking Ensembling Machine Learning
# make a prediction with a stacking ensemble
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
# define dataset
# lasso + mlp -> mlp = 0.66 r^2 18.77 MAE,  eğer 30 devire bakarssa 0.68, cv=2, iter = 200 
# lasso + mlp -> mlp = 0.7 r^2, 17.57 MAE, 22.76 RMSE, 31 look_back, 250 iterasyon, 0.75 feature threshold, removesame , cv=2-> Paper'daki en iyi sonuçtan daha başarılı
# lasso + mlp -> mlp = 0.71 r^2, 17.03 MAE, 22.50 RMSE, 31 look_back, 300 iterasyon, 0.75 feature threshold, removesame , cv=2-> Paper'daki en iyi sonuçtan daha başarılı
# lasso + mlp -> mlp = 0.72 r^2, 17.27 MAE, 22.09 RMSE, 31 look_back, 300 iterasyon, 0.75 feature threshold, removesame th=2, cv=2-> Paper'daki en iyi sonuçtan daha başarılı

# Lasso + mlp -> svm = 0.68 r^2, 17.61 MAE, 30 devir cv=2
# 31 devir, Lasso + mlp -> mlp = 0.67 r^2, 18.42 MAE
# define the base models
def ScatterPredictions(models,X,Y):
    axis = np.arange(X.shape[0]) 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(axis, Y, s=10,  label="REAL Y", c='#FF4500')   
    for model in models: 
        ax1.scatter(axis, model.predict(X), s=10,  label=str(model)[:10]) 
    
    plt.title("Comparison of model predictions")
    plt.show()  
    
MtX = Reshape3D(tX)  
MX = Reshape3D(X)
print(MX.shape,Y.shape,MtX.shape,tY.shape)
max_iter = 300
"""
models = [Lasso().fit(MX,Y), MLPRegressor(max_iter=max_iter).fit(MX,Y)]
ScatterPredictions(models,MX,Y) """ 
level0 = list()
level0.append(('lasso', Lasso())) 
level0.append(('mlp', MLPRegressor(max_iter=max_iter)))    
# define meta learner model
level1 = MLPRegressor(max_iter=max_iter )
# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=2)
# fit the model on all available data
model =  model.fit(MX, Y)
# make a prediction for one example
testModel(model,MtX,tY)

""""""
#%% ANALYSIS ABOUT REGRESSION
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X, Y, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=trainX.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


#%% AVE MODEL

from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(model, ' .pkl') 

#%% LOAD MODEL 
model = joblib.load("model71.pkl")