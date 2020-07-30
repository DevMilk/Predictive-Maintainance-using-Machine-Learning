# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:20:12 2020

@author: Ugur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

figDir = "Figures/" 
#%%
from sklearn import preprocessing

def split(): print("\n____________________________________________________________________________________\n")
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
        
def XCorrWithY(df, X, Y):
    for col in  X:
        print(col,"-",Y)
        plt.scatter(df[col],df[Y]) 
        plt.title("{}-{}".format(col,Y))
        plt.show()
        plt.savefig(figDir+'{}.png'.format(col+"-"+Y))
        
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset.iloc[i:(i+look_back), :-1]
		dataX.append(a.to_numpy())
		dataY.append(dataset.iloc[i + look_back, -1]) 
	return np.array(dataX), np.array(dataY)       
def normalizedf(df,offset=0):
    min_max_scaler = preprocessing.MinMaxScaler() 
    new = df.copy()
    cols = df.columns[offset:]
    new[cols] = min_max_scaler.fit_transform(new[cols]) 
    return new    
#%%
    
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

#%% Get one
index = 0
traindf = traindfs[index]
testdf  = testdfs[index]
testY   = testYs[index] 
 

#%% ANALİZ
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
#trainX,deletedCols = removeSame(traindf.iloc[:,:])
#testX = testdf.drop(columns=deletedCols)   
#input_features = get_train_columns(trainX,0.75) 
#trainX, testX = trainX[input_features], testX[input_features]
#Her unit için çalıştığı en fazla devir
trainYs = traindf.groupby(["unit_num"]).time_in_cycles.max()
#%% Method 1
trainX.groupby(["unit_num"]).agg([min,max,'mean','std','var','last'])

input_features = get_train_columns(trainX,0.75) 
#Aynı kalanlara gerek yok (Eğer her texttekileri ayrı ele alacaksak)          
#Bir korelasyon thresholdu belirleyip o thresholdu aşanları train'e katmayalım
#Training'de s14 ve op_setting 3 yer almayacak
#Method 1: std, average, max ve min ve son cycle'ın son değerlerini al çıkış olarak kalanı ver
#Method 2: LookBack ile tahmin et (Buradaki look_back en fazla trainYs'deki en düşük değeri kadar olabilir.)
#Method 3: 


#%% Method 2


# Targetleri ayarla 
X = trainX.copy()
Y = trainYs.copy()
tX = testX.copy()
tY = testY.copy()
X["time"] = X["time_in_cycles"]
tX["time"] = tX["time_in_cycles"]
unitnums = trainX.unit_num.unique() 


X["Y"]=X.time_in_cycles
for i in range(len(unitnums)): 
    X["Y"].loc[X.unit_num==unitnums[i]] -= Y.iloc[i] 
    X["Y"]= abs(X["Y"]) 

       
    
X, tX = normalizedf(X,offset=2).fillna(0), normalizedf(tX,offset=2).fillna(0) 

#Prepare train for all units without intercepting with each other     
look_back = 24
assert look_back<=min(trainYs.min(),testX.groupby(["unit_num"]).time_in_cycles.max().min())
 
unitnums = trainX.unit_num.unique()  
x, y = create_dataset(X.loc[X.unit_num==unitnums[0]].iloc[:,2:],look_back)
arrX = x
arrY = y
for i in range(1,len(unitnums)): 
    x, y = create_dataset(X.loc[X.unit_num==unitnums[i]].iloc[:,2:],look_back)
    arrX = np.vstack((arrX,x))
    arrY = np.append(arrY,y)
print(tX.shape,X.shape)

#Prepare test for all units without intercepting with each other    
    
testarrX = [tX[tX['unit_num']==id].values[-look_back:] for id in unitnums ]
print(tX.shape,X.shape)

testarrX = np.asarray(testarrX).astype(np.float32)
tX = testarrX[:,:,2:] 
X = arrX
Y = arrY 
tY = testY
featureCount = X.shape[2] 
#%% Check Shapes
print("Train shape: ",X.shape," ", Y.shape,"\nTest shape:",tX.shape," ",tY.shape)
np.random.seed(42)
assert(X.shape[1:]==tX.shape[1:]) 
#%%
track = pd.DataFrame(data={"Machine Type":[],"batch_size":[],"modelSum":[],"look_back":[],"optimizer":[],"lr":lr,"epochs":[],"history":[],"RMSE":[],"r2":[]})

#%% Train 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, LeakyReLU
from keras.optimizers import Adam, SGD, Adamax,RMSprop 
import keras.backend as K
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
def r2custom(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model = Sequential()
model.add(LSTM(
         input_shape=(look_back, featureCount),
         units=100,
         return_sequences=True,activation="relu"))
model.add(Dropout(0.2))
model.add(LSTM(
          units=50,
          return_sequences=False,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation="relu")) 
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae',r2custom])

print(model.summary())

# fit the network
history = model.fit(X, Y, epochs=100, batch_size=200,  validation_split=0.05, verbose=2,use_multiprocessing=True,
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min'),
                       ModelCheckpoint("model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=1)]
          )

#Plot History
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("LSTM Training")
plt.show()
#%% Visualize Forecast
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import  load_model 
#model = load_model("model.h5",compile=True)
testPredict = model.predict(tX)
plt.plot(testPredict)
plt.plot(tY)
plt.show()
testScore = math.sqrt(mean_squared_error(tY, testPredict))
r2score = r2_score(tY,testPredict)
print('Test Score: %.2f RMSE' % (testScore))
print('Test Score: %.2f r2' % (r2score)) 
import io
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_summary_string = get_model_summary(model)

track = track.append({"Machine Type":[],"batch_size":batch_size,"modelSum":get_model_summary(model),"look_back":look_back,"optimizer":optimizer,"lr":lr,"epochs":epochs,"history":history.history,"RMSE":testScore,"r2":r2score},ignore_index=True)
#%% 
trackML = pd.DataFrame(data={"MachineType":[],"Model":[],"look_back":[],"RMSE":[],"r2":[]})
#%% ML Preprocesssing and testing functionss 
def Reshape3D(X):
    return np.reshape(X,(X.shape[0], featureCount*look_back)) 
def testModel(model,MX,Y,MtX,tY):
    Mmodel = model.fit(MX, Y)  
    testPredict = Mmodel.predict(MtX)
    testPredict = np.reshape(testPredict,testPredict.shape[0]) 
    tY = tY.astype("float")
    testPredict = testPredict.astype("float")
    crop = -1
    plt.plot(tY[:crop])
    plt.plot(testPredict[:crop])
    plt.show()
    testScore = math.sqrt(mean_squared_error(tY, testPredict))
    r2score = r2_score(tY,testPredict)
    print('Test Score: %.2f RMSE' % (testScore))
    print('Test Score: %.2f r2' % (r2score))
    plt.title(str(model))
    plt.plot(testPredict)
    plt.plot(tY)
    plt.show()
    return testScore,r2score
#%%          
from sklearn.linear_model import LinearRegression, Lasso,Ridge, BayesianRidge 
import xgboost as xgb
from sklearn import svm
from sklearn import tree
models = [xgb.XGBRegressor(),LinearRegression(),Lasso(),Ridge(),BayesianRidge()]#LinearRegression(), xgb.XGBRegressor()] 
MtX = Reshape3D(tX)  
MX = Reshape3D(X)
testScores = []
r2scores = []
for MLmodel in models:
    testScore,r2score = testModel(MLmodel,MX,Y,MtX,tY)
    testScores.append(testScore)
    r2scores.append(r2score)
# make predictions 

#%%
for i in range(len(models)):
    
    trackML = trackML.append({"MachineType":index,"Model":models[i],"look_back":look_back,"RMSE":testScores[i],"r2":r2scores[i]},ignore_index=True)