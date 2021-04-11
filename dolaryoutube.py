# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:25:33 2021

@author: Feyza
"""

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import pandas_datareader as web


plt.style.use('fivethirtyeight')

#fiyatları al


df = web.DataReader('USDTRY=X', data_source='yahoo',start='2000-05-24',end='2021-04-02')#tarih değeraralıkları

# = pd.read_excel('GBPUSDK.xlsx', date_parser=[0])
# İlk 5 Satır



df.shape



# plt.figure(figsize=(16,8))

# plt.title("Kapanış Fiyatları Tarihsel")

# plt.plot(df['close'])

# plt.xlabel('Tarih',fontsize=18)

# plt.ylabel('USDTRY Açılış Fiyatları',fontsize=18)

# plt.show()





#kapanış fiyatları ile ilgili yeni bir data oluşturpip





data = df.filter(['Close'])



#datayı numpy serisine dönüştür



dataset = data.values



#egitilecek satırları al



training_data_len = math.ceil(len(dataset) * 0.8)



#datayı ölçscalendir



scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

#egitilecek satırları yarat



train_data = scaled_data[0:training_data_len, :]

#egitilmis datayı x ve y kordinatlarına ayır



x_train = []

y_train = []





for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])

    # if i<=61:

    #    print(x_train)

    #    print(y_train)

    #    print()

#x_train  ve y_train modellerini numpy serilerine çevir



x_train, y_train = np.array(x_train), np.array(y_train)



#datayı yeniden şekillerndir



x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#print(x_train.shape)



#LSTM MODEL YAPILANDIRMA



model =Sequential()

model.add(LSTM(50, return_sequences=True, input_shape =(x_train.shape[1],1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))



#fiyat modellemesini derle



model.compile(optimizer='Adam', loss='mean_squared_error')

#modeli egit



model.fit(x_train,y_train, batch_size=1,epochs=1)#train sayısı



#test et





test_data=scaled_data[training_data_len-60:,:]

x_test=[]
y_test=dataset[training_data_len:,:]



for y in range(60, len(test_data)):

    x_test.append(test_data[y-60:y, 0])

x_test=np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))



#tahmin edlen fiyatları al



predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)



#RMSE



rmse = np.sqrt(np.mean(predictions - y_test)**2)



#veriyi taşı



train = data[:training_data_len]

valid = data[:training_data_len]

valid['Tahminler']= predictions


#veriyi görselleştir

plt.figure(figsize=(16,8))

plt.title('model')

plt.xlabel('Tarih', fontsize=18)

plt.ylabel('Kapanış Fiyatları', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close','Tahminler']])

plt.legend(['Eğitilen Alan','Değer','Tahminler'],loc='lower right')



plt.show()



print(valid)