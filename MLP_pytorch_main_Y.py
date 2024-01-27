# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
from neural import MLPptorch
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# функция обучения
def train(x, y, num_iter):
    for i in range(0,num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%(num_iter/10)==0:
           print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    return loss.item()


#df = pd.read_excel('res1mod1.xls')
#df.to_csv('res1mod2.csv')

df = pd.read_csv('res1mod2.csv')
df = df.iloc[np.random.permutation(len(df))]

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3) #разбиваем выборку на обучающую и тестовую

Y = np.zeros((y.shape[0], np.unique(y).shape[0]))


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 50 # задаем число нейронов скрытого слоя 
outputSize = Y.shape[1] if len(Y.shape) else 1 # количество выходных сигналов равно количеству классов задачи


net = MLPptorch(inputSize,hiddenSizes,outputSize)
lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

loss_ = train(torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 5000)


pred = net.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print(err)   

pred = net.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print(err)

