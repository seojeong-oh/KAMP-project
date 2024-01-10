# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# 성능평가지표
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, mean_absolute_percentage_error,mean_squared_error

# 필요한 패키지 불러오기
import tensorflow as tf
tf.random.set_seed(123)  # 랜덤 시드 설정
# 모델 학습 및 평가 코드
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import seaborn as sns
from tensorflow.keras.models import load_model
from os import listdir
# 랜덤 시드 설정
import random
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import glob
import os
from keras.models import Sequential 
from keras.layers import Dense, SimpleRNN 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from os import listdir
random.seed(42)

data = pd.read_excel('/UHome/etri32842/Predictive_Maintenance/Data/KAMP/20_SCM/20_SCM_Data/data/data.xls')
data.CRET_TIME = pd.to_datetime(data.CRET_TIME, format = "%Y%m%d%H%M")

print('시작 전 data.shape : ',data.shape)

def shape_clean(data):
    cut_list = []
    sum = 0
    k1 = pd.DataFrame(data.loc[:,('D일06~08(08)H 투입계획(발주) 수량', 'D일08~10(10)H 투입계획(발주) 수량',
        'D일10~12(13)H 투입계획(발주) 수량', 'D일13~15(15)H 투입계획(발주) 수량',
        'D일15~17(18)H 투입계획(발주) 수량', 'D일18~20(21)H 투입계획(발주) 수량',
        'D일21~23(23)H 투입계획(발주) 수량', 'D일23~01(02)H 투입계획(발주) 수량',
        'D일 02~04H 투입계획 수량', 'D일 04~06H 투입계획 수량')].sum(axis=1))

    for i in range(0, len(data)):
        if k1.loc[i,0] != data.loc[i, 'D일 투입예정 수량(D일계획)']:
            cut_list.append(i)
            sum +=1
            data.loc[i, 'D일 투입예정 수량(D일계획)'] = k1.loc[i, 0]  # 값 대입
    # data = data.drop(cut_list).reset_index(drop = True)

    cut_list = []
    k2 = pd.DataFrame(data.loc[:,('D+1일 투입예정 수량(D+1일)', 'D+1일 투입예정 수량(D+1일).1', 'D+1일 투입예정 수량(D+1일).2',
        'D+1일 투입예정 수량(D+1일).3', 'D+1일 투입예정 수량(D+1일).4', 'D+1일 투입예정 수량(D+1일).5',
        'D+1일 투입예정 수량(D+1일).6', 'D+1일 투입예정 수량(D+1일).7', 'D+1일 투입예정 수량(D+1일).8',
        'D+1일 투입예정 수량(D+1일).9')].sum(axis=1))
    for i in range(0, len(data)):
        if k2[0][i] != data['D+1일 투입예정 수량(Total)'][i]:
            cut_list.append(i)
            sum +=1
            data.loc[i, 'D일 투입예정 수량(D일계획)'] = k2[0][i]  # 값 대입
    # data = data.drop(cut_list).reset_index(drop = True)

    #d+2일
    cut_list = []
    k3 = pd.DataFrame(data.loc[:,('D+2일 투입예정 수량(과부족수량)',
        'D+2일 투입예정 수량(과부족수량).1', 'D+2일 투입예정 수량(과부족수량).2',
        'D+2일 투입예정 수량(과부족수량).3', 'D+2일 투입예정 수량(과부족수량).4',
        'D+2일 투입예정 수량(과부족수량).5', 'D+2일 투입예정 수량(과부족수량).6',
        'D+2일 투입예정 수량(과부족수량).7', 'D+2일 투입예정 수량(과부족수량).8',
        'D+2일 투입예정 수량(과부족수량).9')].sum(axis=1))
    for j in range(0, len(data)):
        if k3[0][j] != data['D+2일 투입예정 수량(Total)'][j]:
            cut_list.append(j)
            sum +=1
            data.loc[i, 'D일 투입예정 수량(D일계획)'] = k3[0][j]  # 값 대입
    # data = data.drop(cut_list).reset_index(drop = True)

    #d+3일
    cut_list = []
    k4 = pd.DataFrame(data.loc[:,('D+3일 투입예정 수량(과부족수량)',
        'D+3일 투입예정 수량(과부족수량).1', 'D+3일 투입예정 수량(과부족수량).2',
        'D+3일 투입예정 수량(과부족수량).3', 'D+3일 투입예정 수량(과부족수량).4',
        'D+3일 투입예정 수량(과부족수량).5', 'D+3일 투입예정 수량(과부족수량).6',
        'D+3일 투입예정 수량(과부족수량).7', 'D+3일 투입예정 수량(과부족수량).8',
        'D+3일 투입예정 수량(과부족수량).9')].sum(axis=1))
    for k in range(0, len(data)):
        if k4[0][k] != data['D+3일 투입예정 수량(Total)'][k]:
            cut_list.append(k)
            sum +=1
            data.loc[i, 'D일 투입예정 수량(D일계획)'] = k4[0][k]   # 값 대입
    # data = data.drop(cut_list).reset_index(drop = True)

    #d+4일
    cut_list = []
    k5 = pd.DataFrame(data.loc[:,('D+4일 투입예정 수량(과부족수량)','D+4일 투입예정 수량(과부족수량).1',
    'D+4일 투입예정 수량(과부족수량).2','D+4일 투입예정 수량(과부족수량).3', 'D+4일 투입예정 수량(과부족수량).4',
        'D+4일 투입예정 수량(과부족수량).5', 'D+4일 투입예정 수량(과부족수량).6',
        'D+4일 투입예정 수량(과부족수량).7', 'D+4일 투입예정 수량(과부족수량).8',
        'D+4일 투입예정 수량(과부족수량).9')].sum(axis=1))
    for j in range(0, len(data)):
        if k5[0][j] != data['D+4일 투입예정 수량(Total)'][j]:
            cut_list.append(j)
            sum +=1
            data.loc[i, 'D일 투입예정 수량(D일계획)'] =  k5[0][j]  # 값 대입
    # data = data.drop(cut_list).reset_index(drop = True)
    return data
data = shape_clean(data)
print('정리 끝난 후 data.shape : ',data.shape)


def to_timeseries_data(data, lookback, delay): #상관계수 input으로 넣을 데이터 생성하는 함수
    # data는 원본 tabular 데이터
    # lookback: 입력으로 사용하기 위해 거슬러 올라갈 시간단위의 개수=3일전0
    # delay: target으로 사용할 미래의 시점=3일후
    output_len = len(data)-(lookback+delay)+1 # N=total_length-(3+3)+1 #output_len만큼 반복할 것임
    n_feature = data.shape[-1] # =4 
    
    inputs = np.zeros((output_len, lookback, n_feature)) # (N,3,4)
    targets = np.zeros((output_len,)) # (N,)
    
    for i in range(output_len):
        inputs[i] = data.iloc[i:i+lookback, :]
        targets[i] = data.iloc[i+lookback+delay-1, 0]
    return inputs, targets
def get_best_model(model_directory):
    # 모든 모델 파일을 가져옵니다.
    model_files = glob.glob(os.path.join(model_directory, '*.hdf5'))    
    # 모델 파일들을 정렬합니다. (val_loss 기준으로 오름차순 정렬)
    model_files.sort(key=lambda x: float(x.split('-')[1][:-5]))

    # 가장 낮은 val_loss 값을 가진 모델을 불러옵니다.
    best_model = load_model(model_files[0])
    return best_model
def adj_r2_score(y_true, y_pred):
            return 1-(1-r2_score(y_true, y_pred)) * (len(y_true)-1) / (len(y_true) - (X_test_26.shape[1]) - 1)

def confirm_result(n, y_test, y_pred,scaling_name, model_name, d, epoch_size,batch_size_num,k,certify):
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE =np.sqrt(mean_squared_error(y_test,y_pred))
    R2 = r2_score(y_test, y_pred)
    adj_R2 = adj_r2_score(y_test, y_pred )
    # MAPE= np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    pd.options.display.float_format = '{:.5f}'.format
    Result = pd.DataFrame(data = [MAE, MAPE, R2, adj_R2,RMSE],
                        index = ['MAE','MAPE','R2_score','Adjusted R2_score','RMSE'],
                        columns = ['{sd}- 은닉층 : {f}'.format(sd = model_name, f = d)])
    
    
    if  data21_df.columns[1].split(' ')[0] == data21_df.columns[-1].split(' ')[0]:
        col = data21_df.columns[0].split(' ')[0], data21_df.columns[1].split(' ')[0]
    else:
        coll = '{a} ~ {b}'.format(a= data21_df.columns[1].split(' ')[0], b= data21_df.columns[-1].split(' ')[0])
        col = data21_df.columns[0].split(' ')[0],coll
    output = pd.DataFrame(data = [k, n,  scaling_name,  col, epoch_size, batch_size_num, MAE, MAPE, R2, adj_R2,RMSE,certify],
                    index = ['저장된 폴더명','부품 번호', '스케일링 방법', '사용된 변수','epoch', 'batch_size_num',
                                                                            'MAE','MAPE','R2_score','Adjusted R2_score','RMSE','0인 값이 들어있나(0=없다, 1=있다)'],
                        columns = ['{sd}- 은닉층 : {f}'.format(sd = model_name, f = d)])
    
    return output
not_list = []
final_output = pd.DataFrame()
for j in range(0, 117):
    data21_df = data[data['Part Number']==f'Part {j}'].reset_index(drop = True)
    data21_df = data21_df.groupby(['CRET_TIME']).sum().reset_index()
    data21_df = data21_df.drop(['Part Number' ,'CRET_TIME'], inplace = False, axis= 1).groupby(by=[data21_df.CRET_TIME.dt.year, 
                                                                                                data21_df.CRET_TIME.dt.month, 
                                                                                                data21_df.CRET_TIME.dt.day]).last().reset_index(drop = True)                                                                         
    if data21_df.shape[0] != 49:
        not_list.append(j)
        j+=1

for scale_num in range(1,4):
    print('scale_num:',scale_num)
    n=0
    if scale_num == 1 :
        Xscaler_26 = StandardScaler()
        yscaler_26 = StandardScaler()
        scaling_name = 'Standard'
    elif scale_num ==2:
        Xscaler_26 = RobustScaler()
        yscaler_26 = RobustScaler()
        scaling_name = 'Robust'
    else:
        Xscaler_26 = MinMaxScaler()
        yscaler_26 = MinMaxScaler()
        scaling_name = 'MinMax'
    print('scaling_name : ', scaling_name)
    while n <117:

        if n in not_list : #부품 n번의 데이터가 49개가 아니라면 분석을 진행시키지 않음
            n +=1
        elif n not in not_list: #부품 n번의 데이터가 49개일 경우
                

            data21_df = data[data['Part Number']==f'Part {n}'].reset_index(drop = True)
            data21_df = data21_df.groupby(['CRET_TIME']).sum().reset_index()
            data21_df = data21_df.drop(['Part Number' ,'CRET_TIME'], inplace = False, axis= 1).groupby(by=[data21_df.CRET_TIME.dt.year, 
                                                                                                            data21_df.CRET_TIME.dt.month, 
                                                                                                            data21_df.CRET_TIME.dt.day]).last().reset_index(drop = True)    
            
            data21_df = data21_df.loc[:,['D일 투입예정 수량(D일계획)', 'D+3일 투입예정 수량(Total)','D+4일 투입예정 수량(Total)', 'D+5일 투입예정 수량']]
            delay = int(data21_df.columns[1].split('일')[0].split('+')[1])

            n21 = data21_df.shape[1]-1
            X_26,y_26 = to_timeseries_data(data21_df, n21, delay)


            # 데이터셋 분리, train:validation:test = 7:1:2
            X_train_26, X_val_26, X_test_26 = np.split(X_26, [int(0.7*len(X_26)), int(0.8*len(X_26))])
            y_train_26, y_val_26, y_test_26 = np.split(y_26, [int(0.7*len(y_26)), int(0.8*len(y_26))])
    
            
            up = 1
            certify = 0 #0인 값이 있어 대체했음 0이 아닌 숫자가 들어감
            replace_value = data21_df.iloc[len(data21_df)-y_test_26.shape[0]-up,0] #0을 대체할 값 탐색하기 위한 변수 생성
            while replace_value == 0:
                up += 1
                replace_value = data21_df.iloc[len(data21_df) - y_test_26.shape[0] - up, 0]  # up 값을 증가시키면서 대체값 재탐색
                # print('up:', up, ', replace_value:', replace_value)
                certify +=1
            # 0인 값을 replace_value로 대체
            y_test_26[y_test_26 == 0] = replace_value
            
            
            #스케일러
            X_train_26 = Xscaler_26.fit_transform(X_train_26.reshape(-1, X_train_26.shape[-1])).reshape(X_train_26.shape)
            X_val_26 = Xscaler_26.transform(X_val_26.reshape(-1, X_val_26.shape[-1])).reshape(X_val_26.shape)
            X_test_26 = Xscaler_26.transform(X_test_26.reshape(-1, X_test_26.shape[-1])).reshape(X_test_26.shape)
            
            y_train_26 = yscaler_26.fit_transform(y_train_26.reshape(-1,1))
            y_val_26 = yscaler_26.transform(y_val_26.reshape(-1,1))
            y_test_26 = yscaler_26.transform(y_test_26.reshape(-1,1))
            
            
            list_path = os.listdir('./weights')
            v = 1.0  # 초기 버전 설정
            k = '부품번호_{}_가중치 저장 파일_v{}'.format(n, v)
            while k in list_path:
                v = round(v + 0.1, 3)
                k = '부품번호_{}_가중치 저장 파일_v{}'.format(n, v)
                
            # 파일명 k가 폴더 안에 저장되지 않는 경우, while 루프를 빠져나온 후에 k 변수에 최종 버전이 포함된 파일명이 저장됩니다.
            # print('최종 저장될 파일명:', k)

            batch_size_num= 4
            epoch_num = 100
            
            #파일 경로
            lstm_directory1 = f'./weights/부품번호_{n}_가중치 저장 파일_v{v}/LSTM_model_v1/'
            lstm_directory1=lstm_directory1

            lstm_model_path1 = lstm_directory1 +'{epoch:02d}-{val_loss:.4f}.hdf5'

            a = X_train_26.shape[1]
            b = X_train_26.shape[2]
            
            model = Sequential()
            model.add(LSTM(8, dropout=0.2, activation='relu', input_shape=(a,b), return_sequences=True))
            model.add(LSTM(8, dropout=0.2, activation='relu'))
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer='adam', loss='mae')
            callbacks = [EarlyStopping(monitor='val_loss', patience=15),
                    ModelCheckpoint(filepath=lstm_model_path1, monitor='val_loss', verbose=0, save_best_only=True)]
            history1 = model.fit(X_train_26, y_train_26, epochs=100, batch_size=4, verbose = 0,
                            validation_data=(X_val_26, y_val_26), callbacks=callbacks)

            best_model1 = get_best_model(lstm_directory1)
            y_pred_26_LSTM = best_model1.predict(X_test_26)
            y_pred_26_inv_LSTM = yscaler_26.inverse_transform(y_pred_26_LSTM)
            y_test_26_inv = yscaler_26.inverse_transform(y_test_26)
            # if n == 2 and scaling_name == 'Standard':
            #     final_output = pd.DataFrame()
                
            # else:
            #     final_output
            result1 = confirm_result(n,y_test_26_inv, y_pred_26_inv_LSTM, scaling_name,'Reference 코드(LSTM)',len(model.layers)-2,len(history1.epoch),batch_size_num,k,certify)
            ek = result1
            final_output = pd.concat([final_output, ek.T], axis=0) 
            n+=1
            display(final_output)
final_output.to_csv('final_output_reference.csv',index=False)
final_output