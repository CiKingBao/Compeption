# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from os.path import *
from sklearn.cluster import KMeans
import pickle as pkl
from  sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
def get_cluster(df):
    print('deal la_lo....')
    la = df['longitude']
    lo = df['latitude']
    path = 'kmeans_model.pkl'
    la_lo = np.array(list(zip(la, lo)))
    kmeans_model = KMeans(n_clusters=3, random_state=1).fit(la_lo)
    with open('kmeans_model.pkl', 'wb') as output:
         pkl.dump(kmeans_model, output)
    print('cluster done...')


if __name__ =='__main__':
    with open('results.csv', 'w') as f:
        f.write('row_id,shop_id' + '\n')
    # df = pd.read_csv('train-ccf_first_round_user_shop_behavior.csv', encoding="utf-8")
    # shop = pd.read_csv('train-ccf_first_round_shop_info.csv', encoding="utf-8")
    # test = pd.read_csv('test-evaluation_public.csv', encoding="utf-8")

    df = pd.read_csv('user_demo.csv')
    shop = pd.read_csv( 'train-ccf_first_round_shop_info.csv')
    test = pd.read_csv( 'test_demo.csv')

    df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
    df['time_stamp']=pd.to_datetime(df['time_stamp'])
    train=pd.concat([df,test])
    mall_list=list(set(list(shop.mall_id)))
    result=pd.DataFrame()

    #kmeans_model = get_cluster(train)
    get_cluster(train)
    pkl_file = open('kmeans_model.pkl', 'rb')
    kmeans_model = pkl.load(pkl_file)

    for mall in mall_list:
        mall = 'm_2123'
        train1=train[train.mall_id==mall].reset_index(drop=True)
        l=[]
        wifi_dict = {}
        for index,row in train1.iterrows():
            r = {}
            wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
            for i in wifi_list:
                r[i[0]]=int(i[1])
                if i[0] not in wifi_dict:
                    wifi_dict[i[0]]=1
                else:
                    wifi_dict[i[0]]+=1
            l.append(r)

        delate_wifi=[]
        for i in wifi_dict:
            if wifi_dict[i]>1:
                delate_wifi.append(i)
        m=[]
        for row in l:
            new={}
            for n in row.keys():
                if n in delate_wifi:
                    new[n]=row[n]
            m.append(new)
        print(len(train1['shop_id']))
        #print(len(train1['longitude']))
        # user_la_lo = np.array(list(zip(train1['longitude'], train1['longitude'])))
        # list_k = kmeans_model.predict(user_la_lo)
        #
        # train1['K'] =list(list_k)
        #print(train1)

        train1 = pd.concat([train1 , pd.DataFrame(m)], axis=1)
        #print(train1)
        df_train=train1[train1.shop_id.notnull()].fillna(0)
        df_test=train1[train1.shop_id.isnull()].fillna(0)

        len_df_train = len(df_train)
        #l = [i for i in range(len_df_train)]
        #df_train['Y_label'] = [1]*len(df_train)
        shopid_list = list(set(list(df_train['shop_id'])))
        #df_train['index'] = l
        mode_list = []

        shop_trupe = []
        for i in shopid_list:
            ind = shopid_list.index(i)
            if ind + 1 == len(shopid_list):
                break
            for i in itertools.product([i], shopid_list[ind + 1:]):
                shop_trupe.append(i)
        print(shop_trupe)
        features = [each for each in df_train.columns if \
                    each not in ['index', 'user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]

        for ele in shop_trupe:
            pos_train = df_train[df_train.shop_id == ele[0]]
            neg_train = df_train[df_train.shop_id == ele[1]]

            pos_index = shopid_list.index(ele[0])
            neg_index = shopid_list.index(ele[1])
            pos_train['index'] = [pos_index]*len(pos_train)
            neg_train['index'] = [neg_index]*len(neg_train)

            train_data = pd.concat([pos_train, neg_train]).reset_index(drop=True)
            print('train_data', train_data)
            # print(train_data)
            # train_X, test_X, train_Y, test_Y = train_test_split(train_data[features], train_data['index'], \
            #                                                              test_size=0.25, random_state=33)

            ss = StandardScaler()
            #train_X = ss.fit_transform(train_data[train_X])
            #test_X = ss.transform(train_data[test_X])
            lsvc = LinearSVC()
            model = lsvc.fit(train_data[features],train_data['index'])
            mode_list.append(model)
            # pre=model.predict(test_X)
            # print(pre)
            break


        pre_list = []

        for index, row in df_test.iterrows():
            #print(list(row[features]))
            print(str(row['row_id']))
            row_d = pd.DataFrame([list(row[features])])
            for ele in mode_list:
                #print(row[features].shape)
                pre_index = ele.predict(row_d)
                print(pre_index)
                pre_list.append(pre_index[0])
                print(pre_list)
            myset = list(set(pre_list))
            count = []
            for item in myset:
                count.append(pre_list.count(item))

            max_count =myset[count.index(max(count))]
            real_shopid = shopid_list[max_count]
            print(real_shopid)


            with open('results.csv', 'a') as f:
                    f.write(str(row['row_id']) + ',' + str(real_shopid) + '\n')

            break

        # for each1 in shopid_list:
        #     each = 's_2588'
        #     df_train.loc[df_train.shop_id != each,'Y_label']*=-1
        #     #print(df_train)
        #     pos_train = df_train[df_train.shop_id==each]
        #     neg_train = df_train[df_train.shop_id!=each]
        #     features = [each for each in df_train.columns if each not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]
        #     print(df_train[features])
        #     while len(pos_train)<len(neg_train):
        #         pos_train.loc[str(len(pos_train)),features]= pos_train[features].apply(lambda x: x.mean())
        #     while len(pos_train) > len(neg_train):
        #         neg_train.loc[str(len(neg_train)), features] = neg_train[features].apply(lambda x: x.mean())
        #     print(len(pos_train) ,len(neg_train))
        #     assert len(pos_train) ==len(neg_train)
        #     train_data = pd.concat([pos_train,neg_train])
        #     features = [each for each in df_train.columns if each not in ['Y_label','user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]
        #
        #     train_X, test_X, train_Y, test_Y = train_test_split(train_data[features], train_data['Y_label'], \
        #                                                         test_size=0.25, random_state=33)
        #     ss = StandardScaler()
        #     #train_X = ss.fit_transform(train_data[train_X])
        #     #test_X = ss.transform(train_data[test_X])
        #     lsvc = LinearSVC()
        #     model = lsvc.fit(train_X,train_Y)
        #     mode_list.append(model)
        #     pre=model.predict(test_X)
        #     print(pre)



        '''
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train['shop_id'].values))
        df_train['label'] = lbl.transform(list(df_train['shop_id'].values))

        features = [ each for each in df_train.columns if each not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]

        train_X, test_X, train_Y, test_Y = train_test_split(df_train[features], df_train['label'], \
                                                            test_size=0.25, random_state=33)



        #print(train_Y)
        #num_class = int(list(train_X)[0])

        num_class = train_Y.max() + 1
        #print(num_class)
        params = {
                'objective': 'multi:softmax',
                'eta': 0.1,
                'max_depth': 9,
                'eval_metric': 'merror',
                'seed': 0,
                'missing': -999,
                'num_class':num_class,
                'silent': 1
                }

        xgbtrain = xgb.DMatrix(train_X, train_Y)
        xgbtest = xgb.DMatrix(test_X)
        watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test')]
        num_rounds = 60
        model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
        pre_label = model.predict(xgbtest)

        id_pre = [lbl.inverse_transform(int(each)) for each in list(pre_label)]
        id_real = [lbl.inverse_transform(int(each)) for each in list(test_Y)]

        
        acc=sum(id_pre[i] == id_real[i] for i in range(len(id_pre)))/len(id_pre)
        ss = mall + ',precision:' + str(acc) + '\n'
        print('precision:', acc)
        # pre_shopid=pre_test.apply(lambda x:lbl.inverse_transform(int(x)))
        with open('data.txt', 'a')as f:
            f.write(ss)

        # k = 0
        # while acc <0.9 and k<5:
        #     k = k + 1
        #     num_rounds = 60 + k*10
        #     model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
        #     pre_label = model.predict(xgbtest)
        #     id_pre = [lbl.inverse_transform(int(each)) for each in list(pre_label)]
        #     id_real = [lbl.inverse_transform(int(each)) for each in list(test_Y)]
        #     acc = sum(id_pre[i] == id_real[i] for i in range(len(id_pre))) / len(id_pre)
        #     ss = mall + ',precision:' + str(acc) + '\n'
        #     print('precision:', acc)
        #     # pre_shopid=pre_test.apply(lambda x:lbl.inverse_transform(int(x)))
        #     with open('data.txt', 'a')as f:
        #         f.write(ss)

        xgbtest = xgb.DMatrix(df_test[features])
        label_pre = model.predict(xgbtest)
        shopid = [lbl.inverse_transform(int(each)) for each in list(label_pre)]
        df_test['shop_id'] = shopid
        r=df_test[['row_id','shop_id']]
        result=pd.concat([result,r])
        result['row_id']=result['row_id'].astype('int')
        result.to_csv('result.csv',index=False)
        print('over')
        '''
        break
