
#coding : utf-8

import numpy as np
import pandas as pd
import bottleneck
from sklearn.cluster  import KMeans
import heapq
import  re
import datetime
import xgboost as xgb
from sklearn.cross_validation import train_test_split

wname_2_idx = dict()
sname_2_idx = dict()
uname_2_idx = dict()

def data_train(sub_user_df,sub_shop_df):
    count = 0
    x_data = []
    y_data = []

    kmeans = get_cluster(sub_shop_df)
    for index, row in sub_user_df.iterrows():
        pre = get_kmeans_pre(kmeans,row['longitude_x'],row['latitude_x'])
        top5_wifi_name, conn_w_n =get_dealwifi(row['wifi_infos'])
        time_point = get_time_point(row['time_stamp'])
        username_index = get_username(row['user_id'])
        x_data.append(np.concatenate([pre,top5_wifi_name,conn_w_n,time_point,[username_index]]))
        shopid_index = get_shopindex(row['shop_id'])
        y_data.append(shopid_index)
        break
    assert len(x_data)==len(y_data)
    return x_data,y_data
def data_test(sub_user_df,sub_shop_df):
    x_data = []
    kmeans = get_cluster(sub_shop_df)
    for index, row in sub_user_df.iterrows():
        pre = get_kmeans_pre(kmeans,row['longitude'],row['latitude'])
        top5_wifi_name, conn_w_n =get_dealwifi(row['wifi_infos'])
        time_point = get_time_point(row['time_stamp'])
        username_index = get_username(row['user_id'])
        x_data.append(np.concatenate([pre,top5_wifi_name,conn_w_n,time_point,[username_index]]))
    return x_data


def get_shopindex(shop_id):

    if shop_id in uname_2_idx.keys():
        return uname_2_idx[shop_id]
    else:
        uname_2_idx[shop_id] = len(uname_2_idx)
        return uname_2_idx[shop_id]

def get_username(name_id):
    if name_id in uname_2_idx.keys():
        return uname_2_idx[name_id]
    else:
        uname_2_idx[name_id] = len(uname_2_idx)
        return uname_2_idx[name_id]


def get_dealwifi(wifi_info,ntop=8):
    str = wifi_info.split(';')
    every_wifi = np.array([each.split('|')for each in str]) # 转化为矩阵
    wifi_id = every_wifi[:,0]
    wifi_value = every_wifi[:,1]
    wifi_state = every_wifi[:,2]
    print(wifi_state)
    print(wifi_value)
    #print(np.array(wifi_value))
    if 'true' in wifi_state:
        connection_wifi_name = wifi_id[wifi_state.tolist().index('true')]
    else:
        connection_wifi_name = 'null'
    if len(wifi_id)>= 8:
        top_5_idx = bottleneck.argpartition(np.array(wifi_value), ntop)[:ntop] # 找到前n大的几个数的索引
        return wf_name_2_idx(wifi_id[top_5_idx]),wf_name_2_idx([connection_wifi_name])
    else:
        sort_index  = np.argsort(-np.array(wifi_value))

        w_name = wifi_id[sort_index].tolist()
        w_name.extend(['null']*(8-len(wifi_value)))
        return wf_name_2_idx(w_name), wf_name_2_idx([connection_wifi_name])

def get_time_point(time):
    # rh = re.compile(r'[\d]+/[\d]+/[\d]+')
    # data = rh.findall(time)
    rh = re.compile(r'[\d]+:[\d]+')
    t = rh.findall(time)
    clock = t[0].split(':')[0]

    rd = re.compile(r'[\d]+-[\d]+-[\d]+')  # 日期
    d = rd.findall(time)
    week = datetime.datetime(*[int(each)for each in d[0].split('-')]).weekday()
    return [clock,week]

def wf_name_2_idx(w_name):
    print(1)
    wifi_name=[]
    for each in w_name:
        if each in wname_2_idx.keys():
            wifi_name.append(wname_2_idx[each])
        else:
            wname_2_idx[each] = len(wname_2_idx)
            wifi_name.append(wname_2_idx[each])
    return wifi_name


def get_cluster(sub_shop_df):
    lo_la = list(zip(sub_shop_df['mean_lo'],sub_shop_df['mean_la']))
    print(lo_la)
    kmeans = KMeans(n_clusters=30, random_state=1).fit(lo_la)
    print('kmeans done')
    return kmeans

def get_kmeans_pre(kmeans,lo,la):
    print('sdasdasd',[lo],[la])
    #print(zip([lo]))
    pre_lo_la = list(zip([lo],[la]))
    pre = kmeans.predict(pre_lo_la)
    print(pre)
    return pre
def get_shop_dis(loc,lo,la):
    shop_list = loc['shop_id']
    lo = np.array([lo] * len(shop_list))
    la = np.array([la] * len(shop_list))
    usrloc_fill_mat = np.matrix(list(zip(lo,la)))
    loc_mat = np.matrix(list(zip(loc['mean_lo'], loc['mean_la'])))
    diff_mat = np.power(usrloc_fill_mat - loc_mat,2).sum(axis=1)
    diff_mat =diff_mat.reshape(1,len(diff_mat))
    top_20_idx = bottleneck.argpartition(diff_mat, 20)[:20] # ？
    #print(top_20_idx)

def begin_training(x_train,y_train,mall_id):
    train_x,train_y,test_x,test_y =train_test_split(x_train,y_train,test_size=0.25, random_state=33)

    nclass = len(set(y_train))

    xg_train = xgb.DMatrix(train_x, label = train_y)
    xg_test = xgb.DMatrix(test_x, label = test_y)

    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 18
    param['silent'] = 1
    param['nthread'] = 6
    param['num_class'] = nclass

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 80
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pred = bst.predict(xg_test)
    print('predicting, classification ' + str(mall_id) + ':acc=%f' \
          % (sum(int(pred[i]) == test_y[i] for i in range(len(test_y)))/float(len(test_y))))
    return bst
def Beginpre(bst,y_test,row_id):
    test_y = xgb.DMatrix(y_test)
    pre = bst.predict(test_y)
    shopDict = list(zip(sname_2_idx.values(),sname_2_idx.keys()))
    pre_shop = [shopDict[int(each)]for each in pre]
    result = list(zip(row_id.tolist(),pre_shop))
    with open('result.csv','a') as f:
        for x,y in result:
            f.write(str(x) + ',' + str(y) + '\n')


if __name__ == '__main__':
    with open('result.csv', 'w') as f:
        f.write('row_id,shop_id' + '\n')
    loc = pd.read_csv('shop_location.csv')
    user_info = pd.read_csv('user_demo.csv')
    shop_info = pd.read_csv('train-ccf_first_round_shop_info.csv')
    test_info = pd.read_csv('test-evaluation_public.csv')

    user_shop = pd.merge(user_info,shop_info,on=['shop_id'],how='left')
    shop_loc  = pd.merge(loc,shop_info, on =['shop_id'],how = 'left')

    all_mall  = shop_info['mall_id'].drop_duplicates()

    for mall_id in all_mall:
        sub_user_df = user_shop[user_shop['mall_id']==mall_id]
        sub_shop_df = shop_loc[shop_loc['mall_id']==mall_id]
        sub_test_df = test_info[test_info['mall_id']==mall_id]

        print(sub_user_df)
        x_train, y_train = data_train(sub_user_df,sub_shop_df)
        y_test = data_test(sub_test_df,sub_shop_df)

        bst = begin_training(np.array(x_train).astype(float),np.array(y_train).astype(float),mall_id)
        Beginpre(bst,np.array(y_test),sub_test_df['row_id'])


        break















