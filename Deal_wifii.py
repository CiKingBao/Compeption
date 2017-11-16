#coding: utf-8
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import pickle as pkl
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import sys
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
def get_user_time(user_infos):
    print('deal user_time....')
    #user_time = np.array([each.split()[1].split(":")[0] for each in user_infos["time_stamp"]])
    user_time = [int(each.split()[1].split(":")[0]) for each in user_infos["time_stamp"]]
    return user_time
def get_list_k(shop_infos,user_infos):
    # deal la_l0
    print('deal la_lo....')
    la = shop_infos['longitude']
    lo = shop_infos['latitude']
    la_lo = np.array(list(zip(la, lo)))
    kmeans_model = KMeans(n_clusters=1000, random_state=1).fit(la_lo)
    user_la_lo =np.array(list(zip(user_infos['longitude'],user_infos['latitude'])))
    list_k = kmeans_model.predict(user_la_lo)
    return list(list_k)

def get_mall(user_infos,shop_infos):
    # deal mall
    print('deal mall.....')
    shop_mall = list(shop_infos["mall_id"])
    shop_id = list(shop_infos["shop_id"])
    mall_all = []
    for row in user_infos["shop_id"]:
        mall_all.append(shop_mall[shop_id.index(row)])
    #mall_all = pd.Series(mall_all).str.split('_').apply(lambda x: x[1])
    shop_malls = list(set(shop_mall))
    mall_info=[shop_malls.index(each) for each in mall_all]
    print('mall=',mall_info)
    return mall_info
def get_mall_test(test_infos,shop_infos):
    # deal mall
    print('deal mall.....')
    shop_mall = list(shop_infos["mall_id"])
    shop_malls = list(set(shop_mall))
    mall_info=[shop_malls.index(each) for each in test_infos["mall_id"]]
    return mall_info
def get_wifi(user_infos):
    # deal wifi_infos
    print('deal wifi-info....')
    wifi = []
    user_wifi_infos = list(
        set([each1.split("|")[0] for each in list(user_infos["wifi_infos"]) for each1 in each.split(";")]))
    print(len(user_wifi_infos))
    for row in user_infos["wifi_infos"]:
        wifi_id = []
        wifi_qd = []
        for each in row.split(';'):
            wifi_id.append(each.split('|')[0])
            wifi_qd.append(each.split('|')[1])
        wifi.append(user_wifi_infos.index(wifi_id[wifi_qd.index(max(wifi_qd))]))
    print('wifi_id.len=', len(wifi))
    with open('wifi_train.pkl', 'wb') as output:
        pkl.dump(wifi, output)
    return wifi

def get_wifis_train():
    csv_reader = csv.reader(open('train-ccf_first_round_user_shop_behavior.csv', encoding='utf-8'))
    csv_shop = pd.read_csv('train-ccf_first_round_shop_info.csv', encoding="utf-8")
    shops_id =list(csv_shop['shop_id'])
    malls_id =list(csv_shop['mall_id'])
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)

#    dic = {}
    wifi =[]

#    i = 0
#    for row in csv_reader:
#        i = i + 1
#        if i == 1:
#            continue
#        print(i,row[1])
#        print(shops_id.index(row[1]))
#        mall =malls_id[shops_id.index(row[1])]
#        if mall in dic.keys():
#            ele = [each.split('|')[0] for each in row[5].split(';') ]
#            dic[mall].extend(ele)
#            list(set(dic[mall]))
#        else:
#            dic[mall] = [each.split('|')[0]   for each in row[5].split(';') ]
#
#        sys.stdout.write('generated:{0}/total:{1}\r'.format(i, 1138016))
#        sys.stdout.flush()
#
    #print(dic)
    # with open('wifi_idc.pkl', 'wb') as output:
    #     pkl.dump(dic, output)
    csv_reader_1 = csv.reader(open('train-ccf_first_round_user_shop_behavior.csv', encoding='utf-8'))
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[5].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        #each = sorted(mei[6].split(";"), key=lambda x: x.split("|")[1], reverse=True)
        each1 = each[0].split("|")[0]
        mall_1 = malls_id[shops_id.index(mei[1])]
        wifi.append(dic[mall_1].index(each1))
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 1138016))
        sys.stdout.flush()
    with open('wifi_train.pkl', 'wb') as output:
        pkl.dump(wifi, output)
    return wifi

def get_wifi_secend():
    print('deal train_wifi_sencend....')
    wifi_sencend = []
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)
    csv_shop = pd.read_csv('train-ccf_first_round_shop_info.csv', encoding="utf-8")
    shops_id = list(csv_shop['shop_id'])
    malls_id = list(csv_shop['mall_id'])
    csv_reader_1 = csv.reader(open('train-ccf_first_round_user_shop_behavior.csv', encoding='utf-8'))
    # csv_reader_1 = csv.reader(open('test-evaluation_public.csv', encoding='utf-8'))
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[5].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        #print(each)
        # each = sorted(mei[6].split(";"), key=lambda x: x.split("|")[1], reverse=True)
        #print(len(each),'\n')
        if len(each)>=2:
            each1 = each[1].split("|")[0]
        else:
            each1 = each[0].split("|")[0]
        #each1 = each[1].split("|")[0]
        mall_1 = malls_id[shops_id.index(mei[1])]
        wifi_sencend.append(dic[mall_1].index(each1))
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 1138016))
        sys.stdout.flush()
    with open('wifi_sencend_train.pkl', 'wb') as output:
        pkl.dump(wifi_sencend, output)
    return wifi_sencend

def get_wifi_third():
    print('deal train_wifi_third....')
    csv_shop = pd.read_csv('train-ccf_first_round_shop_info.csv', encoding="utf-8")
    csv_reader_1 = csv.reader(open('train-ccf_first_round_user_shop_behavior.csv', encoding='utf-8'))
    wifi_third = []
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)
    shops_id = list(csv_shop['shop_id'])
    malls_id = list(csv_shop['mall_id'])
    k = 0
    for mei in csv_reader_1:    
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[5].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        if len(each)>=3:
            each1 = each[2].split("|")[0]
        else:
            each1 = each[0].split("|")[0]
        mall_1 = malls_id[shops_id.index(mei[1])]
        wifi_third.append(dic[mall_1].index(each1))
        sys.stdout.write('wifi_third generated:{0}/total:{1}\r'.format(k, 1138016))
        sys.stdout.flush()
    with open('wifi_third_train.pkl', 'wb') as output:
        pkl.dump(wifi_third, output)
    return wifi_third    # bug*****************************
def get_wifis_test():
    print('deal test__wifi....')
    wifi =[]
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)
    csv_reader_1 = csv.reader(open('test-evaluation_public.csv', encoding='utf-8'))
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[6].split(";"), key = lambda x: x.split("|")[1], reverse=False)
        each1 = each[0].split("|")[0]
        if each1 in dic[mei[2]]:
            wifi.append(dic[mei[2]].index(each1))
        else:
            wifi.append(1)
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 483932))
        sys.stdout.flush()
    with open('wifi_test.pkl', 'wb') as output:
        pkl.dump(wifi, output)
    return wifi
def get_wifis_secend_test():
    print('test_secend_wifi')
    wifi_secend_test =[]
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)
    csv_reader_1 = csv.reader(open('test-evaluation_public.csv', encoding='utf-8'))
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[6].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        if len(each) >= 2:
            each1 = each[1].split("|")[0]
        else:
            each1 = each[0].split("|")[0]
        if each1 in dic[mei[2]]:
            wifi_secend_test.append(dic[mei[2]].index(each1))
        else:
            wifi_secend_test.append(1)
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 483932))
        sys.stdout.flush()
    with open('wifi_secend_test.pkl', 'wb') as output:
        pkl.dump(wifi_secend_test, output)
    return wifi_secend_test
def get_wifis_third_test():
    print('test_third_wifi')
    wifi_third_test =[]
    pkl_file = open('wifi_idc.pkl', 'rb')
    dic = pkl.load(pkl_file)
    csv_reader_1 = csv.reader(open('test-evaluation_public.csv', encoding='utf-8'))
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[6].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        if len(each) >=3:
            each1 = each[2].split("|")[0]
        else:
            each1 = each[0].split("|")[0]
        if each1 in dic[mei[2]]:
            wifi_third_test.append(dic[mei[2]].index(each1))
        else:
            wifi_third_test.append(1)
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 483932))
        sys.stdout.flush()
    with open('wifi_third_test.pkl', 'wb') as output:
        pkl.dump(wifi_third_test, output)
    return wifi_third_test

def get_shop_id(user_infos):
    # deal shop_id
    print('deal shop_id....')
    shop_ids = [each for each in user_infos["shop_id"]]
    return shop_ids

def get_user_id(user_infos,test_infos):
    print('deal user_id')
    k = 0
    user_id = [each for each in user_infos["shop_id"]]
    print(user_id)
    user_ids = list(set(user_id))
    print('user_id',len(user_ids))
    userid = []
    for each in test_infos['user_id']:
        k=k+1
        userid.append(user_ids.index(each))
        sys.stdout.write('generated:{0}/total:{1}\r'.format(k, 1138016))
        sys.stdout.flush()
    #userid = [user_ids.index(each) for each in test_infos['user_id']]
    return userid

def get_wifi_sorted():
    print('deal train_wifi_third....')
    csv_reader_1 = csv.reader(open('train-ccf_first_round_user_shop_behavior.csv', encoding='utf-8'))
    wifi_input = []
    k = 0
    for mei in csv_reader_1:
        k = k + 1
        if k == 1:
            continue
        each = sorted(mei[5].split(";"), key=lambda x: x.split("|")[1], reverse=False)
        wifi= [ele.split('|')[0] for ele in each]
        if len(wifi)<5:
            while len(wifi)<5:
                wifi.append('1')
            wifi_id = wifi
        else:
            wifi_id = wifi[:5]
        wifi_input.append(wifi_id)
        if k >9:
            break
    return wifi_input

if __name__ =="__main__":
    print('read csv......')
    user_infos = pd.read_csv('train-ccf_first_round_user_shop_behavior.csv',encoding="utf-8")
    shop_infos = pd.read_csv('train-ccf_first_round_shop_info.csv',encoding="utf-8")

    mall_all=get_mall(user_infos,shop_infos)
    user_time  = get_user_time(user_infos)
    list_k = get_list_k(shop_infos,user_infos)
    #wifi = get_wifi(user_infos)
    #wifi = get_wifis_train()
    pkl_file = open('wifi_train.pkl', 'rb')
    wifi = pkl.load(pkl_file)
    #wifi_secend=get_wifi_secend()
    pkl_file = open('wifi_sencend_train.pkl', 'rb')
    wifi_secend = pkl.load(pkl_file)
    wifi_third = get_wifi_third()
    # pkl_file = open('wifi_third_train.pkl', 'rb')
    # wifi_third = pkl.load(pkl_file)

    wifi_input = get_wifi_sorted()


    shop_ids = get_shop_id(user_infos)
    #userid = get_user_id(user_infos, user_infos)
    #x_train = pd.DataFrame([mall_all,user_time,list_k,wifi])
    x_train = pd.DataFrame([list_k,wifi,wifi_secend,wifi_third])
    y_train = pd.Series(shop_ids)
    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train.T)  # 归一化后的结果
    #y_train_minmax = min_max_scalar.fit_teansfoem(y_train)
    with open('x_train.pkl','wb') as output:
        pkl.dump(x_train,output)
    with open('y_train.pkl','wb') as f:
        pkl.dump(y_train,f)

    # print("train model...")
    # print("fitting.....")
    # x_train ,x_test ,y_train ,y_test = train_test_split(x_train_minmax,y_train,test_size=0.4,random_state=33)
    # dtc = RandomForestClassifier()
    # dtc = DecisionTreeClassifier()
    # dtc.fit(x_train, y_train)
    # print(dtc.score(x_test,y_test))


    # deal test data
    test_infos = pd.read_csv('test-evaluation_public.csv',encoding="utf-8")
    #test_mall_all = list(test_infos['mall_id'])
    test_mall_all=get_mall_test(test_infos,shop_infos)
    #test_user_time = get_user_time(test_infos)
    test_list_k = get_list_k(shop_infos,test_infos)
    test_wifi = get_wifis_test()
    #pkl_file = open('wifi_test.pkl', 'rb')
    #test_wifi = pkl.load(pkl_file)
    wifi_secend_test = get_wifis_secend_test()
    #pkl_file = open('wifi_secend_test.pkl', 'rb')
    #wifi_secend_test = pkl.load(pkl_file)
    wifi_third_test  = get_wifis_third_test()
    #pkl_file = open('wifi_third_test.pkl', 'rb')
    #wifi_third_test = pkl.load(pkl_file)
    x_test = pd.DataFrame([test_list_k,test_wifi,wifi_secend_test])
    min_max_scaler = MinMaxScaler()
    x_test_minmax = min_max_scaler.fit_transform(x_test.T)


    #print(x_train)
    #vet = DictVectorizer(sparse=False)
    #x_train = vet.fit_transform(x_train)
    #x_test = vet.transform(x_test)
    #base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=4)
    #dtc = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=15, learning_rate=0.1)
    #dtc.fit(x_train, y_train)
    

    #Decision Tree
    print('fitting....')
    dtc = DecisionTreeClassifier()
    #dtc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    dtc.fit(x_train_minmax, y_train)
    #joblib.dump(dtc, 'DecisionTreeClassifier_GBDT.model')
    #dtc = joblib.load('DecisionTreeClassifier.model')
    pred = dtc.predict(x_test_minmax)

    with open('y_pre_0.83.pkl', 'wb') as f:
       pkl.dump(pred, f)
    df = pd.DataFrame(list(zip(list(test_infos['row_id']), list(pred))), columns=['row_id', 'shop_id'])
    df.to_csv("result_new.csv", sep=',', index=False)
