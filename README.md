# Compeption
# 赛题目的
 根据测试集给出的用户 user_id、mall_id、longitude、latitude、wifi_infos，预测数用于会出现在哪个shop_id中。
 
 训练集first_round_user_shop_behavior.csv有用户的user_id、shop_id、longitude、latitude、wifi_infos等信息。
 训练集first_round_shop_info.csv有店铺的shop_id、category_id、longitude、latitude、mall_id等信息。
  
 由于一个商场（mall）会有多个店铺（shop_id），就把所有的训练集和测试集样本根据mall进行划分。即根据训练集中某一个mall的数据训练生成模型，然后此模型预测测试集中的属于同一个mall_id中的数据。
 
 #特征提取
 此数据中用到的标签有longitude、latitude、wifi_infos。
 1、经纬度用keams 进行聚类，即每个用户的的位置用keams聚类的类标号来表示。
 2、每个用户的wifi_infos信息是出现在客户身边wifi信号的强度、wifi的id、是否连接。在构建特征时，将所有的wifi信号的ID都作为是一个标签，例如出现在某一个mall中的wifi信号一共有10000个，即有wifi信号构成向量是10000维。某一个用户身边的wifi是40个，即可构成10000维的向量中40个位置是真实的wifi强度值，其余位置补0。
 3、在利用处理时间信息时，将用户出现在店铺的时间用一个值表示，例如14：:20分出现过，即用14代表用户出现在商城的时间。
 
 
