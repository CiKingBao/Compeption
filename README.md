# Compeption
# 赛题目的
 根据测试集给出的用户 user_id、mall_id、longitude、latitude、wifi_infos，预测数用于会出现在哪个shop_id中。
 
 训练集first_round_user_shop_behavior.csv有用户的user_id、shop_id、longitude、latitude、wifi_infos等信息。
 训练集first_round_shop_info.csv有店铺的shop_id、category_id、longitude、latitude、mall_id等信息。
  
 由于一个商场（mall）会有多个店铺（shop_id），就把所有的训练集和测试集样本根据mall进行划分。即根据训练集中某一个mall的数据训练生成模型，然后此模型预测测试集中的属于同一个mall_id中的数据。
 
 # 特征构建
 此数据中用到的特征有longitude、latitude、wifi_infos。<br>
 1、经纬度用keams 进行聚类，即每个用户的的位置用keams聚类的类标号来表示。<br>
 2、每个用户的wifi_infos信息是出现在客户身边wifi信号的强度、wifi的id、是否连接。在构建特征时，将所有的wifi信号的ID都作为是一个标签，例如出现在某一个mall中的wifi信号一共有10000个，即有wifi信号构成向量是10000维。某一个用户身边的wifi是40个，即可构成10000维的向量中40个位置是真实的wifi强度值，其余位置补0。<br>
 3、在利用处理时间信息时，将用户出现在店铺的时间用一个值表示，例如14：:20分出现过，即用14代表用户出现在商城的时间。<br>
 
 # 模型选择
 对于某一个用户可能出现在商场中任何一个店铺中，所以选择模型应该是多分类模型。多分类的基础模型是决策树，随机森林，xgboost。
 ## 模型效果
 由于之前构建的特征维度较高，且样本量大（113万条数，每条数据提取的向量近千维），决策树非常容易过拟合。
 随机森林虽然解决决策树的过拟合问题，但是它投票进行最后结果的选择，其效果不如提升树模型xgboost。
 最后使用的是xgboost模型，使用时除了将数据提取为该模型所需的输入的格式外，就是调参问题，根据数据量的大小，以及分类个数等等..调节参数。
