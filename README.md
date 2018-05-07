# Compeption
# 赛题目的
 根据测试集给出的用户 user_id、mall_id、longitude、latitude、wifi_infos，预测数用于会出现在哪个shop_id中。
 
 训练集first_round_user_shop_behavior.csv有用户的user_id、shop_id、longitude、latitude、wifi_infos等信息。
 训练集first_round_shop_info.csv有店铺的shop_id、category_id、longitude、latitude、mall_id等信息。
  
 由于一个商场（mall）会有多个店铺（shop_id），就把所有的训练集和测试集样本根据mall进行划分。即根据训练集中某一个mall的训练生成该模型，然后
 
