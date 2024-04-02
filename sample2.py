import os
import numpy as np
import pandas as pd
import random
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch_model as TM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


data_name = "data_mode_label"
data_path = os.path.join("data", data_name + ".csv")
data = pd.read_csv(data_path)
ytemp = data['label'].values
y = ytemp-np.ones(ytemp.shape)
X1 = data[['0', '1', '2', '3', '4', '5', '15', '16', '17', '18', '19', '20']].values  # 气压计
X2 = data[['6', '7', '8', '21', '22', '23', '30', '31', '32', '39', '40', '41']].values  # 加速度
X3 = data[['9', '10', '11', '24', '25', '26', '33', '34', '35', '42', '43', '44']].values  # 陀螺仪
X4 = data[['12', '13', '14', '27', '28', '29', '36', '37', '38', '45', '46', '47']].values  # 欧拉角
X5 = data[['0', '1', '2', '3', '4', '5', '12', '13', '14', '15', '16', '17', '18', '19', '20', '27', '28', '29',
           '36', '37', '38', '45', '46', '47']].values  # 气压计+欧拉角
X6 = data[['6', '7', '8', '9', '10', '11', '21', '22', '23', '24', '25', '26', '30', '31', '32', '33', '34', '35',
           '39', '40', '41', '42', '43', '44']].values  # 加速度+陀螺仪
X7 = data.drop(['12', '13', '14', '27', '28', '29', '36', '37', '38', '45', '46', '47', '48', 'label'],
               axis=1).values  # 气压+加速度+陀螺仪
X8 = data.drop(['0', '1', '2', '3', '4', '5', '15', '16', '17', '18', '19', '20', '48', 'label'],
               axis=1).values  # 加速度+陀螺仪+欧拉角
X9 = data.drop(['48', 'label'], axis=1).values  # 使用全部数据
# 定义数据信息-需修改
X = X9

# 打乱数据顺序，重新洗牌
data_size = X.shape[0]
print('data_size :', data_size)
arr = np.arange(data_size)  # 生成0到datasize个数
np.random.shuffle(arr)  # 随机打乱arr数组
X = X[arr]  # 将data以arr索引重新组合
y = y[arr]  # 将label以arr索引重新组合

print('load data:', data_name)
print('X :', X.shape, '|label :', y.shape)
in_dim = X.shape[1]  # 数据维度
out_dim = 7  # 定义分类问题类别数
n_rules = 10  # 定义TSK规则数
k = 5

kf = KFold(n_splits=k)     # k折交叉验证
acc_sum = 0
for train_index, valid_index in kf.split(X):
    x_train = X[train_index]
    x_test = X[valid_index]
    y_train = y[train_index]
    y_test = y[valid_index]
    # 预处理
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # 将训练集分为训练-验证两部分
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    # 模型训练
    model = TM.TSK(in_dim, out_dim, n_rules, "tsk")  # 定义模型，可选htsk,tsk,logtsk
    model.init_model(x_train)  # 根据训练集进行规则前件的初始化
    optimizer = Adam(model.parameters(), lr=0.01)  # 定义优化器，强烈建议使用Adam，学习率可以从0.1, 0.01和0.001三个中间看哪个效果好，推荐0.01
    criterion = CrossEntropyLoss()  # 定义损失函数
    es = TM.EarlyStopping([x_val, y_val], metrics="acc", patience=60,  # 如果效果不好建议先把patience改大再试
        larger_is_better=True, eps=1e-4, save_path="model2.pkl",
        only_save_best=True)  # 使用早停训练模型
    cp = TM.CheckPerformance([x_test, y_test], metrics="acc", name="Test")  # 训练过程中监控测试集ACC
    T = TM.Trainer(model, optimizer, criterion, device="cpu", callbacks=[es, cp], verbose=1)  # 定义训练器, device 代表训练的使用cpu还是gpu，如果为gpu，请写 device='cuda'，如果为cpu，倾斜 device='cpu'
    T.fit(x_train, y_train, max_epoch=500, batch_size=32)  # 模型训练

    # 测试
    model.load("model2.pkl")  # 早停存储的模型路径
    y_pred = model.predict_score(x_test)  # 预测概率
    y_pred = np.argmax(y_pred, axis=1)  # 概率转为类别结果
    acc_sum += np.mean(y_pred == y_test)  # 计算正确率
    print("Test ACC_sum: {:.4f}".format(acc_sum))

acc = acc_sum/k
print("Test ACC: {:.6f}".format(acc))

