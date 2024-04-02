import os, time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch_model as TM
from torch_model.optim import AdaBound
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.utils import shuffle

data_name = "zhang"
data_path = os.path.join("data", "data_twotask_" + data_name + ".csv")
data = pd.read_csv(data_path)
data = shuffle(data)
# y = data['label1'].values  # 步态相位分类
# y = data['label2'].values  # 运动模式分类
y = data[['label1','label2']].values  # 多任务分类

# <editor-fold desc="多模态数据加载和选择">
X1 = data[['0', '1', '2', '3', '4', '5', '15', '16', '17', '18', '19', '20']].values  # 气压计
X2 = data[['6', '7', '8', '21', '22', '23', '30', '31', '32', '39', '40', '41']].values  # 加速度
X3 = data[['9', '10', '11', '24', '25', '26', '33', '34', '35', '42', '43', '44']].values  # 陀螺仪
X4 = data[['12', '13', '14', '27', '28', '29', '36', '37', '38', '45', '46', '47']].values  # 欧拉角
X5 = data[['0', '1', '2', '3', '4', '5', '12', '13', '14', '15', '16', '17', '18', '19', '20', '27', '28', '29',
           '36', '37', '38', '45', '46', '47']].values  # 气压计+欧拉角
X6 = data[['6', '7', '8', '9', '10', '11', '21', '22', '23', '24', '25', '26', '30', '31', '32', '33', '34', '35',
           '39', '40', '41', '42', '43', '44']].values  # 加速度+陀螺仪
X7 = data.drop(['12', '13', '14', '27', '28', '29', '36', '37', '38', '45', '46', '47', 'label1', 'label2'],
               axis=1).values  # 气压+加速度+陀螺仪
X8 = data.drop(['0', '1', '2', '3', '4', '5', '15', '16', '17', '18', '19', '20', 'label1', 'label2'],
               axis=1).values  # 加速度+陀螺仪+欧拉角
X9 = data.drop(['label1', 'label2'], axis=1).values  # 使用全部数据
X10 = data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']].values  # 左右大腿节点数据
X11 = data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']].values  # 左右大腿气压计+加速度+陀螺仪
# </editor-fold>

# <editor-fold desc="定义数据信息-需修改">
X = X11
Xname = "X11"
in_dim = X.shape[1]  # 数据维度
task_out_dim = [4, 5]     # 定义分类问题类别数
batchsize = 512
LR = 0.01          # 学习率可以从0.1, 0.01和0.001三个中间看哪个效果好，推荐0.01
n_rules = 30        # 定义TSK规则数
W_l2=1e-4                  # w_l2是L2正则化项系数
W_ur=100                   # w_ur是标准正则化项系数
k = 1                      # k折交叉验证
train_time = 0             # 训练用时
note = ""                  # 备注
# </editor-fold>

print('load data:', data_name)
print('X :', X.shape, '|y :', y.shape)
# ss = StandardScaler()    # Z-score标准化
# X = ss.fit_transform(X)
# min_max_scaler = preprocessing.MinMaxScaler()         # max-min标准化
# X = min_max_scaler.fit_transform(X)
x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size=0.2)

# <editor-fold desc="储存k折交叉验证中每折分类评估结果的数组">
Acc1 = []            # 准确率，越高越好
Acc2 = []            # 准确率，越高越好
Precision1 = []      # 查准率或者精度: precision(查准率)=TP/(TP+FP),精确率直观地可以说是分类器不将负样本标记为正样本的能力.
Precision2 = []      # 查准率或者精度: precision(查准率)=TP/(TP+FP),精确率直观地可以说是分类器不将负样本标记为正样本的能力.
Recall1 = []         # 查全率: recall(查全率)=TP/(TP+FN)
Recall2 = []         # 查全率: recall(查全率)=TP/(TP+FN)
F1Score1 = []        # F1值 F1 = 2 * (precision * recall) / (precision + recall)
F1Score2 = []        # F1值 F1 = 2 * (precision * recall) / (precision + recall)
AUC1 = []            # 计算ROC曲线下的面积就是AUC的值，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的performance
AUC2 = []            # 计算ROC曲线下的面积就是AUC的值，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的performance
CM1 = []             # 混淆矩阵
CM2 = []             # 混淆矩阵
# </editor-fold>

# <editor-fold desc="数据预处理">
# min_max_scaler = preprocessing.MinMaxScaler()         # 0-1标准化
# x_train = min_max_scaler.fit_transform(x_train)
# x_test = min_max_scaler.transform(x_test)
ss = StandardScaler()                                   # Z-Score标准化
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# </editor-fold>

# 将训练集分为训练-验证两部分
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

# <editor-fold desc="模型训练">
start = time.perf_counter()
model = TM.TSK(in_dim, task_out_dim, n_rules, "tsk")  # 定义模型，可选htsk,tsk,logtsk
model.init_model(x_train, method="cluster")  # 根据训练集进行规则前件的初始化
optimizer = Adam(model.parameters(), lr=LR)  # 定义优化器，Adam，学习率可以从0.1, 0.01和0.001三个中间看哪个效果好，推荐0.01
# optimizer = AdaBound(model.parameters(), lr=0.001)   # 定义优化器，AdaBound，学习率可以从0.1, 0.01和0.001三个中间看哪个效果好，推荐0.01
criterion = CrossEntropyLoss()  # 定义损失函数
es = TM.EarlyStopping([x_val, y_val], metrics="acc", patience=50,  # 如果效果不好建议先把patience改大再试
   larger_is_better=True, eps=1e-5, save_path="model.pkl",
   only_save_best=True)  # 使用早停训练模型
cp = TM.CheckPerformance([x_test, y_test], metrics="acc", name="Test")  # 训练过程中监控测试集ACC
T = TM.Trainer(model, n_rules, task_out_dim, optimizer, criterion, device="cpu", callbacks=[es, cp], verbose=1) # 定义训练器, device 代表训练的使用cpu还是gpu，verbose=1显示log
T.fit(x_train, y_train, max_epoch=1000, w_l2=W_l2/(n_rules*in_dim), w_ur=W_ur, batch_size=batchsize)  # 模型训练，w_l2是L2正则化项系数，w_ur是标准正则化项系数，batch_size不能太大或太小，建议在32-512之间
train_time = time.perf_counter() - start
print("Train time used: %.2f s" % train_time)
# </editor-fold>

# <editor-fold desc="模型测试">
start = time.perf_counter()
model.load("model.pkl")  # 早停存储的模型路径
[y_pred1_prob, y_pred2_prob] = model.predict_score(x_test)  # 预测概率
y_pred1 = np.argmax(y_pred1_prob, axis=1)  # 概率转为类别结果
y_pred2 = np.argmax(y_pred2_prob, axis=1)  # 概率转为类别结果
elapsed = time.perf_counter() - start
print("Test Time used: %.6f s" % elapsed)
avr_predtime = 1000 * elapsed / x_test.shape[0]
print("Test samples number: {:d}, average predictive time: {:.6f} ms".format(x_test.shape[0], avr_predtime))
# </editor-fold>

# <editor-fold desc="Performance Evalution">
print("[------------------------------ 性能评估 ----------------------------------]")
Acc1 = accuracy_score(y_test[:, 0], y_pred1)  # 计算正确率
Acc2 = accuracy_score(y_test[:, 1], y_pred2)  # 计算正确率
print("Task1 Test ACC: {:.6f}           |    Task2 Test ACC: {:.6f}".format(Acc1, Acc2))
Precision1 = precision_score(y_test[:, 0], y_pred1, average='weighted')  # 精确率，weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量
Precision2 = precision_score(y_test[:, 1], y_pred2, average='weighted')  # 精确率，weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量
print("Task1 Test Precision: {:.6f}     |    Task2 Test Precision: {:.6f}".format(Precision1, Precision2))
Recall1 = recall_score(y_test[:, 0], y_pred1, average='weighted')  # 召回率，weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量
Recall2 = recall_score(y_test[:, 1], y_pred2, average='weighted')  # 召回率，weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量
print("Task1 Test Recall: {:.6f}        |    Task2 Test Recall: {:.6f}".format(Recall1, Recall2))
F1Score1 = f1_score(y_test[:, 0], y_pred1, average='weighted')  # F1值，weighted:各类别的F1 × 该类别的样本数量（实际值而非预测值）/ 样本总数量
F1Score2 = f1_score(y_test[:, 1], y_pred2, average='weighted')  # F1值，weighted:各类别的F1 × 该类别的样本数量（实际值而非预测值）/ 样本总数量
print("Task1 Test F1-Score: {:.6f}      |    Task2 Test F1-Score: {:.6f}".format(F1Score1, F1Score2))
y_test1_onehot = label_binarize(y_test[:, 0], classes = np.arange(task_out_dim[0]))  # 装换成类似二进制的编码
y_test2_onehot = label_binarize(y_test[:, 1], classes = np.arange(task_out_dim[1]))  # 装换成类似二进制的编码
AUC1 = roc_auc_score(y_test1_onehot, y_pred1_prob)  # AUC值
AUC2 = roc_auc_score(y_test2_onehot, y_pred2_prob)  # AUC值
print("Task1 Test AUC: {:.6f}           |    Task2 Test AUC: {:.6f}".format(AUC1, AUC2))

# 保存结果到csv文件
localtime = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
outcome_path1 = os.path.join("Outcome", data_name + "_task1" + ".csv")
outcome_path2 = os.path.join("Outcome", data_name + "_task2" + ".csv")
outcome1 = pd.DataFrame({'localtime':localtime, 'sample-num':X.shape[0], 'batch_size':batchsize, 'LR':LR, 'W_l2':W_l2, 'W_ur':W_ur, 'n_rules':n_rules, 'data':Xname, 'k-fold':k,
                             'Accuracy':Acc1, 'Precision':Precision1, 'Recall':Recall1, 'F1Score':F1Score1, 'AUC':AUC1, 'traintime(s)':train_time, 'testtime(ms)':avr_predtime, 'note':note},index=[1])
outcome2 = pd.DataFrame({'localtime':localtime, 'sample-num':X.shape[0], 'batch_size':batchsize, 'LR':LR, 'W_l2':W_l2, 'W_ur':W_ur, 'n_rules':n_rules, 'data':Xname, 'k-fold':k,
                             'Accuracy':Acc2, 'Precision':Precision2, 'Recall':Recall2, 'F1Score':F1Score2, 'AUC':AUC2, 'traintime(s)':train_time, 'testtime(ms)':avr_predtime, 'note':note},index=[1])
if os.path.exists(outcome_path1):
    outcome1.to_csv(outcome_path1, mode='a', header=False, index=False)
else:
    outcome1.to_csv(outcome_path1, mode='a', header=True, index=False)
if os.path.exists(outcome_path2):
    outcome2.to_csv(outcome_path2, mode='a',header=False, index=False)
else:
    outcome2.to_csv(outcome_path2, mode='a', header=True, index=False)

CM1 = confusion_matrix(y_test[:, 0], y_pred1)  # 混淆矩阵
CM2 = confusion_matrix(y_test[:, 1], y_pred2)  # 混淆矩阵
CM1 = 100 * CM1.astype('float') / CM1.sum(axis=1)[:, np.newaxis]     # 归一化
CM2 = 100 * CM2.astype('float') / CM2.sum(axis=1)[:, np.newaxis]     # 归一化
# np.savetxt("CM1.csv", CM1, delimiter=',')
# np.savetxt("CM2.csv", CM2, delimiter=',')
CM1 = pd.DataFrame(CM1,columns=["LS","WR-L","RS","WL-R"],index=["LS","WR-L","RS","WL-R"])
CM2 = pd.DataFrame(CM2,columns=["LW","RA","RD","SA","SD"],index=["LW","RA","RD","SA","SD"])  # level walking, ramp ascent, ramp descent, stair ascent and stair descent.
plt.rc('font',family='Times New Roman') # 设置字体为Times New Roman
plt.figure('Task1: Gait Phase', dpi=150, figsize=(6.5, 5))
sns.heatmap(CM1, annot=True, square=True, cmap='Greys', fmt='.2f', cbar=True)  # cmap，热力图颜色，cbar，是否画一个颜色条
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
CM1_savepath = "./Outcome/ConfusionMatrix/" + localtime + data_name + "_CM1"
plt.savefig(CM1_savepath)
plt.figure('Task2: Locomotion Mode', dpi=150, figsize=(6.5, 5))
sns.heatmap(CM2, annot=True, square=True, cmap='Greys', fmt='.2f', cbar=True)  # cmap，热力图颜色，cbar，是否画一个颜色条
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
CM2_savepath = "./Outcome/ConfusionMatrix/" + localtime + data_name + "_CM2"
plt.savefig(CM2_savepath)
# plt.show()
# </editor-fold>
