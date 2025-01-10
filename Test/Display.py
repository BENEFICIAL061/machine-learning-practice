import pandas as pd
from dlframe import WebManager, Logger
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score, \
    silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

import Model as model
import DataLoader






# "load_diabetes",
# "load_digits",
# "load_files",
# "load_iris",
# "load_breast_cancer",
# "load_linnerud",
# "load_sample_image",
# "load_sample_images",
# "load_svmlight_file",
# "load_svmlight_files",
# "load_wine",





# 数据集
class TestDataset:
    def __init__(self, dataloader,name) -> None:
        super().__init__()
        self.name=name
        if self.name == 'HMMDataSet':
            df = dataloader
            encoder = LabelEncoder()
            self.data = encoder.fit_transform(df['天气状态'].values)
            self.target = self.data  # HMM中目标就是序列本身
            self.dates = df['日期'].values
            self.encoder = encoder  # 保存编码器以供后续使用
        else:
            self.data = dataloader.data
            self.target= dataloader.target
            self.label=dataloader.target_names
            self.feature=dataloader.feature_names
        self.logger = Logger.get_logger('TestDataset')
        self.logger.print("My Dataset is {}".format(name))


    # def __len__(self) -> int:
    #     return len(self.data)
    #
    # def __getitem__(self, idx: int):
    #     return self.data[idx]



# 数据集切分器
class TestSplitter:
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset,data,target,name1,name2):

        locker=isMatch(name1, name2)

        if name1=='ME'and locker:#编码转化为原标签
            labels = []
            Y=[]
            for x in dataset.label:
                labels.append(x)
            for item in target:
                Y.append(labels[item])
            target=Y
        if name1 == 'HMM'and locker:
            # 特殊处理HMM数据 - 使用最后30条作为测试集
            total_length = len(dataset.data)
            test_size = 30  # 固定使用最后30条数据作为测试集

            # 分割训练集和测试集
            train_data = dataset.data[:-test_size]
            test_data = dataset.data[-test_size:]
            train_target = dataset.target[:-test_size]
            test_target = dataset.target[-test_size:]

            # 确保数据是numpy数组
            train_data = np.array(train_data)
            test_data = np.array(test_data)
            train_target = np.array(train_target)
            test_target = np.array(test_target)

            self.logger.print(f"HMM数据分割: 训练集 {len(train_data)}条, 测试集 {test_size}条")
            return train_data, test_data, train_target, test_target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1-self.ratio, random_state=1)

        if locker :

            self.logger.print("split!")
            self.logger.print("training shape = {}".format(len(X_train)))
        else:
            self.logger.print("所选模型不适用于所选数据集")

        if name1 == 'HMM' and name2 == 'HMMDataSet':
            return data, data, target, target
        return X_train, X_test, y_train, y_test

# 模型
class TestModel:
    def __init__(self, name) -> None:
        super().__init__()
        self.name=name
        self.model=self.switch()
        self.logger = Logger.get_logger('TestModel')

    def train(self,dataset, trainData,trainTarget,name1,name2)-> None :
        locker = isMatch(name1, name2)
        if locker:
            self.logger.print("trainingModel={},  trainDataset = {}".format(self.name,name2))
            if name1=='HMM' and name2=='HMMDataSet':
                self.model.train(trainData)
            elif name1=='ME':
                Y=[]
                labels=[]
                for x in dataset.label:
                    labels.append(x)

                print(trainData)
                self.model.train(trainData,dataset.feature,trainTarget,labels)
            else:
                self.model.train(trainData,trainTarget)
        else:
            self.logger.print("所选模型不适用于所选数据集")


    def test(self, testData,name1,name2):
        locker = isMatch(name1, name2)
        if locker:
            self.logger.print("testing")
            return self.model.test(testData)
        else:
            self.logger.print("所选模型不适用于所选数据集")
            return 0

    def switch(self):
        if self.name=='KNN':
            model_=model.KNN()
        elif self.name=='GNB':
            model_=model.GNB()
        elif self.name=='KM':
            model_=model.KM()
        elif self.name=='CART':
            model_=model.CART()
        elif self.name=='SVM':
            model_=model.SVM()
        elif self.name == 'LR':
            model_ = model.LR()
        elif self.name == 'BA':
            model_ = model.BA()
        elif self.name == 'EM':
            model_ = model.EM()
        elif self.name=='HMM':
            model_ = model.HMM()
        elif self.name == 'ME':
            model_ = model.ME()
        else:
            model_=None
        return model_


# 结果判别器
class TestJudger:
    def __init__(self,name) -> None:
        super().__init__()
        self.name=name
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, testTarget, name1, name2) -> None:
        locker = isMatch(name1, name2)

        if locker:
            if (name1=='KM' and self.name!='SC')or(name1!='KM' and self.name=='SC'):
                self.logger.print("所选评价指标不适用")
            elif name1=='ME' and self.name=='ROC':
                self.logger.print("所选评价指标不适用")
            elif self.name == 'Confusion':
                self.confusion(y_hat, testTarget)
            elif self.name == 'Acc':
                self.accuracy(y_hat, testTarget,name1,name2)
            elif self.name == 'Precision':
                self.precision(y_hat, testTarget)
            elif self.name == 'Recall':
                self.recall(y_hat, testTarget)
            elif self.name == 'F1':
                self.F1(y_hat, testTarget)
            elif self.name == 'ROC':
                self.ROC(y_hat, testTarget)
            elif self.name == 'SC':
                self.SC(y_hat, testTarget)
        else:
            self.logger.print("所选模型不适用于所选数据集")
    def accuracy(self,y_hat, testTarget,name1,name2)-> None:
        correct=0
        if name1=='ME':
            for i in range(len(y_hat)):
                if y_hat[i]==testTarget[i]:
                    correct+=1
            self.logger.print("Accuracy is: %.3f" % (correct / len(y_hat)))
        elif name1=='HMM':
            self.HMMAccuracy(y_hat, testTarget)
        else:
            correct = np.count_nonzero((y_hat == testTarget) == True)
            self.logger.print ("Accuracy is: %.3f" %(correct/len(y_hat)))
        # print(correct)


        # 混淆矩阵

    def confusion(self, y_hat, testTarget) -> None:
        confusion = confusion_matrix(y_hat, testTarget)
        # 设置类别标签
        class_names = list(set(testTarget))  # 获取数据集中的类别名称
        # 绘制混淆矩阵热图
        plt.figure(figsize=(6, 6))  # 创建一个指定大小的画布
        ax = sns.heatmap(confusion, annot=True, xticklabels=class_names, yticklabels=class_names, cmap='Blues', fmt="d")
        # 使用seaborn库中的heatmap函数绘制混淆矩阵的热图
        # annot=True表示在热图中显示数值，xticklabels和yticklabels分别设置x轴和y轴的标签，cmap设置颜色映射，fmt设置数值格式
        # 设置字体大小
        ax.set_xticklabels(class_names, fontsize=10)  # 设置x轴标签字体大小
        ax.set_yticklabels(class_names, fontsize=10)  # 设置y轴标签字体大小

        plt.xlabel('y_hat', fontsize=14)  # 设置x轴标签
        plt.ylabel('testTarget', fontsize=14)  # 设置y轴标签
        plt.show()  # 显示图形

    def precision(self, y_hat, testTarget) -> None:

        PRE = precision_score(testTarget, y_hat, average='macro')
        self.logger.print("Precision is: %.3f" % PRE)

    def recall(self, y_hat, testTarget) -> None:

        REC = recall_score(testTarget, y_hat, average='macro')
        self.logger.print("Recall is: %.3f" % REC)

    def F1(self, y_hat, testTarget) -> None:

        macro_f1 = f1_score(testTarget, y_hat, average='macro')
        self.logger.print("F1 is: %.3f" % macro_f1)

    def HMMAccuracy(self, y_hat, test_target):
        """HMM模型的准确率计算"""
        try:
            # 重新加载数据以获取编码器和日期
            df = pd.read_csv('HMMdataSet/GWeatherDataset2019.csv')
            df['日期'] = pd.to_datetime(df['日期'], format='%Y年%m月%d日')
            df['天气状态'] = df['天气'].apply(lambda x: x.split('/')[0])
            top_weather = df['天气状态'].value_counts().nlargest(5).index

            # 按日期排序并采样
            df = df.sort_values('日期')
            df = df[df['天气状态'].isin(top_weather)]
            df = df.iloc[::10].copy()

            # 获取最后30天的日期
            test_dates = df['日期'].values[-30:]

            encoder = LabelEncoder()
            encoder.fit(top_weather)

            correct_count = 0
            results = []

            for date, true, pred in zip(test_dates, test_target, y_hat):
                true_weather = encoder.inverse_transform([true])[0]
                pred_weather = encoder.inverse_transform([pred])[0]
                is_correct = true == pred
                if is_correct:
                    correct_count += 1
                results.append((date, true_weather, pred_weather, is_correct))

            accuracy = correct_count / len(test_target)

            # 使用logger输出结果
            self.logger.print(f"\n预测准确率: {accuracy:.2%}\n")


            for date, true_weather, pred_weather, is_correct in results:
                result_symbol = "✓" if is_correct else "✗"
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                self.logger.print(f"{date_str}  {true_weather:<8}{pred_weather:<8}{result_symbol}")

        except Exception as e:
            self.logger.print(f"评估过程出错: {e}")
            raise e

    def SC(self, y_hat, testTarget) -> None:
        score = silhouette_score(testTarget.reshape(-1, 1), y_hat.reshape(-1, 1))
        self.logger.print("Silhouette Coefficient is: %.3f" % score)

    def ROC(self, y_hat,test_target):
        fpr, tpr, threshold = roc_curve(test_target, y_hat, pos_label=1)
        roc_auc = auc(fpr,tpr)  # 准确率代表所有正确的占所有数据的比值
        self.logger.print("AUC is: %.3f" % roc_auc)
        lw = 2
        plt.subplot(1,1,1)
        plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1 - specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC', y=0.1)
        plt.legend(loc="lower right")
        plt.show()
# 匹配判别
def isMatch(modelName, datasetName):
    lock=True
    if modelName=='ME' and datasetName!='MEDataSet':
        lock=False
    elif modelName!='ME' and datasetName=='MEDataSet':
        lock=False
    elif modelName == 'HMM' and datasetName != 'HMMDataSet':
        lock = False
    elif modelName != 'HMM' and datasetName == 'HMMDataSet':
        lock = False
    elif modelName=='SVM'and datasetName=='linnerud':
        return False
    elif modelName=='EM'and datasetName!='iris':
        return False
    else:
        lock=True
    return lock




if __name__ == '__main__':

    #数据集加载
    #host='0.0.0.0',port=3000,
    print('如果你启动了自己的前端服务器，请点击该链接: http://localhost:3000/')
    #前端显示
    with WebManager(parallel=False) as manager:

        dataset = manager.register_element('数据集', {'iris': TestDataset(DataLoader.select('iris'),'iris'),
                                                      'wine': TestDataset(DataLoader.select('wine'),'wine'),
                                                      'breast_cancer':TestDataset(DataLoader.select('breast_cancer'),'breast_cancer'),
                                                      'HMMDataSet':TestDataset(DataLoader.select('HMMDataSet'),'HMMDataSet'),
                                                      'MEDataSet':TestDataset(DataLoader.select('MEDataSet'),'MEDataSet'),
                                                      'digits':TestDataset(DataLoader.select('digits'),'digits')
                                                      # 'linnerud':TestDataset(DataLoader.select('linnerud'),'linnerud'),# 'diabetes':TestDataset(DataLoader.select('diabetes'),'diabetes')
                                                     })
        splitter = manager.register_element('数据分割', {'ratio:0.8': TestSplitter(0.8), 'ratio:0.6': TestSplitter(0.6), 'ratio:0.4': TestSplitter(0.4), 'ratio:0.2': TestSplitter(0.2)})
        model = manager.register_element('模型', {'K-邻近(KNN)': TestModel('KNN'),
                                                  '高斯—朴素贝叶斯(GNB)': TestModel('GNB'),'K—means(KM)': TestModel('KM'),
                                                  '决策树CART(CART)': TestModel('CART'),'支持向量机(SVM)': TestModel('SVM'),
                                                  '逻辑回归(LR)': TestModel('LR'),'Boosting&AdaBoost(BA)': TestModel('BA'),
                                                  'EM算法(EM)': TestModel('EM'),'隐马尔可夫(HMM)': TestModel('HMM'),
                                                  '最大熵(ME)':TestModel('ME')})
        judger = manager.register_element('评价指标', {'混淆矩阵': TestJudger('Confusion'),
                                                       'ACC': TestJudger('Acc'),
                                                       'PRE': TestJudger('Precision'),
                                                       'RECALL': TestJudger('Recall'),
                                                       'F1': TestJudger('F1'),
                                                       'ROC': TestJudger('ROC'),
                                                       '轮廓系数(SC)': TestJudger('SC')})


        train_data_test_data = splitter.split(dataset,dataset.data,dataset.target,model.name,dataset.name)

        train_data, test_data,train_target,test_target = train_data_test_data[0], train_data_test_data[1],train_data_test_data[2], train_data_test_data[3]
        model.train(dataset,train_data,train_target,model.name,dataset.name)
        y_hat = model.test(test_data,model.name,dataset.name)
        judger.judge(y_hat, test_target,model.name,dataset.name)
