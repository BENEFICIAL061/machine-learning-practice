import numpy as np
from dlframe import Logger
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from hmmlearn import hmm

class KNN:
    def __init__(self):
        super().__init__()
        self.clf = KNeighborsClassifier(n_neighbors=3)

    def train(self,trainData,trainTarget):
        self.clf.fit(trainData, trainTarget)

    def test(self,testData):
        return self.clf.predict(testData)
class GNB:
    #原理解析 高斯——朴素贝叶斯(适用于数值特征分类) https: // blog.csdn.net / zcz0101 / article / details / 109577494
    def __init__(self):
        super().__init__()
        self.clf = GaussianNB()

    def train(self,trainData,trainTarget):
        self.clf.fit(trainData, trainTarget)

    def test(self,testData):
        return self.clf.predict(testData)
class KM:
    def __init__(self):
        super().__init__()
        self.clf = KMeans(n_clusters=2, random_state=0)

    def train(self, trainData, trainTarget):
        self.clf.fit(trainData, trainTarget)

    def test(self, testData):
        return self.clf.predict(testData)
class CART:
    def __init__(self):
        super().__init__()
        self.clf = DecisionTreeClassifier(criterion='gini', random_state=42)

    def train(self, trainData, trainTarget):
        self.clf.fit(trainData, trainTarget)

    def test(self, testData):
        return self.clf.predict(testData)
class SVM:
    def __init__(self):
        super().__init__()
        self.clf =  SVC(kernel='linear', random_state=42)

    def train(self, trainData, trainTarget):
        scaler = StandardScaler()
        trainData=scaler.fit_transform(trainData)
        self.clf.fit(trainData, trainTarget)

    def test(self, testData):
        scaler = StandardScaler()
        testData = scaler.fit_transform(testData)
        return self.clf.predict(testData)
class LR:
    def __init__(self):
        super().__init__()
        self.lr = LogisticRegression()

    def train(self, trainData, trainTarget):
        scaler = StandardScaler()
        trainData=scaler.fit_transform(trainData)
        self.lr.fit(trainData, trainTarget)

    def test(self, testData):
        scaler = StandardScaler()
        testData = scaler.fit_transform(testData)
        return self.lr.predict(testData)
class ME:
    def __init__(self):
        # 初始化 MaxEnt 模型，需要传递特征函数和类别数量
        self.clf = MaxEnt()

    def train(self, trainData, X_coloums,trainTarget,lable):
        # 直接调用 MaxEnt 的 fit 方法来训练模型
        self.clf.fit(trainData, X_coloums,trainTarget,lable)

    def test(self, testData):
        # 使用训练好的模型进行预测
        return self.clf.predict(testData)
class BA:
    def __init__(self):
        super().__init__()
        for depth in [1, 2, 10]:
            self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))

    def train(self, trainData, trainTarget):

        self.clf.fit(trainData, trainTarget)

    def test(self, testData):
        return self.clf.predict(testData)

class EM:
    def __init__(self):
        super().__init__()
        self.clf = GMM_EM(n_components=3)

    def train(self, trainData, trainTarget):
        self.clf.fit(trainData)

    def test(self, testData):
        return self.clf.predict(testData)
class GMM_EM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components  # 混合模型中高斯分布的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛的容忍度
        self.weights = None  # 每个高斯分布的权重
        self.means = None  # 每个高斯分布的均值
        self.covariances = None  # 每个高斯分布的协方差矩阵

    def initialize_parameters(self, X):
        """随机初始化参数"""
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        random_indices = np.random.permutation(n_samples)
        self.means = X[random_indices[:self.n_components]]
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])

    def e_step(self, X):
        """E步：计算隐变量的期望"""
        n_samples, n_features = X.shape
        resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def m_step(self, X, resp):
        """M步：最大化期望对数似然"""
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)
        self.weights = Nk / n_samples
        self.means = np.dot(resp.T, X) / Nk[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot((resp[:, k][:, np.newaxis] * diff).T, diff) / Nk[k]

    def log_likelihood(self, X):
        """计算对数似然"""
        log_lik = 0
        for k in range(self.n_components):
            log_lik += self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        log_lik = np.log(log_lik).sum()
        return log_lik

    def fit(self, X):
        """拟合模型"""
        self.initialize_parameters(X)
        log_likelihoods = []
        for i in range(self.max_iter):
            resp = self.e_step(X)
            self.m_step(X, resp)
            log_lik = self.log_likelihood(X)
            log_likelihoods.append(log_lik)
            if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def predict(self, X):
        """预测样本所属的高斯分布"""
        resp = self.e_step(X)
        return np.argmax(resp, axis=1)

class HMM:
    def __init__(self):
        super().__init__()
        self.model = hmm.CategoricalHMM(
            n_components=5,
            n_iter=200,
            random_state=42,
            init_params='',
            tol=0.0001
        )

        self.monthly_weights = {
            1: {'多云': 1.0, '晴': 1.2, '阴': 1.1, '小雨': 0.8, '阵雨': 0.7},  # 一月
            2: {'多云': 1.1, '晴': 1.2, '阴': 1.0, '小雨': 0.8, '阵雨': 0.7},
            3: {'多云': 1.2, '晴': 1.1, '阴': 0.9, '小雨': 0.9, '阵雨': 0.8},
            4: {'多云': 1.1, '晴': 1.0, '阴': 1.0, '小雨': 1.0, '阵雨': 0.9},
            5: {'多云': 1.0, '晴': 0.9, '阴': 1.1, '小雨': 1.1, '阵雨': 1.0},
            6: {'多云': 0.9, '晴': 0.8, '阴': 1.1, '小雨': 1.2, '阵雨': 1.2},
            7: {'多云': 0.8, '晴': 0.7, '阴': 1.2, '小雨': 1.3, '阵雨': 1.3},
            8: {'多云': 0.9, '晴': 0.8, '阴': 1.1, '小雨': 1.2, '阵雨': 1.2},
            9: {'多云': 1.0, '晴': 0.9, '阴': 1.1, '小雨': 1.1, '阵雨': 1.0},
            10: {'多云': 1.1, '晴': 1.0, '阴': 1.0, '小雨': 1.0, '阵雨': 0.9},
            11: {'多云': 1.2, '晴': 1.1, '阴': 0.9, '小雨': 0.9, '阵雨': 0.8},
            12: {'多云': 1.1, '晴': 1.2, '阴': 1.0, '小雨': 0.8, '阵雨': 0.7}
        }

    def _init_transition_matrix(self, states, n_states):
        """简化版的转移矩阵初始化"""
        trans_mat = np.zeros((n_states, n_states))

        # 统计转移次数
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            trans_mat[current_state, next_state] += 1

        # 归一化
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_mat = np.where(row_sums > 0, trans_mat / row_sums, 1.0 / n_states)

        # 添加平滑
        trans_mat = trans_mat * 0.95 + np.ones_like(trans_mat) * 0.05 / n_states

        return trans_mat

    def train(self, train_data):
        """训练模型"""
        try:
            train_data = np.array(train_data).reshape(-1)

            # 使用进度信息替代进度条
            trans_mat = self._init_transition_matrix(train_data, n_states=5)

            self.model.transmat_ = trans_mat
            self.model.startprob_ = np.mean(trans_mat, axis=0)

            emission_mat = np.eye(5) * 0.85 + np.ones((5, 5)) * 0.15 / 5
            self.model.emissionprob_ = emission_mat

            train_data = train_data.reshape(-1, 1)
            self.model.fit(train_data)


        except Exception as e:
            raise e

    def test(self, test_data):
        """预测"""
        try:
            test_data = np.array(test_data).reshape(-1, 1)
            return self.model.predict(test_data)
        except Exception as e:
            print(f"预测过程出错: {e}")
            raise e
class MaxEnt:
    def __init__(self, lr=0.0001):
        """
        最大熵模型的实现，为了方便理解，尽可能的将参数都存储为字典形式
        :param lr: 学习率，默认值为0.0001

        其他参数：
        :param w: 模型的参数，字典
        :param N: 样本数量
        :param label: 标签空间
        :param hat_p_x: 边缘分布P(X)的经验分布
        :param hat_p_x_y: 联合分布P(X,Y)的经验分布
        :param E_p: 特征函数f(x,y)关于模型P(X|Y)与经验分布hatP(X)的期望值
        :param E_hat_p: 特征函数f(x,y)关于经验分布hatP(X,Y)的期望值
        :param eps: 一个接近于0的正数极小值，这个值放在log的计算中，防止报错
        """
        self.lr = lr
        self.params = {'w': None}

        self.N = None
        self.label = None

        self.hat_p_x = {}
        self.hat_p_x_y = {}

        self.E_p = {}
        self.E_hat_p = {}

        self.eps = np.finfo(np.float32).eps

    def _init_params(self):
        """
        随机初始化模型参数w
        :return:
        """
        w = {}
        for key in self.hat_p_x_y.keys():
            w[key] = np.random.rand()
        self.params['w'] = w

    def _rebuild_X(self, X):
        """
        为了自变量的差异化处理，重新命名自变量
        :param X: 原始自变量
        :return:
        """
        X_result = []
        for x in X:
            X_result.append([y_s + '_' + x_s for x_s, y_s in zip(x, self.X_columns)])
        return X_result

    def _build_mapping(self, X, Y):
        """
        求取经验分布，参照公式(1)(2)
        :param X: 训练样本的输入值
        :param Y: 训练样本的输出值
        :return:
        """
        for x, y in zip(X, Y):
            for x_s in x:
                if x_s in self.hat_p_x.keys():
                    self.hat_p_x[x_s] += 1
                else:
                    self.hat_p_x[x_s] = 1
                if (x_s, y) in self.hat_p_x_y.keys():
                    self.hat_p_x_y[(x_s, y)] += 1
                else:
                    self.hat_p_x_y[(x_s, y)] = 1

        self.hat_p_x = {key: count / self.N for key, count in self.hat_p_x.items()}
        self.hat_p_x_y = {key: count / self.N for key, count in self.hat_p_x_y.items()}

    def _cal_E_hat_p(self):
        """
        计算特征函数f(x,y)关于经验分布hatP(X,Y)的期望值，参照公式(3)
        :return:
        """
        self.E_hat_p = self.hat_p_x_y

    def _cal_E_p(self, X):
        """
        计算特征函数f(x,y)关于模型P(X|Y)与经验分布hatP(X)的期望值，参照公式(4)
        :param X:
        :return:
        """
        for key in self.params['w'].keys():
            self.E_p[key] = 0
        for x in X:
            p_y_x = self._cal_prob(x)
            for x_s in x:
                for (p_y_x_s, y) in p_y_x:
                    if (x_s, y) not in self.E_p.keys():
                        continue
                    self.E_p[(x_s, y)] += (1 / self.N) * p_y_x_s

    def _cal_p_y_x(self, x, y):
        """
        计算模型条件概率值，参照公式(9)的指数部分
        :param x: 单个样本的输入值
        :param y: 单个样本的输出值
        :return:
        """

        sum = 0.0
        for x_s in x:
            sum += self.params['w'].get((x_s, y), 0)
        return np.exp(sum), y

    def _cal_prob(self, x):
        """
        计算模型条件概率值，参照公式(9)
        :param x: 单个样本的输入值
        :return:
        """
        p_y_x = [(self._cal_p_y_x(x, y)) for y in self.label]
        sum_y = np.sum([p_y_x_s for p_y_x_s, y in p_y_x])
        return [(p_y_x_s / sum_y, y) for p_y_x_s, y in p_y_x]

    def fit(self, X, X_columns, Y, label, max_iter=20000):
        """
        模型训练入口
        :param X: 训练样本输入值
        :param X_columns: 训练样本的columns
        :param Y: 训练样本的输出值
        :param label: 训练样本的输出空间
        :param max_iter: 最大训练次数
        :return:
        """
        self.N = len(X)
        self.label = label
        self.X_columns = X_columns

        X = self._rebuild_X(X)

        self._build_mapping(X, Y)

        self._cal_E_hat_p()

        self._init_params()

        for iter in range(max_iter):

            self._cal_E_p(X)

            for key in self.params['w'].keys():
                sigma = self.lr * np.log(self.E_hat_p.get(key, self.eps) / self.E_p.get(key, self.eps))
                self.params['w'][key] += sigma

    def predict(self, X):
        """
        预测结果
        :param X: 样本
        :return:
        """
        X = self._rebuild_X(X)
        result_list = []

        for x in X:
            max_result = 0
            y_result = self.label[0]
            p_y_x = self._cal_prob(x)
            for (p_y_x_s, y) in p_y_x:
                if p_y_x_s > max_result:
                    max_result = p_y_x_s
                    y_result = y
            result_list.append((y_result))
        return result_list
































