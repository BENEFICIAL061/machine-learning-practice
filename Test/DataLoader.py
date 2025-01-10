from sklearn import datasets
import pandas as pd
import numpy as np



def select(name):
    if name=='iris' :
        return datasets.load_iris()
    elif name=='wine' :
        return datasets.load_wine()
    elif name=='breast_cancer':
        return datasets.load_breast_cancer()
    elif name=='digits':
        return datasets.load_digits()
    # elif name=='diabetes':#糖尿病
    #     return datasets.load_diabetes()
    # elif name=='linnerud':
    #     return datasets.load_linnerud()
    elif name=='HMMDataSet':
        return Data_HMM()
    elif name=='MEDataSet':
        data_set = [['youth', 'no', 'no', '1'],
                    ['youth', 'no', 'no', '2'],
                    ['youth', 'yes', 'no', '2'],
                    ['youth', 'yes', 'yes', '1'],
                    ['youth', 'no', 'no', '1'],
                    ['mid', 'no', 'no', '1'],
                    ['mid', 'no', 'no', '2'],
                    ['mid', 'yes', 'yes', '2'],
                    ['mid', 'no', 'yes', '3'],
                    ['mid', 'no', 'yes', '3'],
                    ['elder', 'no', 'yes', '3'],
                    ['elder', 'no', 'yes', '2'],
                    ['elder', 'yes', 'no', '2'],
                    ['elder', 'yes', 'no', '3'],
                    ['elder', 'no', 'no', '1'],
                    ]
        target=[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
        labels=['refuse','agree']
        feature = ['age', 'working', 'house', 'credit_situation']
        dataloader = myDataLoader(data_set, target, labels, feature)
        return dataloader



def generate_observations(hidden_states):
    observations = []
    emission_probs_sunny = [0.6, 0.3, 0.1]  # Sunny 状态下观测的概率
    emission_probs_rainy = [0.1, 0.4, 0.5]  # Rainy 状态下观测的概率

    for state in hidden_states:
        if state == 0:  # Sunny
            obs_index = np.random.choice(3, p=emission_probs_sunny)
        else:  # Rainy
            obs_index = np.random.choice(3, p=emission_probs_rainy)
        observations.append(obs_index)

    return np.array(observations)


def Data_HMM():
    """加载HMM数据"""
    try:
        df = pd.read_csv('HMMdataSet/GWeatherDataset2019.csv')

        # 数据预处理
        df['日期'] = pd.to_datetime(df['日期'], format='%Y年%m月%d日')
        df['天气状态'] = df['天气'].apply(lambda x: x.split('/')[0])

        # 只保留最常见的5种天气状态
        top_weather = df['天气状态'].value_counts().nlargest(5).index
        df = df[df['天气状态'].isin(top_weather)]

        # 按日期排序并采样
        df = df.sort_values('日期')
        df = df.iloc[::10].copy()  # 每10条数据取1条

        print(f"加载数据总量: {len(df)}条")
        return df
    except Exception as e:
        print(f"HMM数据加载错误: {e}")
        return None

class myDataLoader:
    def __init__(self,data,target,label,feature_names):
        super().__init__()
        self.data=data
        self.target=target
        self.target_names=label
        self.feature_names=feature_names