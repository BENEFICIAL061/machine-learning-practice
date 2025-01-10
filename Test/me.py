import pandas as pd


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

data=Data_HMM()
print(data)