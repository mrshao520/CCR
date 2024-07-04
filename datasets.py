import os
import zipfile
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


# 路径
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
FOLDS_PATH = os.path.join(os.path.dirname(__file__), "folds")


def download(url):
    """下载文件"""
    name = url.split("/")[-1]
    download_path = os.path.join(DATA_PATH, name)
    # 检测是否存在文件夹
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    # 检测文件是否已下载
    if not os.path.exists(download_path):
        # 下载文件到指定路径
        urlretrieve(url, download_path)
    # 检查文件后缀名
    if not os.path.exists(download_path.replace(".zip", ".dat")):
        # 如果是 zip 格式，则解压到本地
        if name.endswith(".zip"):
            print(download_path)
            with zipfile.ZipFile(download_path) as zip:
                zip.extractall(DATA_PATH)
        else:
            raise Exception("Unrecognized file type.")


def encode(X, y, encode_features=True):
    """对特征矩阵X和标签向量y进行编码，确保所有的特征和标签都可以数值化

    Args:
        X (_type_): 特征矩阵
        y (_type_): 标签向量
        encode_features (bool, optional): True 对特征x进行编码. Defaults to True.
    """
    # fit 学习标签的分别，transform将标签转换为对应的整数编码
    # LabelEncoder 将标签 y 进行编码
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    # 对特征x进行编码
    if encode_features:
        encoded = []  # 存储编码后的特征
        # 循环遍历特征矩阵 x 的每一列
        for i in range(X.shape[1]):
            try:
                # 将特征矩阵 x 的第一行中第 i 列的元素转换为浮点数
                # 若成功，则说明这一列是数值型特征，不需要编码
                float(X[0, i])
                encoded.append(X[:, i])
            except:
                # 使用LabelEncoder将这一列的特征进行编码，转换为整数类型
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))

        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def partition(X, y):
    """将给定的特征集x和标签集y分割成5个交叉验证集 folds"""
    partitions = []

    for _ in range(5):
        folds = []  # 存储5个folds的索引
        # 指定folds的数量为2（即每个folds包含训练集和测试集）
        # 设置 shuffle=True 以便在分割时随机打乱数据
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        # 分割数据，返回训练集索引和测试集索引
        for train_idx, test_idx in skf.split(X, y):
            folds.append([train_idx, test_idx])

        partitions.append(folds)

    return partitions


def load(name, url=None, encode_features=True, remove_metadata=True, scale=True):
    """加载数据

    Args:
        name (_type_): 文件名
        url (_type_, optional): 下载路径url. Defaults to None.
        encode_features (bool, optional): _description_. Defaults to True.
        remove_metadata (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to True.

    Returns:
        list: [特征， 标签]
    """
    # 文件名
    file_name = "%s.dat" % name

    if url is not None:
        download(url)

    skiprows = 0
    # 如果需要移除元数据，则打开数据计算需要跳过的行数
    if remove_metadata:
        with open(os.path.join(DATA_PATH, file_name)) as f:
            for line in f:
                if line.startswith("@"):
                    skiprows += 1
                else:
                    break
    # 读取数据
    df = pd.read_csv(
        os.path.join(DATA_PATH, file_name),  # 文件路径
        header=None,  # 列名
        skiprows=skiprows,  # 跳过指定行数
        skipinitialspace=True,  # 指定空格不应该视为列分隔符
        sep=" *, *",  # 指定列分隔符是任意数量的空格包围的逗号
        na_values="?",  # 指定 ？ 作为缺失值
        engine="python",  # 使用Python解析器来读取文件
    )

    # 去除NaN值得行，并将剩余得数据转换成 Numpy 数组
    # matrix = df.dropna().as_matrix() DataFrame.as_matrix 已弃用。请改用 DataFrame.values GH C18458
    matrix = df.dropna().values

    # 对特征和标签进行编码
    X, y = matrix[:, :-1], matrix[:, -1]
    X, y = encode(X, y, encode_features)

    # 交叉验证文件保存路径
    partitions_path = os.path.join(
        FOLDS_PATH, file_name.replace(".dat", ".folds.pickle")
    )

    if not os.path.exists(FOLDS_PATH):
        os.mkdir(FOLDS_PATH)

    print(partitions_path)
    if os.path.exists(partitions_path):
        partitions = pickle.load(open(partitions_path, "rb"), encoding="iso-8859-1")
    else:
        partitions = partition(X, y)
        pickle.dump(partitions, open(partitions_path, "wb"))

    folds = []

    for i in range(5):
        for j in range(2):
            train_idx, test_idx = partitions[i][j]
            train_set = [X[train_idx], y[train_idx]]
            test_set = [X[test_idx], y[test_idx]]
            folds.append([train_set, test_set])

            if scale:
                # 对每个folds得训练集数据进行最小-最大缩放
                scaler = MinMaxScaler().fit(train_set[0])
                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

    return folds


def load_all(type=None):
    assert type in [None, "preliminary", "final"]

    urls = []
    # 打开 url 文件，将 url 添加到 urls
    for current_type in ["preliminary", "final"]:
        if type in [None, current_type]:
            with open(
                os.path.join(
                    os.path.dirname(__file__), "datasets_%s.txt" % current_type
                )
            ) as file:
                for line in file.readlines():
                    urls.append(line.rstrip())

    datasets = {}

    for url in urls:
        name = url.split("/")[-1].replace(".zip", "")
        datasets[name] = load(name, url)

    return datasets
