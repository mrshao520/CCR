import numpy as np

from sklearn.model_selection import StratifiedKFold


def distance(x, y):
    """曼哈顿距离"""
    return np.sum(np.abs(x - y))


def taxicab_sample(n, r):
    """出租车悖论：如果有无限数量的出租车在无限大的城市中随机分布，那么任意选择一个点并请求出租车，出租车距离你的平均距离是多少
    每个出租车都对应一个二维平面的点，而每次新出的出租车的位置是之前所有出租车位置绝对值的和加上一个在 [-r, r] 范围内的随机数
    确保新生成的出租车不会与已有的出租车重叠，同时满足总距离不超过 r 的条件

    Args:
        n (_type_): 要生成的出租车数量
        r (_type_): 出租车能够行驶的最大距离
    """
    sample = []

    for _ in range(n):
        # 计算新的出租车可以放置的最大距离 spread
        spread = r - np.sum([np.abs(x) for x in sample])
        sample.append(spread * (2 * np.random.rand() - 1))

    return np.random.permutation(sample)


class CCR:
    def __init__(self, energy=0.25, scaling=0.0, n=None):
        """combined cleaning and resampling algorithm

        Args:
            energy (float, optional):  energy 值. Defaults to 0.25.
            scaling (float, optional): energy 缩放比例. Defaults to 0.0.
            n (_type_, optional): 合成的样本数量. Defaults to None.
        """
        self.energy = energy
        self.scaling = scaling
        self.n = n

    def fit_resample(self, X, y):
        """重采样

        Args:
            X (tuple): 特征  (number, features)
            y (tuple): 标签  (number, )
        """
        # print(f"shape: {X.shape} - {y.shape}")
        classes = np.unique(y)  # 标签类别 [0.0, 1.0]
        sizes = [sum(y == c) for c in classes]  # 标签类别对应的总数 [少，多]
        # print(sizes)
        # 确保是二分类问题
        assert len(classes) == len(set(sizes)) == 2

        minority_class = classes[np.argmin(sizes)]  # 少数类
        majority_class = classes[np.argmax(sizes)]  # 多数类
        # print(f"classes: {np.argmin(sizes)} - {np.argmax(sizes)}")
        # print(f"class:   {minority_class} - {majority_class}")
        minority = X[y == minority_class]
        majority = X[y == majority_class]
        # print(f"number: {len(minority)} - {len(majority)}")

        if self.n is None:
            n = len(majority) - len(minority)  # 重采样到与多数类数量一致
        else:
            n = self.n
        # 缩放 energy
        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = np.zeros((len(minority), len(majority)))  # 存储距离向量

        for i in range(len(minority)):
            for j in range(len(majority)):
                # 计算每个少数样本和多数样本的距离
                distances[i][j] = distance(minority[i], majority[j])

        radii = np.zeros(len(minority))  # 存储每个少数点的邻域半径
        translations = np.zeros(majority.shape)  # 存储每个多数点的位移距离

        # 计算每个少数样本的clean区域
        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy  # remaining energy budget
            r = 0.0
            # 将距离进行升序排列，返回对应的索引
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            # 根据 energy 值、少数类样本和多数类样本之间的距离 确定半径
            while True:
                if current_majority == len(majority):
                    # 当遍历到最后一个多数类样本时
                    if current_majority == 0:
                        radius_change = remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change

                    break

                # 根据 energy 值设置半径
                radius_change = remaining_energy / (current_majority + 1.0)

                if (
                    distances[i, sorted_distances[current_majority]]
                    >= r + radius_change
                ):  # 如果距离最近的多数类样本大于 r + radius_change
                    r += radius_change

                    break
                else:  # 有多数类样本在半径区域内
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[
                            i, sorted_distances[current_majority - 1]
                        ]

                    radius_change = (
                        distances[i, sorted_distances[current_majority]] - last_distance
                    )
                    # 更新半径
                    r += radius_change
                    # 更新 energy 值
                    remaining_energy -= radius_change * (current_majority + 1.0)

                    # 遍历下一个多数类样本
                    current_majority += 1

            radii[i] = r

            # current_majority指在半径范围内的多数类样本
            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:  # 对多数类样本进行随机调整，避免除零错误
                    majority_point += (
                        1e-6 * np.random.rand(len(majority_point)) + 1e-6
                    ) * np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                # 计算位移向量
                translation = (r - d) / d * (majority_point - minority_point)
                # 累积每个多数点的位移向量
                translations[sorted_distances[j]] += translation

        # 累加
        majority += translations

        appended = []
        # 生成少数类样本
        for i in range(len(minority)):
            minority_point = minority[i]
            # 计算生成的合成样本数量
            synthetic_samples = int(
                np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n)
            )
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        return np.concatenate([majority, minority, appended]), np.concatenate(
            [
                np.tile([majority_class], len(majority)),
                np.tile([minority_class], len(minority) + len(appended)),
            ]
        )


class CCRSelection:
    def __init__(
        self,
        classifier,
        measure,
        n_splits=5,
        energies=(0.25,),
        scaling_factors=(0.0,),
        n=None,
    ):
        """通过交叉验证来选择最佳的特征缩放和能量参数，以便对给定的数据集进行特征选择

        Args:
            classifier (_type_): 分类器实例
            measure (_type_): 评估分类器在交叉验证上的性能
            n_splits (int, optional): 交叉验证数. Defaults to 5.
            energies (tuple, optional): 包含能量值的列表. Defaults to (0.25,).
            scaling_factors (tuple, optional): 包含特征缩放因子的列表. Defaults to (0.0,).
            n (_type_, optional): 生成的合成少数样本. Defaults to None.
        """
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.selected_energy = None
        self.selected_scaling = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_resample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for energy in self.energies:
            for scaling in self.scaling_factors:
                scores = []

                for train_idx, test_idx in self.skf.split(X, y):
                    # 重采样
                    X_train, y_train = CCR(
                        energy=energy, scaling=scaling, n=self.n
                    ).fit_resample(X[train_idx], y[train_idx])
                    # 训练模型
                    classifier = self.classifier.fit(X_train, y_train)
                    # 预测
                    predictions = classifier.predict(X[test_idx])
                    # 将分数保存
                    scores.append(self.measure(y[test_idx], predictions))

                score = np.mean(scores)

                if score > best_score:
                    # 保存最佳结果
                    self.selected_energy = energy
                    self.selected_scaling = scaling

                    best_score = score

        return CCR(
            energy=self.selected_energy, scaling=self.selected_scaling, n=self.n
        ).fit_resample(X, y)
