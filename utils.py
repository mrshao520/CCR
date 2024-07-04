import os
import datasets
import numpy as np
import pandas as pd

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def evaluate(method, classifier, output_file, type=None):
    """评估给定分类器再不同数据集上的性能，并将结果保存到指定文件中

    Args:
        method (_type_): 用于数据增强的类
        classifier (_type_): 用于分类的机器学习模型
        output_file (_type_): 保存结果的文件名
        type (_type_, optional): 数据集类型. Defaults to None.
    """
    names = []
    partitions = []
    accuracies = []
    precisions = []
    recalls = []
    f_measures = []
    aucs = []
    g_means = []

    for name, folds in datasets.load_all(type).items():
        # name 数据集名称        folds 数据集
        # print(f"{name} - {len(folds)}")
        for i in range(len(folds)):
            # 训练集             测试集
            (X_train, y_train), (X_test, y_test) = folds[i]
            # 0   1
            labels = np.unique(y_test)
            # 统计分类总数
            counts = [len(y_test[y_test == label]) for label in labels]

            minority_class = labels[np.argmin(counts)]

            # 确定是二分类问题
            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            if method.__class__ in [SMOTE, SMOTEENN, SMOTETomek]:
                # 设置 k 邻域值
                method.k = method.k_neighbors = np.min(
                    [len(y_train[y_train == minority_class]) - 1, 5]
                )

            if method is not None:
                # 重采样 - 数据增强
                X_train, y_train = method.fit_resample(X_train, y_train)
            # 训练模型
            clf = classifier.fit(X_train, y_train)
            # 预测
            predictions = clf.predict(X_test)

            names.append(name)  # 数据集名称
            partitions.append(i)  # 第 i 个交叉验证集
            accuracies.append(metrics.accuracy_score(y_test, predictions))  # 准确率
            precisions.append(
                metrics.precision_score(y_test, predictions, pos_label=minority_class)
            )  # 查准率
            recalls.append(
                metrics.recall_score(y_test, predictions, pos_label=minority_class)
            )  # 查全率
            f_measures.append(metrics.f1_score(y_test, predictions))  # f_measures
            aucs.append(metrics.roc_auc_score(y_test, predictions))  # auc

            g_mean = 1.0

            for label in np.unique(y_test):
                idx = y_test == label
                g_mean *= metrics.accuracy_score(y_test[idx], predictions[idx])

            g_mean = np.sqrt(g_mean)
            g_means.append(g_mean)

    results_path = os.path.join(os.path.dirname(__file__), "self_results")

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    output_path = os.path.join(os.path.dirname(__file__), "self_results", output_file)
    df = pd.DataFrame(
        {
            "dataset": names,
            "partition": partitions,
            "accuracy": accuracies,
            "precision": precisions,
            "recall": recalls,
            "f-measure": f_measures,
            "auc": aucs,
            "g-mean": g_means,
        }
    )
    df = df[
        [
            "dataset",
            "partition",
            "accuracy",
            "precision",
            "recall",
            "f-measure",
            "auc",
            "g-mean",
        ]
    ]
    df.to_csv(output_path, index=False)


def compare(output_files):
    """
    比较多个输出文件中不同数据评估指标，将结果汇总到一个数据框中
    """
    dfs = {}  # 存储读取的CSV文件数据
    results = {}  # 存储每个评估指标的结果
    summary = {}  # 存储每个评估指标的总结数据
    tables = {}  # 存储生成的表格数据

    for f in output_files:
        path = os.path.join(os.path.dirname(__file__), "self_results", f)
        dfs[f] = pd.read_csv(path)

    datasets = list(dfs.values())[0]["dataset"].unique()  # 数据集名称
    measures = ["accuracy", "precision", "recall", "f-measure", "auc", "g-mean"]

    for measure in measures:
        results[measure] = {}
        summary[measure] = {}
        tables[measure] = []

        for dataset in datasets:
            results[measure][dataset] = {}
            row = [dataset]

            for method in output_files:
                df = dfs[method]
                result = df[df["dataset"] == dataset][measure].mean()
                results[measure][dataset][method] = result
                row.append(result)

            tables[measure].append(row)

        for method in output_files:
            summary[measure][method] = 0

        tables[measure] = pd.DataFrame(
            tables[measure], columns=["dataset"] + output_files
        )

    for measure in measures:
        for dataset in datasets:
            best_method = max(
                results[measure][dataset], key=results[measure][dataset].get
            )
            summary[measure][best_method] += 1

    return summary, tables
