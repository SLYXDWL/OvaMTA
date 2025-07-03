import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,roc_auc_score
import scipy.stats as st
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import brier_score_loss
class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("\033[91mThere is a significant difference\033[0m")
        else:
            print("There is NO significant difference")

def better_CCcurve(yy, yyP, n_bootstrap=200, alpha=0.05):
    # yyP为模型预测值
    # yy为真实label
    p = yyP
    q = yy

    # 主lowess拟合
    smo = lowess(q, p, frac=2 / 3, it=0, delta=0.01 * (np.max(p) - np.min(p)))
    clf_score = brier_score_loss(q, p)

    # 自助法计算置信区间
    bootstrap_results = []
    n_samples = len(p)

    for _ in range(n_bootstrap):
        # 有放回抽样
        idx = np.random.choice(n_samples, n_samples, replace=True)
        p_boot = p[idx]
        q_boot = q[idx]

        # 对bootstrap样本进行lowess拟合
        smo_boot = lowess(q_boot, p_boot, frac=2 / 3, it=0,
                          delta=0.01 * (np.max(p_boot) - np.min(p_boot)))
        bootstrap_results.append(smo_boot)

    # 创建插值网格
    x_grid = np.linspace(np.min(p), np.max(p), 100)
    y_bootstrap = np.zeros((n_bootstrap, len(x_grid)))

    # 在网格上插值所有bootstrap曲线
    for i, smo_boot in enumerate(bootstrap_results):
        # 使用线性插值
        y_bootstrap[i] = np.interp(x_grid, smo_boot[:, 0], smo_boot[:, 1])

    # 计算置信区间
    lower_bound = np.percentile(y_bootstrap, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(y_bootstrap, 100 * (1 - alpha / 2), axis=0)

    # 绘制图形
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot()

    # 绘制置信区间（灰色填充）
    plt.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3)
    # label=f'{int(100*(1-alpha))}% Confidence Interval')

    # 绘制主校准曲线
    plt.plot(smo[:, 0], smo[:, 1], color='blue', linestyle='-',
             label=f"Brier Score = {clf_score:.3f}")

    # 绘制理想对角线
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.title('Calibration Curve with Confidence Interval')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')

    plt.show()

    # 返回结果供进一步分析
    return {
        'x_grid': x_grid,
        'calibration_curve': np.interp(x_grid, smo[:, 0], smo[:, 1]),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }



def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            # print(score)
            statistics[c][i] = score

    return statistics

def get_ss(y_true, y_pred,type_id=0):
    class_names=['Benign','Borderline','Malignant']
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    sensitivity, specificity = [], []
    for i in range(len(class_names)):
        TP, FP, TN, FN = 0, 0, 0, 0
        for j in range(len(y_true)):
            if y_true[j] == i and y_pred[j] == i:
                TP += 1
            elif y_true[j] != i and y_pred[j] == i:
                FP += 1
            elif y_true[j] != i and y_pred[j] != i:
                TN += 1
            elif y_true[j] == i and y_pred[j] != i:
                FN += 1
        if i==type_id:
            return TP / (TP + FN),TN / (TN + FP)

def score_ci(
        y_true,
        y_pred,
        score_fun,
        sample_weight=None,
        n_bootstraps=5000,
        confidence_level=0.95,
        seed=None,
        reject_one_class_samples=True,
):
    """
    Compute confidence interval for given score function based on labels and predictions using bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_pred: 1D list or array of predictions corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param confidence_level: Confidence level for computing confidence interval. (default: 0.95)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Score evaluated on labels and predictions, lower confidence interval, upper confidence interval, array of
    bootstrapped scores.
    """

    assert len(y_true) == len(y_pred)

    score = score_fun(y_true, y_pred)
    _, ci_lower, ci_upper, scores = score_stat_ci(
        y_true=y_true,
        y_preds=y_pred,
        score_fun=score_fun,
        sample_weight=sample_weight,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
        reject_one_class_samples=reject_one_class_samples,
    )

    score = round(score, 2)
    ci_lower = round(ci_lower, 2)
    ci_upper = round(ci_upper, 2)

    return score, ci_lower, ci_upper, scores


def score_stat_ci(
        y_true,
        y_preds,
        score_fun,
        stat_fun=np.mean,
        sample_weight=None,
        n_bootstraps=2000,
        confidence_level=0.95,
        seed=None,
        reject_one_class_samples=True,
):
    """
    Compute confidence interval for given statistic of a score function based on labels and predictions using
    bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_preds: A list of lists or 2D array of predictions corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param stat_fun: Statistic for which confidence interval is computed. (e.g. np.mean)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param confidence_level: Confidence level for computing confidence interval. (default: 0.95)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Mean score statistic evaluated on labels and predictions, lower confidence interval, upper confidence
    interval, array of bootstrapped scores.
    """

    y_true = np.array(y_true)
    y_preds = np.atleast_2d(y_preds)
    assert all(len(y_true) == len(y) for y in y_preds)

    np.random.seed(seed)
    scores = []
    for i in range(n_bootstraps):
        readers = np.random.randint(0, len(y_preds), len(y_preds))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        reader_scores = []
        for r in readers:
            if sample_weight is not None:
                reader_scores.append(
                    score_fun(y_true[indices], y_preds[r][indices], sample_weight=sample_weight[indices]))
            else:
                reader_scores.append(score_fun(y_true[indices], y_preds[r][indices]))
        scores.append(stat_fun(reader_scores))

    mean_score = np.mean(scores)
    sorted_scores = np.array(sorted(scores))
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    return mean_score, ci_lower, ci_upper, scores


def pvalue(
        y_true,
        y_pred1,
        y_pred2,
        score_fun,
        sample_weight=None,
        n_bootstraps=2000,
        two_tailed=True,
        seed=None,
        reject_one_class_samples=True,
):
    """
    Compute p-value for hypothesis that score function for model I predictions is higher than for model II predictions
    using bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_pred1: 1D list or array of predictions for model I corresponding to elements in y_true.
    :param y_pred2: 1D list or array of predictions for model II corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param two_tailed: Whether to use two-tailed test. (default: True)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Computed p-value, array of bootstrapped differences of scores.
    """

    assert len(y_true) == len(y_pred1)
    assert len(y_true) == len(y_pred2)

    return pvalue_stat(
        y_true=y_true,
        y_preds1=y_pred1,
        y_preds2=y_pred2,
        score_fun=score_fun,
        sample_weight=sample_weight,
        n_bootstraps=n_bootstraps,
        two_tailed=two_tailed,
        seed=seed,
        reject_one_class_samples=reject_one_class_samples,
    )


def pvalue_stat(
        y_true,
        y_preds1,
        y_preds2,
        score_fun,
        stat_fun=np.mean,
        compare_fun=np.subtract,
        sample_weight=None,
        n_bootstraps=2000,
        two_tailed=True,
        seed=None,
        reject_one_class_samples=True,
):
    """
    Compute p-value for hypothesis that given statistic of score function for model I predictions is higher than for
    model II predictions using bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_preds1: A list of lists or 2D array of predictions for model I corresponding to elements in y_true.
    :param y_preds2: A list of lists or 2D array of predictions for model II corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param stat_fun: Statistic for which p-value is computed. (e.g. np.mean)
    :param compare_fun: Function to determine relative performance. (default: score1 - score2)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param two_tailed: Whether to use two-tailed test. (default: True)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Computed p-value, array of bootstrapped differences of scores.
    """

    y_true = np.array(y_true)
    y_preds1 = np.atleast_2d(y_preds1)
    y_preds2 = np.atleast_2d(y_preds2)
    assert all(len(y_true) == len(y) for y in y_preds1)
    assert all(len(y_true) == len(y) for y in y_preds2)

    np.random.seed(seed)
    z = []
    for i in range(n_bootstraps):
        readers1 = np.random.randint(0, len(y_preds1), len(y_preds1))
        readers2 = np.random.randint(0, len(y_preds2), len(y_preds2))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        reader1_scores = []
        for r in readers1:
            if sample_weight is not None:
                reader1_scores.append(
                    score_fun(y_true[indices], y_preds1[r][indices], sample_weight=sample_weight[indices]))
            else:
                reader1_scores.append(score_fun(y_true[indices], y_preds1[r][indices]))
        score1 = stat_fun(reader1_scores)
        reader2_scores = []
        for r in readers2:
            if sample_weight is not None:
                reader2_scores.append(
                    score_fun(y_true[indices], y_preds2[r][indices], sample_weight=sample_weight[indices]))
            else:
                reader2_scores.append(score_fun(y_true[indices], y_preds2[r][indices]))
        score2 = stat_fun(reader2_scores)
        z.append(compare_fun(score1, score2))

    p = percentileofscore(z, 0.0, kind="weak") / 100.0
    if two_tailed:
        p *= 2.0
    return p, z


def get_sensi0(y_true, y_pred):
    i = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TP , (TP + FN))
    return TP / (TP + FN)


def get_speci0(y_true, y_pred):
    i = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TN , (TN + FP))
    return TN / (TN + FP)


def get_sensi1(y_true, y_pred):
    i = 1
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TP , (TP + FN))
    return TP / (TP + FN)


def get_speci1(y_true, y_pred):
    i = 1
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TN , (TN + FP))
    return TN / (TN + FP)


def get_sensi2(y_true, y_pred):
    i = 2
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TP , (TP + FN))
    return TP / (TP + FN)


def get_speci2(y_true, y_pred):
    i = 2
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(len(y_true)):
        if y_true[j] == i and y_pred[j] == i:
            TP += 1
        elif y_true[j] != i and y_pred[j] == i:
            FP += 1
        elif y_true[j] != i and y_pred[j] != i:
            TN += 1
        elif y_true[j] == i and y_pred[j] != i:
            FN += 1
    # print('分子分母：',TN , (TN + FP))
    return TN / (TN + FP)


def get_ss_kappa_acc(y_true, y_pred):
    class_names = ['Benign', 'Borderline', 'Malignant']
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitivity, specificity = [], []
    for i in range(len(class_names)):
        TP, FP, TN, FN = 0, 0, 0, 0
        for j in range(len(y_true)):
            if y_true[j] == i and y_pred[j] == i:
                TP += 1
            elif y_true[j] != i and y_pred[j] == i:
                FP += 1
            elif y_true[j] != i and y_pred[j] != i:
                TN += 1
            elif y_true[j] == i and y_pred[j] != i:
                FN += 1
        sensitivity.append(TP / (TP + FN))
        print(i, 'sensitivity分子分母：', TP, (TP + FN))
        specificity.append(TN / (TN + FP))
        print(i, 'specificity分子分母：', TN, (TN + FP))
    df_ = pd.DataFrame()
    df_['class'] = class_names
    df_['sensitivity'] = sensitivity
    df_['specificity'] = specificity
    df_['support'] = [sum(np.array(y_true) == 0), sum(np.array(y_true) == 1), sum(np.array(y_true) == 2)]
    print(df_)

    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_sensi0, seed=42)
    print('良性敏感性：', score, '【', ci_lower, ci_upper, '】')
    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_speci0, seed=42)
    print('良性特异性：', score, '【', ci_lower, ci_upper, '】')
    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_sensi1, seed=42)
    print('交界性敏感性：', score, '【', ci_lower, ci_upper, '】')
    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_speci1, seed=42)
    print('交界性特异性：', score, '【', ci_lower, ci_upper, '】')
    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_sensi2, seed=42)
    print('恶性敏感性：', score, '【', ci_lower, ci_upper, '】')
    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=get_speci2, seed=42)
    print('恶性特异性：', score, '【', ci_lower, ci_upper, '】')

    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=sklearn.metrics.accuracy_score, seed=42)
    print('准确性：', score, '【', ci_lower, ci_upper, '】')

    # print("模型与真实情况的kappa一致性:{:.4f},acc:{:.4f}"
    #       .format(cohen_kappa_score(y_true, y_pred, weights='quadratic'), sum(y_true == y_pred) / len(y_true)))
    def kappa_(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=kappa_, seed=42)
    print('kappa：', score, '【', ci_lower, ci_upper, '】')

    # print('F1:',f1_score(y_true, y_pred, average='weighted'))
    def f1_(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    score, ci_lower, ci_upper, scores = score_ci(y_true, y_pred, score_fun=f1_, seed=42)
    print('F1：', score, '【', ci_lower, ci_upper, '】')

    return 0