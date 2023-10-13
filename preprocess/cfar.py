import numpy as np

def cfar(signal):
    N = len(signal)
    signal_cfar = np.zeros(N)
    Pfa = 1e-10
    n_train = 10  # 一侧的参考单元长度
    n_guard = 5  # 一侧的保护单元的长度

    for i in range(N):
        if n_train + n_guard < i < N - n_train - n_guard:
            a = 2 * n_train * (Pfa ** (-1 / (2 * n_train)) - 1)  # 门限因子
            nsum = np.sum(np.abs(signal[i - n_guard - n_train:i - n_guard]) ** 2) + np.sum(
                np.abs(signal[i + n_guard + 1:i + n_train + n_guard + 1]) ** 2)
            nsum = nsum / (2 * n_train)
            nsum = np.sqrt(a * nsum)
        elif i < n_train + n_guard + 1:  # 测量单元左侧参考单元内数据不足n_train个
            nsum = np.sum(np.abs(signal[i + n_guard + 1:i + n_train + n_guard + 1]) ** 2)
            n_tt = 0
            a = n_train * (Pfa ** (-1 / n_train) - 1)  # 门限因子
            if i > n_guard + 1:  # 进入if表示测试单元左侧数据个数大于n_guard个，小于n_guard+n_train个
                nsum = nsum + np.sum(np.abs(signal[0:i - n_guard]))
                n_tt = i - n_guard - 1
                a = (n_train + n_tt) * (Pfa ** (-1 / (n_train + n_tt)) - 1)
            nsum = nsum / (n_train + n_tt)
            nsum = np.sqrt(a * nsum)
        elif i > N - n_train - n_guard - 1:  # 测量单元右侧参考单元内数据不足n_train个
            nsum = np.sum(np.abs(signal[i - n_guard - n_train:i - n_guard]) ** 2)
            n_tt = 0
            a = n_train * (Pfa ** (-1 / n_train) - 1) # 门限因子
            if i + n_guard < N+1:
                nsum = nsum + sum(abs(signal[i + n_guard : N]))
                n_tt = N - i - n_guard
                a = (n_train + n_tt) * (Pfa ** (-1/(n_train + n_tt)) - 1)
            nsum = nsum/n_train + n_tt
            nsum = np.sqrt(a * nsum)
        signal_cfar[i] = nsum
    return signal_cfar
