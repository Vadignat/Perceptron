import numpy as np

class MLP:

    __a = 1
    __b = 1

    def __init__(self,
                 tri,
                 tro,
                 tsi,
                 tso,
                 neuronNum = (3,)):
        # Количество слоёв
        self.__layers = len(neuronNum) + 2
        nN = [len(tri[0]), len(tro[0])]
        self.__nN = np.insert(nN, 1, neuronNum)
        self.__inp = tri
        self.__out = tro
        self.__tsi = tsi
        self.__tso = tso
        self.__w = [
            np.random.rand(
                self.__nN[i] + 1,
                self.__nN[i + 1]
                + (0 if i == self.__layers - 2 else 1)
            )
            for i in range(self.__layers - 1)
        ]

    def nonLinAct(self, x):
        return self.__a * np.tanh(self.__b * x)

    def nonLinActDer(self, x):
        return np.array(self.__b / self.__a * \
               (self.__a - self.nonLinAct(x)) * \
               (self.__a + self.nonLinAct(x)))

    def linAct(self, x):
        return np.array(x)

    def linActDer(self, x):
        return np.array(1)

    def learn(self,
              eta = 0.005 ,
              epoches = 1000,
              epsilon = 0.0001):
        e_full_tr = []
        e_full_ts = []
        # Индуцированное Локальное Поле
        v = np.array([None for i in range(self.__layers)])
        # Слои
        l = np.array([None for i in range(self.__layers)])
        # Ошибки
        l_err = np.array([None for i in range(1, self.__layers)])
        # deltas
        l_delta = np.array([None for i in range(1, self.__layers)])

        inp = self.__inp
        out = self.__out

        #счётчик эпох
        k = 0
        while k < 2 or\
              k < epoches and (abs(e_full_ts[k-1] - e_full_ts[k-2]) > epsilon) or\
              k < epoches and (e_full_ts[k-1] > epsilon):
            k += 1
            for i in range(len(inp)):
                #прямой ход
                l[0] = np.array([np.insert(inp[i], 0, 1)])
                for j in range(1, self.__layers - 1):
                    v[j] = np.dot(l[j - 1], self.__w[j - 1])
                    l[j] = self.nonLinAct(v[j])
                v[self.__layers - 1] = np.dot(l[self.__layers - 2], self.__w[self.__layers - 2])
                l[self.__layers - 1] = self.linAct(v[self.__layers - 1])

                #обратный ход
                l_err[self.__layers - 2] = out[i] - l[self.__layers - 1]
                l_delta[self.__layers - 2] = l_err[self.__layers - 2] * (self.linActDer(v[self.__layers - 1]))
                for j in range(self.__layers - 2, 0, - 1):
                    l_err[j - 1] = np.dot(l_delta[j], self.__w[j].T)
                    l_delta[j - 1] = l_err[j - 1] * (self.nonLinActDer(v[j]))
                deltaW = [eta * np.dot(l_delta[j].T, l[j]) for j in range(self.__layers - 1)]
                for j in range(self.__layers - 1):
                    self.__w[j] += deltaW[j].T
            outts = self.calc(self.__tsi)
            r_outts = np.array([outts]).T
            err_n = 0.5 * np.sum((self.__tso - r_outts)**2) / len(r_outts) ###
            e_full_ts.append(err_n)
            outtr = self.calc(self.__inp)
            r_outtr = np.array([outtr]).T
            err_n_tr = 0.5 * np.sum((self.__out - r_outtr)**2) / len(r_outtr)
            e_full_tr.append(err_n_tr)
            print("Эпоха", k, "Train error = ", err_n_tr, "Test_error =", err_n)
        return e_full_tr, e_full_ts

    def calc(self, inps):
        outs = np.array([])
        for i in range(len(inps)):
            inp = np.array([np.insert(inps[i], 0, 1)])
            for lr in range(self.__layers - 2):
                inp = self.nonLinAct(np.dot(inp, self.__w[lr]))
            outs = np.append(outs, self.linAct(np.dot(inp, self.__w[self.__layers - 2])))
        return outs



