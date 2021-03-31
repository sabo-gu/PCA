import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import f, norm

class PcaDetection():
    def __init__(self, n_components, alpha, train_data, plot_figure=False):
        self.n_components = n_components
        self.alpha = alpha
        self.plot_figure = plot_figure
        self.train_data_mean = np.mean(train_data, axis=0)
        self.train_data_std = np.std(train_data, axis=0)
        self.train_data=(train_data - self.train_data_mean) / self.train_data_std
        self.n_features = min(train_data.shape)
        self.n_samples = max(train_data.shape)



    def fit(self):
         self.model = PCA(n_components=self.n_features)
         self.model.fit(self.train_data)
         self.eig_vector = self.model.components_[:self.n_components, :]
         self.eig_value_all = self.model.explained_variance_
         self.eig_value = self.model.explained_variance_[:self.n_components]

         T2_lim = self.n_components * (self.n_samples - 1) / (self.n_samples - self.n_components) * f.ppf(1 - self.alpha,
                                                                                                         self.n_components,
                                                                                                         self.n_samples - self.n_components)
         theta = [0, 0, 0]
         for i in range(3):
             theta[i] = sum([self.eig_value_all[j] ** i for j in range(self.n_components, self.n_features)])
         h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
         SPE_lim = theta[0] * (
                1 + norm.ppf(1 - self.alpha) * np.sqrt(2 * theta[1] * h0 ** 2) / theta[0] + theta[1] * h0 * (
                h0 - 1) / theta[0] ** 2) ** (1 / h0)
         self.T2_lim=T2_lim
         self.SPE_lim=SPE_lim

         return T2_lim,SPE_lim

    def predict(self, test_data):
        test_data = (test_data - self.train_data_mean) / self.train_data_std
        self.test_samples = test_data.shape[0]
        self.new_x = self.model.transform(test_data)[:, :self.n_components]
        self.test_data = test_data
        if self.test_data>1:
            T2 = np.diagonal(
                self.new_x @ np.linalg.inv(np.eye(self.n_components, self.n_components) * self.eig_value) @ self.new_x.T,
                offset=0)
            SPE = np.diagonal(
                self.test_data @ (np.eye(self.n_features,
                                         self.n_features) - self.eig_vector.T @ self.eig_vector) @ self.test_data.T,
                offset=0)
        else:
            T2 =self.new_x @ np.linalg.inv(
                    np.eye(self.n_components, self.n_components) * self.eig_value) @ self.new_x.T
            SPE =self.test_data @ (np.eye(self.n_features,
                                         self.n_features) - self.eig_vector.T @ self.eig_vector) @ self.test_data.T





        if self.plot_figure:
            plt.figure()
            plt.plot(self.T2_lim * np.ones((self.n_samples, 1)), color='r',label='threshold')
            plt.plot(T2, 'b',label='T2')
            plt.xlabel('samples')
            plt.ylabel('T2_stactics')
            plt.title('')
            plt.figure()
            plt.plot(self.SPE_lim * np.ones((self.n_samples, 1)), color='r')
            plt.plot(SPE, 'b')
            plt.xlabel('samples')
            plt.ylabel('SPE_stactics')
            plt.title('')

        return T2,SPE
