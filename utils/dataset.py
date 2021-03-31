import numpy as np
import pandas as pd

a=np.loadtxt('data/d01_te.dat')

class TE_dataset():
    def __init__(self,path,test_path):
        self.path=path
        self.test_path=test_path

    def load_data(self):
        train_data=np.loadtxt(self.path)
        test_data=np.loadtxt(self.test_path)
        if train_data.shape==(960,52):
            pass
        else:
            train_data=train_data.T
        if test_data.shape==(960,52):
            pass
        else:
            test_data=test_data.T
        return train_data,test_data



