import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

class ScatterMatrix(object):
    def __init__(self):
        pass

    def draw(self, dfNp, dfHDDp):
        # concatenate two pandas DataFrames
        feat = pd.concat([dfNp, dfHDDp], axis=1)
        # get the column names of the concatenated DataFrame
        cols = feat.columns
        # scale data to prepare for regression model 

        scaler = preprocessing.MaxAbsScaler() 
        feat = scaler.fit_transform(feat)
        # define a new DataFrame with the scaled data
        dfScaled = pd.DataFrame(feat, columns=cols)

        plt.style.use('ggplot')
        ff = pd.tools.plotting.scatter_matrix(dfScaled, diagonal='hist', figsize=(12,12))
        ff.savefig('./ScatterMatrix.png', dpi=150, format='png')

