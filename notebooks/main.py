"""main module
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class WindForecaster:
    def __init__(self, df:pd.DataFrame, test_size: int, prediction_window: int,
                       forecast_delay: int, forecast_size: int, activation_function: str,
                       loss_metric: str, optimizer: str):
        
        
        self.dataset = self.__normalize(self.__reshape(df))
        self.train_set, self.test_set = self.__split(test_size)
        
        self.train_X, self.train_Y = self.__createTimeWindows(
            dataset=self.train_set['normalized'].values,
            prediction_window=prediction_window,
            forecast_delay=forecast_delay)
        
        self.test_X, self.test_Y = self.__createTimeWindows(
            dataset=self.test_set['normalized'].values,
            prediction_window=prediction_window,
            forecast_delay=forecast_delay)
        
        self.model = self.__init_model(prediction_window=prediction_window, forecast_size=forecast_size,
                                       activation_function=activation_function, loss_metric=loss_metric,
                                       optimizer=optimizer)
        
    def __reshape(self, df) -> pd.DataFrame:
        """Reshapes the provided Dataset into a single feature dataset
        """
        df = df.T.stack().to_frame()
        df.index = ['2021-01-{:02d} {:02d}:00'.format(int(i), j) for i, j in df.index]
        df = df.rename(columns={0: 'speed'})

        return df
    
    def __normalize(self, df) -> pd.DataFrame:
        """Wind Speed normalization
        """
        indexes = df.index
        values = df.values
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)
        df['normalized'] = scaled_values

        return df

    def __split(self, test_size:int):
        """Splits the dataset for training and testing
        """
        tscv = TimeSeriesSplit(n_splits=2, test_size=test_size)
        for train_index, test_index in tscv.split(self.dataset):
            train, test = self.dataset.iloc[train_index], self.dataset.iloc[test_index]

        return train, test
    
    def __createTimeWindows(self,dataset, prediction_window: int, forecast_delay: int) -> (np.array, np.array):
        """Creates time windows, which could be used for prediction or testing
        """
        X, Y = [], []
        for i in range(0, dataset.shape[0] - prediction_window - 1):
            X.append(dataset[i:(i + prediction_window)])
            Y.append(dataset[i + prediction_window + forecast_delay - 1])

        X,Y = np.array(X), np.array(Y)
        return np.reshape(X, (X.shape[0], 1, X.shape[-1])), Y
    
    def __init_model(self, prediction_window: int, forecast_size: int,
                     activation_function: str, loss_metric: str, optimizer: str):
    
        units = int((prediction_window * 0.67) + (forecast_size * 0.33))
        input_shape = (1, prediction_window)

        model = Sequential()
        tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999,
                                     epsilon=1e-07, amsgrad=False, name="adam")
        model.add(LSTM(units=units, input_shape=input_shape))
        model.add(Dense(1, activation=activation_function))

        metrics = tf.keras.metrics.RootMeanSquaredError()

        model.compile(
            loss=loss_metric,
            metrics=[metrics],
            optimizer=optimizer)

        print("Model is ready to be trained.")
        return model
    
    def train(self, batch_size:int, epochs:int, verbose: str="auto", plot: bool=True):

        training = self.model.fit(x=self.train_X, y=self.train_Y,batch_size=batch_size,epochs=epochs,verbose=verbose)

        if plot:
            hist_df = pd.DataFrame(training.history)
            hist_df = hist_df.rename(columns={'root_mean_squared_error': 'RMSE'})
            fig = px.line(hist_df['RMSE'], title="Error accros training",labels={"value": "Root Mean Squared value","index": "Epoch","variable": "Erro Metric"})
            fig.show()
            
        return training
    
    def __repr__(self):
        
        self.model.summary()
        return ""