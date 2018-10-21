import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



class Encoder():
    def __init__(self):
        self._encoder = LabelEncoder()
    
    def fit(self, values):
        self._encoder.fit(["---"] + list(values))
        self.classes = self._encoder.classes_
    def transform(self, values):
        values[~np.isin( values , self._encoder.classes_)] = "---"
        return self._encoder.transform(values)
    
    
def expand_data(table,match_id_col,id_col,players_per_game=100):

    table.sort_values(match_id_col, inplace=True)

    unique_games = np.unique(table[match_id_col].values)

    ranks_expanded = []
    for i in range(len(unique_games)):
        ranks_expanded.append(np.arange(0,players_per_game))
    ranks_expanded = np.concatenate(ranks_expanded)

    expanded = pd.DataFrame({match_id_col:np.repeat(unique_games, 
                                                    players_per_game, 
                                                    0),
                             "rank":ranks_expanded})

    counts = table.groupby(match_id_col).size()
    ranks = []
    for value_i in counts.values:
        ranks.append(np.arange(value_i))
    ranks = np.concatenate(ranks)

    table["rank"] = ranks

    expanded = expanded.merge(table, on=[match_id_col, "rank"], how="left")
    
    return expanded


class Batcher():
    '''
    Batcher class. Given a list of np.arrays of same 0-dimension, returns a 
    a list of batches for these elements
    '''
    def __init__(self, data, batch_size, shuffle_on_reset=False):
        '''
        :param data: list containing np.arrays
        :param batch_size: size of each batch
        :param shuffle_on_reset: flag to shuffle data
        '''
        self.data = data
        self.batch_size = batch_size
        self.shuffle_on_reset = shuffle_on_reset
        
        if type(data) == list:
            self.data_size = data[0].shape[0]
        else:
            self.data_size = data.shape[0]
        self.n_batches = int(np.ceil(self.data_size/self.batch_size))
        self.I = np.arange(0, self.data_size, dtype=int)
        if shuffle_on_reset:
            np.random.shuffle(self.I)
        self.current = 0
        
    def shuffle(self):
        '''
        Re-shufle the data
        '''
        np.random.shuffle(self.I)
        
    def reset(self):
        '''
        Reset iteration counter
        '''
        if self.shuffle_on_reset:
            self.shuffle()
        self.current = 0
        
    def next(self):
        '''
        Get next batch
        :return: list of np.arrays
        '''
        I_select = self.I[(self.current*self.batch_size):((self.current+1)*self.batch_size)]
        batch = []
        for elem in self.data:
            batch.append(elem[I_select])
        
        if(self.current<self.n_batches-1):
            self.current = self.current+1
        else:
            self.reset()
            
        return batch



def shuffle_data(X,y):
    I = np.zeros([X.shape[0],100])
    index = np.arange(100)
    for i in range(X.shape[0]):
        new_index = index[np.random.shuffle(index)]
        X[i,:,:] = X[i,new_index,:]
        y[i,:] = y[i,new_index]
    return X,y


class Normalizers():
    def __init__(self,X):
        self._normalizers = []
        for feature_index in range(X.shape[2]):
            normalizer = Normalizer()
            normalizer.fit(X[:,:, feature_index])
            self._normalizers.append(normalizer)
    
    def normalize(self, X):
        XX = X.copy()
        for i in range(len(self._normalizers)):
            normalizer = self._normalizers[i]
            XX[:,:,i] = normalizer.transform(X[:,:,i])
        return XX