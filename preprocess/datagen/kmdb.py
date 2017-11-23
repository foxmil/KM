import os
import lmdb
import json
import numpy as np


class kmdbWriter:
    def __init__(self, path, map_size=1.074e9, commit_freq=8):
        self.path = path
        self.env = lmdb.open(path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.cur_key = 0
        self.commit_freq=10
        
    
    def __del__(self):
        self.close()
        
        
    def write(self, x, y):
        # Check if x is a list. 
        # If it isn't wrap it around a list to make it easier to deal with
        if isinstance(x, list):
            data_x = []
            for elem in x:
                data_x.append(elem.tolist())
            print('islist')
        elif isinstance(x, np.ndarray):
            data_x = [x.tolist()]
        else:
            raise NotImplemented
        
        # Do the same thing for y
        if isinstance(y, list):
            data_y = []
            for elem in y:
                data_x.append(elem.tolist())
            print('islist')
        elif isinstance(y, np.ndarray):
            data_y = [y.tolist()]
        else:
            raise NotImplemented
        
        data = {'x': data_x, 'y': data_y}
        self.txn.put(str(self.cur_key).encode('ascii'), json.dumps(data).encode('ascii')) 
        self.cur_key += 1
        
        if self.cur_key % self.commit_freq:
            self.commit()
        
        
    def commit(self):
        self.txn.commit()
        self.txn = self.env.begin(write=True)
    
    
    def close(self):
        fname = os.path.join(self.path, '__size__')
        F = open(fname, 'w')
        F.write(str(self.cur_key))
        F.close()
        
        try:
            self.txn.commit()
        except:
            pass
        self.env.close()
    
    
class kmdbReader:
    def __init__(self, path):
        self.path = path        
        self.env = lmdb.open(path, readonly=True)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
    
        # Open the size file to determine the number of different datapoints in the lmdb.
        fname = os.path.join(self.path, '__size__')
        F = open(fname, 'r')
        self.key_len = int(F.read())
        F.close()
        
    
    def __del__(self):
        self.env.close()
        
    
    # Creates an iterator that reads the LMDB in order
    def generate(self):
        while True:
            # Check to ensure that this can safely generate keys (# of elements > 0)
            if self.key_len <= 0:
                raise IOError
                
            for key, value in self.cursor:
                data = json.loads(value)
                
                data_x = []
                for dat_x in data['x']:
                    data_x.append(np.array(dat_x))
                    
                data_y = []
                for dat_y in data['y']:
                    data_y.append(np.array(dat_y))
                
                yield data_x, data_y
                
                
    # Creates an iterator that reads through all the elements in a random order before starting again
    def rand_generate(self):
        while True:
            # Check to ensure that this can safely generate keys (# of elements > 0)
            if self.key_len <= 0:
                raise IOError
            
            keys = np.array(range(self.key_len))
            np.random.shuffle(keys)
            
            for key in keys:
                data = json.loads(self.txn.get(str(key).encode('ascii')))
                
                data_x = []
                for dat_x in data['x']:
                    data_x.append(np.array(dat_x))
                    
                data_y = []
                for dat_y in data['y']:
                    data_y.append(np.array(dat_y))
                
                yield data_x, data_y    
            
