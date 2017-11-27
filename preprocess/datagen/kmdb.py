import os
import lmdb
import json
import numpy as np


class kmdbWriter:
    def __init__(self, path, batch_size=32, map_size=1.074e9):
        self.path = path
        self.env = lmdb.open(path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.cur_key = 0
        self.count = 0
        self.batch_size = batch_size
        self.xbatch = []
        self.ybatch = []
        
    
    def __del__(self):
        self.close()
     
        
    # Reset the holding of the current batch
    def __reset_batch(self):
        self.count = 0
        self.xbatch = []
        self.ybatch = []
        
        
    # Adds a point to the lmdb.
    def write_point(self, xpoint, ypoint):
        # Check if a batch has already been started as this changes if a batch needs to be initiated.
        if self.count <= 0:
            # If a batch is new then it needs to be initiated
            for x in xpoint:
                self.xbatch.append(np.empty((self.batch_size,) + x.shape))
            for y in ypoint:
                self.ybatch.append(np.empty((self.batch_size,) + y.shape))
        
        # Add the point the current batch
        for i in range(0, len(self.xbatch)):
            self.xbatch[i][self.count, ...] = xpoint[i]
        for i in range(0, len(self.ybatch)):
            self.ybatch[i][self.count, ...] = ypoint[i]
        self.count += 1
        
        # If the batch is full commit the batch to the lmdb
        if self.count >= self.batch_size:
            self.write(self.xbatch, self.ybatch)
            self.commit()
            self.__reset_batch() # Reset the batch variables for the next batch
            
        
        
    def write(self, x, y):
        # Check if x is a list. 
        # If it isn't wrap it around a list to make it easier to deal with
        if isinstance(x, list):
            data_x = []
            for elem in x:
                data_x.append(elem.tolist())
        elif isinstance(x, np.ndarray):
            data_x = [x.tolist()]
        else:
            raise NotImplemented
        
        # Do the same thing for y
        if isinstance(y, list):
            data_y = []
            for elem in y:
                data_y.append(elem.tolist())
        elif isinstance(y, np.ndarray):
            data_y = [y.tolist()]
        else:
            raise NotImplemented
        
        data = {'x': data_x, 'y': data_y}
        self.txn.put(str(self.cur_key).encode('ascii'), json.dumps(data).encode('ascii')) 
        self.cur_key += 1
        
        
        
    def commit(self):
        self.txn.commit()
        self.txn = self.env.begin(write=True)
    
    
    def close(self):
        fname = os.path.join(self.path, '__size__')
        F = open(fname, 'w')
        F.write(str(self.cur_key))
        F.close()
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
               
        
#    # Creates an iterator that reads in batches. Currently      
#    def generate(self, batch_size=100):
#        while(True):
#            gen = self.rand_generate()
#            xlist, ylist = next(gen)
#            
#            # Create batch list and np array for each input
#            xbatch = []
#            for xpoint in xlist:
#                x = np.empty((batch_size,) + xpoint.shape)
#                
#                # Fill in the first values
#                x[0, ...] = xpoint
#                
#                # Add the batch for this input to the array.
#                xbatch.append(x)
#                
#            # Do the same thing but for outputs
#            ybatch = []
#            for ypoint in ylist:
#                y = np.empty((batch_size,) + ypoint.shape)
#                
#                # Fill in the first values
#                y[0, ...] = ypoint
#                
#                ybatch.append(y)
#                
#            # Fill in the rest of the batch values
#            for i in range(1, batch_size):
#                xlist, ylist = next(gen)
#                
#                # Fill in x values
#                for j in range(0, len(xbatch)):
#                    xbatch[j][i, ...] = xlist[j]
#    
#                # Fill in y valuess
#                for j in range(0, len(ybatch)):
#                    ybatch[j][i, ...] = ylist[j]
#    
#            yield xbatch, ybatch


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
            
