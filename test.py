import numpy as np
import pickle
import os
# https://raw.githubusercontent.com/alibaba/FederatedScope/master/scripts/distributed_scripts/gen_data.py

def generate_data(client_num=3,
                  instance_num=1000,
                  feature_num=5,
                  save_data=True):
    
    val_x = np.random.normal(loc=0.0,
                             scale=1.0,
                             size=(instance_num, feature_num))
    
    print(val_x.shape)
    weights = np.random.normal(loc=0.0, scale=1.0, size=feature_num)
    bias = np.random.normal(loc=0.0, scale=1.0)
    
     # test data
    test_x = np.random.normal(loc=0.0,
                              scale=1.0,
                              size=(instance_num, feature_num))
    
    
    test_y = np.sum(test_x * weights, axis=-1) + bias
    test_y = np.expand_dims(test_y, -1)
    test_data = {'x': test_x, 'y': test_y}
    print(test_x.shape, test_y.shape)
generate_data()