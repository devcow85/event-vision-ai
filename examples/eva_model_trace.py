# This example code for verifying register_hook method (generate HW test vector)
import os
import pickle

import torch
from eva import utils, model

if __name__ == "__main__":
    utils.set_seed(7)
    snn_model = model.NCARSNet(16, 1, n_steps = 5)
    out = snn_model(torch.rand((1,5,1,64,64)))
    
    snn_model.register_hook()
    out = snn_model(torch.rand((1,5,1,64,64)))
    
    trace_file_path = 'examples/testvector_gen4_m2_81p.pickle'
    snn_model.save_trace_dict(trace_file_path)
    
    with open(trace_file_path, 'rb') as file:
        trace_dict = pickle.load(file)
        
    top_level_key = trace_dict.keys()
    print("top level keys", top_level_key)
    print("under level key", trace_dict[list(top_level_key)[0]].keys())
    
    os.remove(trace_file_path)
    