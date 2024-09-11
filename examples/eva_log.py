from eva import utils, model
import torch

if __name__ == "__main__":
    utils.set_seed(7)
    
    
    snn_model = model.NCARSNet(16, 1, n_steps = 5)
    # print(snn_model.layers[0][0].weight[0][0])
    
    out = snn_model(torch.rand((1,5,1,64,64)))
    print(out.shape)
    snn_model.weight_clipper()
    
    snn_model.register_hook()
    out = snn_model(torch.rand((1,5,1,64,64)))
    
    snn_model.save_trace_dict('examples/testvector_v0.pickle')