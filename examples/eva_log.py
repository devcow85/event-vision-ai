from eva import utils, model
import torch

if __name__ == "__main__":
    utils.set_seed(7)
    snn_model = model.NCARSNet(16, 1, n_steps = 5)
    out = snn_model(torch.rand((1,5,1,64,64)))
    snn_model.weight_clipper()
    
    snn_model.register_hook()
    out = snn_model(torch.rand((1,5,1,64,64)))
    
    snn_model.save_trace_dict('examples/testvector_v0.pickle')
    
    snn_model = model.DVSGestureNet(5, 1, n_steps = 5)
    out = snn_model(torch.rand((1,5,2,128,128)))
    snn_model.weight_clipper()
    
    snn_model.register_hook()
    out = snn_model(torch.rand((1,5,2,128,128)))
    
    snn_model.save_trace_dict('examples/testvector_v1.pickle')
    
    snn_model = model.NMNISTNet(5,1, n_steps=7, weight='examples/test_nmnist.pt')
    # snn_model.save_model('examples/test_nmnist.pt')
    
    