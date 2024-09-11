from eva.utils import train, validation

def trainer(model, optimizer, loss, dataloader, max_epochs, device):
    train_loader, val_loader = dataloader
    
    best_acc = 0
    # train history
    # best summary
    
    for epoch in range(max_epochs):
        train(model, epoch, train_loader, optimizer, loss, device)
        validation(model, val_loader, device)
        # add model save routine 
    
    # report generator
    # 1.learning curves
    # 2.final confusion matrix
    # 3.f1 score recall precision detailed
    # 4.val_data trace
    # 5.first batch result picture
    # 6.model summary
    # 7.best model pred-gt pair
    
    return


def inference(model, data, transform, device):
    preprocess_input = transform(data)
    model.eval()
    with torch.no_grad():
        out = model(preprocess_input)
        
    # output generation sum spike
    # top_k output?
    # max output index gather and return
    
    return 
