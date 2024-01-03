import numpy as np
import torch
from torchsummary import summary




from net import CAE, weight_init
from utils import load_data,build_model,train_one_epoch,eval_one_epoch,clustering,Logging
if __name__ == '__main__':

    all_dataloader,train_dataloader, test_dataloader = load_data()

    model,optimizer = build_model()

    Log = Logging(name='PBMC.txt')
    best_model = 0
    for e in range(200):
        train_loss = train_one_epoch(model, optimizer, train_dataloader)
        test_loss = eval_one_epoch(model,test_dataloader)
        print(f'Epoch: {e+1} train: {train_loss:.4f} test: {test_loss:.4f}')
        nmi,ari,acc = clustering(model,all_dataloader)
        print('-'*50)
        Log.write_(e+1,train_loss,test_loss,ari,nmi,acc)
        if acc > best_model:
            best_model = acc
            torch.save(model.state_dict(),'checkpoint/best.pth')