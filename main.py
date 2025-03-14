from models.GCNmodel import *
from utils import *
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torchsummary import summary
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler


if __name__=="__main__":
    init_seed = 112
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    is_train=False
    task_all=['Mag','lattice_constant','E0','Ef','stability']
    #task=[['Mag','lattice_constant']]
    #task=['E0',['Ef','stability']]
    task_id=0
    task=[task_all[task_id]]
    #task=[['lattice_constant','E0'],['Ef','stability']]

    task_flatten=[]
    for i in task:
        if isinstance(i,list):
            task_flatten+=i
        else:
            task_flatten.append(i)
    task=[i if isinstance(i,list) else [i] for i in task]
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    batch_size_train = 128
    batch_size_test = 256
    
    if len(task)==1:
        if len(task_flatten)==1:
            if task_flatten[0]=='Mag':
                datasets=[elementtables_structure_new(task=task_flatten[0])]
                model = attention_CNN(block_num=[2,3,3],fc=[512,128,1],end_with_activation=True,is_vol=True).to(device)
            elif task_flatten[0]=="lattice_constant":
                datasets=[elementtables_structure_new(task=task_flatten[0])]
                model = attention_CNN(block_num=[2,3,3],fc=[512,128,1],end_with_activation=True).to(device)
            elif task_flatten[0]=='E0':
                datasets=[elementtables_structure_new(task=task_flatten[0])]
                model = attention_CNN(block_num=[2,3,3],fc=[512,128,1],end_with_activation=False).to(device)
            elif task_flatten[0]=='Ef':
                datasets=[elementtables_structure_full_heusler()]
                model = attention_CNN(block_num=[2,3,3],fc=[512,128,1],end_with_activation=False).to(device)
            elif task_flatten[0]=='stability':
                datasets=[elementtables_structure_full_heusler()]
                model = attention_CNN(block_num=[2,3,3],fc=[256,128,1],end_with_activation=True).to(device)
            else:
                raise ValueError('task not found')
        elif task_flatten==['Mag','lattice_constant']:
            datasets=[elementtables_structure_new(task='Mag')]
            model = attention_CNN_bitask().to(device)
        else:
            raise ValueError('task maybe in the wrong order or the task should not be use the same dataset')
    elif len(task)==2:
        if task_flatten==['E0','Ef','stability']:
            datasets=[elementtables_structure_new(task='E0'),elementtables_structure_full_heusler()]
            model = attention_CNN_tritask().to(device)
        elif task_flatten==['lattice_constant','E0','Ef','stability']:
            datasets=[elementtables_structure_new(task='E0'),elementtables_structure_full_heusler()]
            model = attention_CNN_quadtask().to(device)
    #dataset=elementtables_structure_new(task='Mag')
    #dataset=elementtables_structure_full_heusler()
    
    
    ratio=[0.8,0.1,0.1]
    
    #if is_MTL:
    #    train_dataloader=torch.utils.data.DataLoader(TensorDataset(*mag_train), batch_size=batch_size_train,shuffle=True)
    #    test_dataloader=torch.utils.data.DataLoader(TensorDataset(*mag_test), batch_size=batch_size_test,shuffle=False)
    #    model = MTL_CNN().to(device)
    #elif is_cnn:
    #    train_dataloader=torch.utils.data.DataLoader(TensorDataset(*mag_train), batch_size=batch_size_train,shuffle=True)
    #    #val_dataloader=torch.utils.data.DataLoader(TensorDataset(*mag_val), batch_size=batch_size_test,shuffle=False)
    #    test_dataloader=torch.utils.data.DataLoader(TensorDataset(*mag_test), batch_size=batch_size_test,shuffle=False)
    #    model = attention_CNN().to(device)
    #else:
    #    train_dataloader=DataLoader(dataset[train_mask], batch_size=batch_size_train,shuffle=True)
    #    test_dataloader=DataLoader(dataset[test_mask], batch_size=batch_size_test,shuffle=False)
    #    model = HeuslerAlloy().to(device)
        
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()

    default_lr=0.005
    specific_lr=0.01
    params_default_lr=[]
    params_specific_lr=[]
    for name, param in model.named_parameters():
        #if 'fc_block' in name or 'embedding' in name:
        if 'fc_block' in name:
            params_specific_lr.append(param)
        else:
            params_default_lr.append(param)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizer= torch.optim.Adam([
        {'params':params_specific_lr,'lr':specific_lr},
        {'params':params_default_lr,'lr':default_lr}])
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    #random_seed=1454880
    np.random.seed(init_seed)

    train_class=Train_Test_Plot(model,datasets,task,loss_fn,optimizer,split_ratio=ratio,\
                                batch_size_train=batch_size_train,batch_size_test=batch_size_test)
    #epoch_max=1000 if is_train else 0
    #patience=50
    #epoch_num=0
    #loss_list=[]
    #contrast_loss=np.inf
    #min_loss=np.inf
    #patience_num=0
    front=time.time()
    train_class.trainer(epoch_max=1000 if is_train else 0,patience=50)
    #while epoch_num<epoch_max and is_train:
    #    epoch_num+=1
    #    if is_MTL:
    #        train_loss=train_MTL(train_dataloader, model, loss_fn, optimizer)
    #        test_loss=test_MTL(test_dataloader, model, loss_fn)
    #    elif is_cnn:
    #        train_loss=train(train_dataloader, model, loss_fn, optimizer)
    #        #val_loss=test_cnn(val_dataloader, model, loss_fn)
    #        test_loss=test(test_dataloader, model, loss_fn)
    #    else:
    #        train_loss=train(train_dataloader, model, loss_fn, optimizer)
    #        test_loss=test(test_dataloader, model, loss_fn)
#
    #    loss_list.append([train_loss,test_loss])
    #    #loss_list.append([train_loss,val_loss,test_loss])
    #    if test_loss<contrast_loss if not is_MTL else sum(test_loss)<sum(contrast_loss):
    #    #if val_loss<contrast_loss:
    #        contrast_loss=test_loss
    #        #contrast_loss=val_loss
    #        patience_num=0
    #        torch.save(model.state_dict(), "model.pth")
    #        print("Saved PyTorch Model State to model.pth")
    #    else:
    #        patience_num+=1
    #        if patience_num>=patience:
    #            print('early stop')
    #            break
    #    if not is_MTL:
    #        print(f"Epoch {epoch_num}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
    #    else:
    #        min_loss=[min_loss[i] if min_loss[i]<test_loss[i] else test_loss[i] for i in range(2)]
    #        print("Epoch:{0}, train loss:{1}, test loss:{2}, contrast loss:{3}, min loss:{4}"
    #                .format(epoch_num,[round(i,4) for i in train_loss],[round(i,4) for i in test_loss],[round(i,4) for i in contrast_loss],[round(i,4) for i in min_loss]))
    
    rear=time.time()
    print('\ntime:')
    print(rear-front)
    
    model.load_state_dict(torch.load("model.pth"))
    #train_class.plot_result()
    #train_class.plot_attention()
    #train_class.plot_pca()
    #train_class.ROC()

    
    summary(model, input_size=(8,10,10))
    
    #if not is_train:
    #    model.eval()
    #    if is_MTL:
    #        test_loss=test_MTL(test_dataloader, model, loss_fn)
    #    elif is_cnn:
    #        test_loss=test(test_dataloader, model, loss_fn)
    #    else:
    #        test_loss=test(test_dataloader, model, loss_fn)
    #    print(f"test loss: {test_loss}")
    #
    #    
    #if is_MTL:
    #    plot_mtl_result(model, train_dataloader, test_dataloader,num=100)
    #    summary(model, input_size=[(1,10,10),(1,10,10)])
    #elif is_cnn:
    #    plot_result(model, train_dataloader, test_dataloader,num=100)
    #    np.save("loss.npy",np.array(loss_list))
    #    summary(model, input_size=(8,10,10))
    #else:
    #    plot_result(model, train_dataloader, test_dataloader,ratio=0.04)
    #    #summary(model, input_size=(dataset.num_features,))


