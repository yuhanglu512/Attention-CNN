import os
import sys
sys.path.append(os.path.abspath('..'))
import torch.utils
import itertools
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.GCNmodel import *
from utils import *
import time

class exchange_datasets(torch.utils.data.Dataset):
    def __init__(self,elements,device='cuda'):
        super().__init__()
        save_dataname="exchange_dataset_init.npz"
        self.device=device
        self.data=[]
        self.spacegroup=[]
        if not os.path.isfile(save_dataname):
            data,spacegroup=self.nonequivalent_permutations(elements,3)
            self.data.append(data)
            self.spacegroup.append(spacegroup)
            data,spacegroup=self.nonequivalent_permutations(elements,4)
            self.data.append(data)
            self.spacegroup.append(spacegroup)
            data,spacegroup=self.nonequivalent_permutations(elements,4,need_void=True)
            self.data.append(data)
            self.spacegroup.append(spacegroup)
            self.data=torch.cat(self.data,dim=0).to(dtype=torch.float32,device=self.device)
            self.spacegroup=torch.tensor(np.concatenate(self.spacegroup),dtype=torch.int32,device=self.device).reshape(-1)
            np.savez(save_dataname,data=self.data.cpu().numpy(),spacegroup=self.spacegroup.cpu().numpy())
        else:
            data=np.load(save_dataname)
            self.data=torch.tensor(data['data'],dtype=torch.float32,device=self.device)
            self.spacegroup=torch.tensor(data['spacegroup'],dtype=torch.int32,device=self.device).reshape(-1)

    def nonequivalent_permutations(self,elements, n, need_void=False):
        index_quad=[[0,1,2,3],[0,1,3,2],[0,2,1,3]]
        index_tri=[[0,1,2,2],[2,0,1,1],[1,2,0,0],[0,2,1,2],[0,1,2,1],[1,0,2,0]]

        if n==4 and not need_void:
            index=index_quad
            combination_list=list(itertools.combinations(elements,n))
        elif n==4 and need_void:
            index=index_quad
            combination_list=list(itertools.combinations(elements,n-1))
        elif n==3:
            index=index_tri
            combination_list=list(itertools.combinations(elements,n))

        data=[]
        spacegroup=[]
        for i,combination in enumerate(combination_list):
            args=[np.argwhere(element==ChemicalSymbols[:100])[0][0] for element in combination]
            if need_void:
                args=args+[0]
            for j,permutation in enumerate(index):
                elementtable_permute=torch.zeros((7,100))
                elementtable_permute[list(range(3,7)),torch.tensor(args)[torch.tensor(permutation)]]=1
                elementtable_permute[:,0]=0
                elementtable=torch.count_nonzero(elementtable_permute,dim=0)/torch.sum(torch.count_nonzero(elementtable_permute,dim=0))
                data.append(torch.concatenate((elementtable.reshape(1,100),elementtable_permute),dim=0).reshape(8,10,10))
                if n==4:
                    spacegroup.append(216)
                else:
                    if j<3:
                        spacegroup.append(216)
                    else:
                        spacegroup.append(225)
        data=torch.stack(data,dim=0)
        data=data.to(self.device)
        return data,spacegroup
    
    def get_lattice(self,data):
        a=data[0]
        E0=data[1]
        Ef=data[2]
        stability=data[3]
        self.a, self.E0, self.Ef, self.stability = a, E0, Ef, stability

    def return_all_but_mag(self):
        return self.a,self.E0,self.Ef,self.stability
    
    def return_all(self):
        return self.data,self.a,self.E0,self.Ef,self.stability,self.mag

    def return_vol(self):
        return (self.a)**3/(2**0.5)

    def input4mag(self):
        return self.data,self.spacegroup,self.return_vol()

    def get_mag(self,data):
        self.mag=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.spacegroup[idx]


if __name__=='__main__':
    os.chdir('../')
    is_train=False

    task=[['lattice_constant','E0'],['Ef','stability']]
    task_flatten=[]
    for i in task:
        if isinstance(i,list):
            task_flatten+=i
        else:
            task_flatten.append(i)
    task=[i if isinstance(i,list) else [i] for i in task]

    device = 'cuda'

    datasets=[elementtables_structure_new(task='Mag',device=device),elementtables_structure_full_heusler(device=device)]

    ratio=[0.7,0.15,0.15]

    RANDOM_SEED = 119
    np.random.seed(RANDOM_SEED)

    index=torch.ones((100),dtype=torch.int32,device=device)
    for i in range(len(datasets)):
        if isinstance(ratio,float) or len(ratio)==2:
            train_mask,test_mask=random_split(len(datasets[i]),ratio)
            train_dataset,test_dataset=datasets[i][train_mask],datasets[i][test_mask]
            index=((index) & (torch.count_nonzero((train_dataset[0][:,0,:,:]>50).reshape(-1,100),dim=0)>0))
        elif len(ratio)==3:
            train_mask,valid_mask,test_mask=random_split(len(datasets[i]),ratio)
            train_dataset,valid_dataset,test_dataset=datasets[i][train_mask],datasets[i][valid_mask],datasets[i][test_mask]
            index=((index) & (torch.count_nonzero((train_dataset[0][:,0,:,:]>50).reshape(-1,100),dim=0)>0))

    elements=ChemicalSymbols[:100][index.cpu().numpy()>0]
    exchange_dataset=exchange_datasets(elements,device=device)

    model=attention_CNN_quadtask().to(device)
    model.load_state_dict(torch.load('result/%s_model.pth'%('_'.join(task_flatten))))

    dataloader=torch.utils.data.DataLoader(exchange_dataset,batch_size=1024,shuffle=False)
    output=[]
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            output_tmp=[]
            for i in range(len(task_flatten)):
                output_tmp.append(model(data[0],data[1],task=task_flatten[i]))
            output.append(output_tmp)
    output=[torch.cat([j[i] for j in output],dim=0) for i in range(len(task_flatten))]

    exchange_dataset.get_lattice(output)
    dataloader=torch.utils.data.DataLoader(TensorDataset(*exchange_dataset.input4mag()),batch_size=1024,shuffle=False)
    model=attention_CNN(end_with_activation=True).to(device)
    model.load_state_dict(torch.load('result/Mag_model.pth'))
    output=[]
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            output.append(model(data[0],data[1],data[2],task='mag'))
    output=torch.cat(output,dim=0)
    exchange_dataset.get_mag(output)

    data,a,E0,Ef,stability,mag=exchange_dataset.return_all()

    #top=10000
    top_index=[]
    order_list_with_index=sorted(enumerate(mag.to('cpu').numpy()),key=lambda x:x[1],reverse=True)
    ordered_list=[i[1] for i in order_list_with_index]
    original_indices=[i[0] for i in order_list_with_index]

    a,E0,Ef,stability,mag=a.to('cpu').numpy(),E0.to('cpu').numpy(),Ef.to('cpu').numpy(),stability.to('cpu').numpy(),mag.to('cpu').numpy()

    #i=0
    #while(len(top_index)<top):
    #    is_same_list=torch.sum(torch.abs(datasets[1][:][0][:,0,:,:]-data[original_indices[i],0,:,:]).reshape(-1,100),dim=1)
    #    if (is_same_list>0.01).all():
    #        top_index.append(original_indices[i])
    #    i+=1

    for i in range(len(original_indices)):
        is_same_list=torch.sum(torch.abs(datasets[1][:][0][:,0,:,:]-data[original_indices[i],0,:,:]).reshape(-1,100),dim=1)
        is_same_list1=torch.sum(torch.abs(datasets[0][:][0][:,0,:,:]-data[original_indices[i],0,:,:]).reshape(-1,100),dim=1)
        if (is_same_list>0.01).all():
            if (is_same_list1>0.01).all():
                top_index.append(original_indices[i])

    data=data.to('cpu').numpy()

    top_data=[]
    for i in top_index:
        top_data_tmp=[]
        top_data_tmp.append(list(ChemicalSymbols[np.argwhere(data[i,-4:,:,:].reshape(4,100)!=0)[:,1]]))
        top_data_tmp.append(mag[i][0])
        top_data_tmp.append(a[i][0])
        top_data_tmp.append(Ef[i][0])
        top_data_tmp.append(stability[i][0])
        top_data.append(top_data_tmp)
    top_data=pd.DataFrame(top_data,columns=['elements','mag','a','Ef','stability'])

    with open('result/top_data.txt','w') as f:
        f.write(top_data.to_string())

    data=torch.tensor(data,dtype=torch.float32,device=device)

    #top=5
    #top_index4mag_stable=[]
    #top_index4mag_metastable=[]
    #i=0
    #while(len(top_index4mag_stable)<top and i<len(top_index)):
    #    is_same_list=torch.sum(torch.abs(datasets[0][:][0][:,0,:,:]-data[top_index[i],0,:,:]).reshape(-1,100),dim=1)
    #    if (is_same_list>0.01).all():
    #        if stability[top_index[i]]<0.01:
    #            top_index4mag_stable.append(top_index[i])
    #        elif stability[top_index[i]]>=0.01 and stability[top_index[i]]<0.1:
    #            top_index4mag_metastable.append(top_index[i])
    #    i+=1

    threshold=0.1
    top_index4mag_stable=[]
    top_index4mag_metastable=[]
    i=0
    for i in range(len(top_index)):
        if stability[top_index[i]]<0.01:
            top_index4mag_stable.append(top_index[i])
        elif stability[top_index[i]]>=0.01 and stability[top_index[i]]<threshold:
            top_index4mag_metastable.append(top_index[i])


    data=data.to('cpu').numpy()

    top_data4mag=[]
    for i in top_index4mag_stable:
        top_data_tmp=[]
        top_data_tmp.append(list(ChemicalSymbols[np.argwhere(data[i,-4:,:,:].reshape(4,100)!=0)[:,1]]))
        top_data_tmp.append(mag[i][0])
        top_data_tmp.append(a[i][0])
        top_data_tmp.append(E0[i][0])
        top_data4mag.append(top_data_tmp)
    top_data4mag=pd.DataFrame(top_data4mag,columns=['elements','mag','a','E0'])

    top_data4mag_metastable=[]
    for i in top_index4mag_metastable:
        top_data_tmp=[]
        top_data_tmp.append(list(ChemicalSymbols[np.argwhere(data[i,-4:,:,:].reshape(4,100)!=0)[:,1]]))
        top_data_tmp.append(mag[i][0])
        top_data_tmp.append(a[i][0])
        top_data_tmp.append(E0[i][0])
        top_data4mag_metastable.append(top_data_tmp)
    top_data4mag_metastable=pd.DataFrame(top_data4mag_metastable,columns=['elements','mag','a','E0'])

    with open('result/top_data4mag.txt','w') as f:
        f.write(top_data4mag.to_string())

    with open('result/top_data4mag_metastable.txt','w') as f:
        f.write(top_data4mag_metastable.to_string())

    for i in range(min(len(top_index4mag_stable),10)):
        with open('search/stable/%s.vasp'%(''.join(top_data4mag.iloc[i,0])),'w') as f:
            f.write('elements: %s\n'%(''.join(top_data4mag['elements'][i])))
            f.write('1.000\n')
            f.write('     %f    %f    %f\n'%(0,top_data4mag['a'][i]/(2**0.5),top_data4mag['a'][i]/(2**0.5)))
            f.write('     %f    %f    %f\n'%(top_data4mag['a'][i]/(2**0.5),0,top_data4mag['a'][i]/(2**0.5)))
            f.write('     %f    %f    %f\n'%(top_data4mag['a'][i]/(2**0.5),top_data4mag['a'][i]/(2**0.5),0))
            element_set=set(top_data4mag['elements'][i])
            for element in element_set:
                f.write('   %s'%element)
            f.write('\n')
            for element in element_set:
                f.write('   %d'%list(top_data4mag['elements'][i]).count(element))
            f.write('\nDirect\n')
            for element in element_set:
                args=np.argwhere(np.array(top_data4mag['elements'][i])==element).reshape(-1)
                for arg in args:
                    f.write('   %f    %f    %f\n'%(arg*0.25,arg*0.25,arg*0.25))

    for i in range(min(len(top_index4mag_metastable),10)):
        with open('search/meta_stable/%s.vasp'%(''.join(top_data4mag_metastable.iloc[i,0])),'w') as f:
            f.write('elements: %s\n'%(''.join(top_data4mag_metastable['elements'][i])))
            f.write('1.000\n')
            f.write('     %f    %f    %f\n'%(0,top_data4mag_metastable['a'][i]/(2**0.5),top_data4mag_metastable['a'][i]/(2**0.5)))
            f.write('     %f    %f    %f\n'%(top_data4mag_metastable['a'][i]/(2**0.5),0,top_data4mag_metastable['a'][i]/(2**0.5)))
            f.write('     %f    %f    %f\n'%(top_data4mag_metastable['a'][i]/(2**0.5),top_data4mag_metastable['a'][i]/(2**0.5),0))
            element_set=set(top_data4mag_metastable['elements'][i])
            for element in element_set:
                f.write('   %s'%element)
            f.write('\n')
            for element in element_set:
                f.write('   %d'%list(top_data4mag_metastable['elements'][i]).count(element))
            f.write('\nDirect\n')
            for element in element_set:
                args=np.argwhere(np.array(top_data4mag_metastable['elements'][i])==element).reshape(-1)
                for arg in args:
                    f.write('   %f    %f    %f\n'%(arg*0.25,arg*0.25,arg*0.25))


    