import torch
import torch.nn as nn
import torch.utils
from torch_geometric.data import InMemoryDataset, download_url, Data
try:
    from .folder_names import aflow_prototype
    from .item2poscar_2 import readComponent,data2elementtable,ChemicalSymbols,PrincipalElements, get_component_vector
except ImportError:
    from folder_names import aflow_prototype
    from item2poscar_2 import readComponent,data2elementtable,ChemicalSymbols, PrincipalElements, get_component_vector
import os
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
import pandas as pd
import ast
from multiprocessing import Pool
import time
from parfor import pmap
import glob
import json

para_thread=200

def read_poscar(file_path):
    poscar = Poscar.from_file(file_path)
    return poscar

class HeuslerDataset(InMemoryDataset):
    def __init__(self, root,transform=None, pre_filter=None, pre_transform=None,cutoff_radius=6.0,device="cuda"):
        self.cutoff_radius = cutoff_radius
        self.dir_root=root
        super(HeuslerDataset,self).__init__(root,transform, pre_filter, pre_transform,cutoff_radius)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data=self.data.to(device)
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return []
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['heusler_dataset.pt']
    #用于从网上下载数据集
    def download(self):
        pass

    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # 这里用于构建data
        datalist=[]
        dataset_dir = self.dir_root.replace("heusler_dataset/","")
        for i in aflow_prototype:
            structure_dir = dataset_dir+"/"+i+"/structure/"
            df=pd.read_csv(dataset_dir+"/%s/%s.txt"%(i,i),sep='\t',header=0)
            total_len=len(os.listdir(structure_dir))
            time0=time.time()
            filepath_list=[]
            j_list=[]
            for count_index,j in enumerate(os.listdir(structure_dir)):
                if j.split(".")[-1]=="cif":
                    continue
                # 读取结构
                filepath_list.append(structure_dir+j)
                j_list.append(j)
                if len(filepath_list)==para_thread or count_index==total_len-1:
                    poscar_list=pmap(read_poscar,filepath_list)
                    for k in range(len(poscar_list)):
                        poscar=poscar_list[k]
                        j=j_list[k]
                    center_indices, points_indices, offset_vectors, distances = \
                        poscar.structure.get_neighbor_list(r=self.cutoff_radius, numerical_tol=1e-8, exclude_self=True)
                    # convert first to array, then to tensor, in order to increase the speed
                    edge_index = torch.tensor(np.array([center_indices, points_indices]),dtype=torch.int32)
                    x=list(poscar.structure.atomic_numbers)
                    pos=torch.tensor(poscar.structure.cart_coords,dtype=torch.float32)
                    y_list=ast.literal_eval(df.at[df[df["compound"]==j.split(".")[0]].index[0],"spinD"])
                    y=torch.tensor(np.array(y_list),dtype=torch.float32)
                    if torch.max(torch.abs(y))<0:
                        y=-y
                    total_spin=df.at[df[df["compound"]==j.split(".")[0]].index[0],"spin_cell"]
                    total_spin=torch.tensor(np.abs(np.array(total_spin))/len(x),dtype=torch.float32)
                    ldaul=df.at[df[df["compound"]==j.split(".")[0]].index[0],"ldau_l"]
                    ldauu=df.at[df[df["compound"]==j.split(".")[0]].index[0],"ldau_u"]
                    ldauj=df.at[df[df["compound"]==j.split(".")[0]].index[0],"ldau_j"]
                    energy_atom=df.at[df[df["compound"]==j.split(".")[0]].index[0],"energy_atom"]
                    energy_atom=torch.tensor(np.array(energy_atom),dtype=torch.float32) if isinstance(energy_atom,str) else torch.tensor(0,dtype=torch.float32)
                    species_pp_ZVAL=df.at[df[df["compound"]==j.split(".")[0]].index[0],"species_pp_ZVAL"]
                    if isinstance(species_pp_ZVAL,str):
                        species_pp_ZVAL=ast.literal_eval(species_pp_ZVAL)
                    else:
                        species_pp_ZVAL=[0]*len(x)
                    if isinstance(ldaul,str) or isinstance(ldauu,str) or isinstance(ldauj,str):
                        ldaul=ast.literal_eval(ldaul)
                        ldauu=ast.literal_eval(ldauu)
                        ldauj=ast.literal_eval(ldauj)
                    else:
                        ldaul=[-1]*len(x)
                        ldauu=[0]*len(x)
                        ldauj=[0]*len(x)
                    name_list,_=readComponent([j.split(".")[0]])
                    name_list=name_list[0]
                    atom_number=np.array([np.argwhere(ChemicalSymbols==i)[0][0] for i in name_list])+1
                    arg_list=np.array([np.argwhere(atom_number==i)[0][0] for i in x])
                    ldaul_list=np.array(ldaul)[arg_list]
                    ldauu_list=np.array(ldauu)[arg_list]
                    ldauj_list=np.array(ldauj)[arg_list]
                    species_pp_ZVAL_list=np.array(species_pp_ZVAL)[arg_list]
                    x=[x,species_pp_ZVAL_list,ldaul_list,ldauu_list,ldauj_list]
                    #x=[x,[Element(ChemicalSymbols[i-1]).X for i in x]]
                    x=torch.tensor(np.array(x),dtype=torch.float32).T
                    time3=time.time()
                    if total_spin>6:
                        continue
                    if len(y)!=len(x[:,0]):
                        continue
                    if any(torch.isnan(x).reshape(-1).tolist()) or any(torch.isnan(y).tolist()):
                        print(j)
                        continue
                    data = Data(x=x, edge_index=edge_index, pos=pos, 
                                distance=torch.tensor(distances,dtype=torch.float32), 
                                offset_vectors=torch.tensor(offset_vectors,dtype=torch.int32), 
                                y=y, total_spin=total_spin,energy_atom=energy_atom)
                    datalist.append(data)
                    filepath_list=[]
                    j_list=[]
                time4=time.time()
                if count_index%1000==0:
                    print(count_index/total_len,time4-time0,i)

        if self.pre_filter is not None:
            datalist = [data for data in datalist if self.pre_filter(data)]

        if self.pre_transform is not None:
            datalist = [self.pre_transform(data) for data in datalist]

        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])

    #def convert(self,)


class EfDataset(torch.utils.data.Dataset):
    def __init__(self,datafile='data/Ef_data.txt',
                  device="cuda", transform=None):
        super().__init__()

        with open(datafile, 'r') as f:
            file=f.readlines()
        elementtables = get_component_vector([i.split()[0] for i in file])
        values_list = np.array([float(i.rstrip().split()[1]) for i in file]).reshape(-1,1)-5

        assert len(elementtables)==len(values_list), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx]]

class elementtables(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None):
        super().__init__()
        elementtables=[]
        values_list=[]
        
        # read the superconductor data
        for i in datafile:
            df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
            sub_elementtables, sub_values_list, scale_factor=data2elementtable([list(df["compound"]),list(df["spin_cell"])],return_scale=True)
            elementtables.append(sub_elementtables)
            values_list.append(sub_values_list/scale_factor)
        
        elementtables=np.concatenate(elementtables,axis=0)
        values_list=np.absolute(np.concatenate(values_list,axis=0))
        assert len(elementtables)==len(values_list), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx]]


class elementtables_appendix(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None,appendix_list=["species_pp_ZVAL","ldau_l","ldau_u","ldau_j"]):
        super().__init__()
        elementtables=[]
        values_list=[]
        space_group=[]
        energy_atom=[]
        if not os.path.isfile("elementtables_appendix.npz"):
            # read the superconductor data
            for i in datafile:
                df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
                appendix=None if appendix_list is None else df[appendix_list]
                sub_elementtables, sub_values_list, scale_factor=data2elementtable([list(df["compound"]),list(df["spin_cell"])],return_scale=True,appendix=appendix)
                elementtables.append(sub_elementtables)
                values_list.append(sub_values_list/scale_factor)
                space_group.append(list(df["spacegroup_relax"].fillna(1)))
                energy_atom.append(list(df["energy_atom"].fillna(0)))

            elementtables=np.concatenate(elementtables,axis=0)
            values_list=np.absolute(np.concatenate(values_list,axis=0))
            space_group=np.concatenate(space_group,axis=0)
            energy_atom=np.concatenate(energy_atom,axis=0)

            np.savez("elementtables_appendix.npz",elementtables=elementtables,values_list=values_list,space_group=space_group,energy_atom=energy_atom)

        else:
            data=np.load("elementtables_appendix.npz")
            elementtables=data["elementtables"]
            values_list=data["values_list"]
            space_group=data["space_group"]
            energy_atom=data["energy_atom"]
        
        remain_index=((values_list<4) & (values_list>0.1)).reshape(-1)
        #remain_index=(values_list<4).reshape(-1)
        elementtables=elementtables[remain_index]
        values_list=values_list[remain_index]
        space_group=space_group[remain_index.squeeze()]
        elementtables=np.concatenate([elementtables[:,0:3,:,:],\
                                      np.expand_dims(elementtables[:,3,:,:]-elementtables[:,4,:,:],axis=1),\
                                        elementtables[:,5:,:,:]],axis=1)
        energy_atom=energy_atom[remain_index.squeeze()]

        assert len(elementtables)==len(values_list)==len(space_group), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.space_group=torch.tensor(space_group,dtype=torch.int32,device=device)
        self.energy_atom=torch.tensor(energy_atom,dtype=torch.float32,device=device)
        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx], self.space_group[idx],self.energy_atom[idx]]


class elementtables_appendix_discard(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None,appendix_list=["species_pp_ZVAL","ldau_l","ldau_u","ldau_j"]):
        super().__init__()
        elementtables=[]
        values_list=[]
        space_group=[]
        energy_atom=[]
        if not os.path.isfile("elementtables_appendix_discard.npz"):
            # read the heusler data
            for i in datafile:
                df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
                df.fillna({"spacegroup_relax":1,"energy_atom":0,"spin_cell":0,"spinD":0},inplace=True)
                dataset_dir="datasets/"+i+"/structure/"
                total_len=len(os.listdir(dataset_dir))
                time0=time.time()
                filepath_list=[]
                compoundname_list=[]
                appendix_dict_list=[]
                spin_list=[]
                for count_num,j in enumerate(os.listdir(dataset_dir)):
                    if j.split(".")[-1]=="cif":
                        continue
                    compoundname=j.split(".")[0]
                    item_index=df[df["compound"]==compoundname].index[0]
                    check_result=check_mag_satisfaction(compoundname,df.at[item_index,"spinD"],df.at[item_index,"spin_cell"])
                    if check_result==-1:
                        continue
                    compoundname_list.append(compoundname)
                    filepath_list.append(dataset_dir+j)
                    appendix_dict_list.append({k:df.at[item_index,k] for k in appendix_list})
                    spin_list.append(df.at[item_index,"spin_cell"])
                    if len(filepath_list)==para_thread or count_num==total_len-1:
                        cart_coords_list=pmap(extract_cartcoords,filepath_list)
                        selected_args=[i for i in range(len(cart_coords_list)) if isinstance(cart_coords_list[i],np.ndarray)]
                        cart_coords_list=np.array(cart_coords_list)
                        sub_elementtables,sub_values_list,scale_factor=data2elementtable([compoundname_list,spin_list],return_scale=True,appendix=pd.DataFrame(appendix_dict_list))
                        sub_elementtables=np.concatenate([sub_elementtables[selected_args],cart_coords_list[selected_args]],axis=1)
                        sub_values_list=(sub_values_list/scale_factor)[selected_args]
                        values_list.append(sub_values_list)
                        elementtables.append(sub_elementtables)
                        for compoundname in np.array(compoundname_list)[selected_args]:
                            space_group.append(df.at[df[df["compound"]==compoundname].index[0],"spacegroup_relax"])
                            energy_atom.append(df.at[df[df["compound"]==compoundname].index[0],"energy_atom"])
                        filepath_list=[]
                        compoundname_list=[]
                        appendix_dict_list=[]
                        spin_list=[]
                    if count_num%1000==0:
                        print(count_num/total_len,time.time()-time0,i)

            elementtables=np.concatenate(elementtables,axis=0)
            values_list=np.absolute(np.concatenate(values_list,axis=0))
            space_group=np.array(space_group)
            energy_atom=np.array(energy_atom)

            np.savez("elementtables_appendix_discard.npz",elementtables=elementtables,values_list=values_list,space_group=space_group,energy_atom=energy_atom)

        else:
            data=np.load("elementtables_appendix_discard.npz")
            elementtables=data["elementtables"]
            values_list=data["values_list"]
            space_group=data["space_group"]
            energy_atom=data["energy_atom"]
        
        remain_index=((values_list<4) & (values_list>0.1)).reshape(-1)
        #remain_index=(values_list<4).reshape(-1)
        elementtables=elementtables[remain_index]
        values_list=values_list[remain_index]
        space_group=space_group[remain_index.squeeze()]
        elementtables=np.concatenate([elementtables[:,0:3,:,:],\
                                      np.expand_dims(elementtables[:,3,:,:]-elementtables[:,4,:,:],axis=1),\
                                        elementtables[:,5:]],axis=1)
        energy_atom=energy_atom[remain_index.squeeze()]

        assert len(elementtables)==len(values_list)==len(space_group), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.space_group=torch.tensor(space_group,dtype=torch.int32,device=device)
        self.energy_atom=torch.tensor(energy_atom,dtype=torch.float32,device=device)
        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx], self.space_group[idx],self.energy_atom[idx]]


class elementtables_structure(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None,appendix_list=["species_pp_ZVAL","ldau_l","ldau_u","ldau_j"]):
        super().__init__()
        save_dataname="elementtables_structure.npz"
        elementtables=[]
        values_list=[]
        space_group=[]
        energy_atom=[]
        vol_list=[]
        if not os.path.isfile(save_dataname):
            # read the heusler data
            for i in datafile:
                df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
                df.fillna({"spacegroup_relax":1,"energy_atom":0,"spin_cell":0,"spinD":0},inplace=True)
                dataset_dir="datasets/"+i+"/structure/"
                total_len=len(os.listdir(dataset_dir))
                time0=time.time()
                filepath_list=[]
                compoundname_list=[]
                appendix_dict_list=[]
                spin_list=[]
                for count_num,j in enumerate(os.listdir(dataset_dir)):
                    if j.split(".")[-1]=="cif":
                        continue
                    compoundname=j.split(".")[0]
                    item_index=df[df["compound"]==compoundname].index[0]
                    #check_result=check_mag_satisfaction(compoundname,df.at[item_index,"spinD"],df.at[item_index,"spin_cell"])
                    #if check_result==-1:
                    #    continue
                    compoundname_list.append(compoundname)
                    filepath_list.append(dataset_dir+j)
                    appendix_dict_list.append({k:df.at[item_index,k] for k in appendix_list})
                    spin_list.append(df.at[item_index,"spin_cell"])
                    if len(filepath_list)==para_thread or count_num==total_len-1:
                        site_rep_and_vol_list=pmap(site_rep,filepath_list)
                        selected_args=[i for i in range(len(site_rep_and_vol_list)) if isinstance(site_rep_and_vol_list[i],tuple)]
                        site_rep_array=np.array([site_rep_and_vol_list[i][0] for i in selected_args])
                        vol_array=np.array([site_rep_and_vol_list[i][1] for i in selected_args])
                        sub_elementtables,sub_values_list,scale_factor=data2elementtable([compoundname_list,spin_list],return_scale=True,appendix=pd.DataFrame(appendix_dict_list))
                        sub_elementtables=np.concatenate([sub_elementtables[selected_args],site_rep_array],axis=1)
                        sub_values_list=(sub_values_list/scale_factor)[selected_args]
                        values_list.append(sub_values_list)
                        elementtables.append(sub_elementtables)
                        vol_list.append(vol_array)
                        for compoundname in np.array(compoundname_list)[selected_args]:
                            space_group.append(df.at[df[df["compound"]==compoundname].index[0],"spacegroup_relax"])
                            energy_atom.append(df.at[df[df["compound"]==compoundname].index[0],"energy_atom"])
                        filepath_list=[]
                        compoundname_list=[]
                        appendix_dict_list=[]
                        spin_list=[]
                    if count_num%1000==0:
                        print(count_num/total_len,time.time()-time0,i)

            elementtables=np.concatenate(elementtables,axis=0)
            values_list=np.absolute(np.concatenate(values_list,axis=0))
            vol_list=np.concatenate(vol_list,axis=0)
            space_group=np.array(space_group)
            energy_atom=np.array(energy_atom)

            np.savez(save_dataname,elementtables=elementtables,\
                     values_list=values_list,space_group=space_group,\
                        energy_atom=energy_atom,vol_list=vol_list)

        else:
            data=np.load(save_dataname)
            elementtables=data["elementtables"]
            values_list=data["values_list"]
            space_group=data["space_group"]
            energy_atom=data["energy_atom"]
            vol_list=data["vol_list"]
        
        remain_index=((values_list<4) & (values_list>0.1)).reshape(-1)
        #remain_index=(values_list<4).reshape(-1)
        elementtables=elementtables[remain_index]
        values_list=values_list[remain_index]
        space_group=space_group[remain_index.squeeze()]
        elementtables=np.concatenate([elementtables[:,0:3,:,:],\
                                      np.expand_dims(elementtables[:,3,:,:]-elementtables[:,4,:,:],axis=1),\
                                        elementtables[:,5:]],axis=1)
        energy_atom=energy_atom[remain_index.squeeze()]
        vol_list=vol_list[remain_index.squeeze()]

        assert len(elementtables)==len(values_list)==len(space_group)==len(vol_list), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.space_group=torch.tensor(space_group,dtype=torch.int32,device=device)
        self.energy_atom=torch.tensor(energy_atom,dtype=torch.float32,device=device)
        self.vol_list=torch.tensor(vol_list,dtype=torch.float32,device=device)

        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx],\
                self.space_group[idx],self.energy_atom[idx],self.vol_list[idx]]


class elementtables_structure_new(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None,appendix_list=["species_pp_ZVAL","ldau_l","ldau_u","ldau_j"],
                  search=False,task='Mag'):
        super().__init__()
        save_dataname="elementtables_structure_new.npz"
        elementtables=[]
        mag_atom=[]
        space_group=[]
        energy_atom=[]
        vol_list=[]
        if not os.path.isfile(save_dataname):
            # read the heusler data
            for i in datafile:
                df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
                df.fillna({"spacegroup_relax":1,"energy_atom":0,"spin_cell":0,"spinD":0},inplace=True)
                dataset_dir="datasets/"+i+"/structure/"
                total_len=len(os.listdir(dataset_dir))
                time0=time.time()
                filepath_list=[]
                compoundname_list=[]
                appendix_dict_list=[]
                spin_list=[]
                for count_num,j in enumerate(os.listdir(dataset_dir)):
                    if j.split(".")[-1]=="cif":
                        continue
                    compoundname=j.split(".")[0]
                    item_index=df[df["compound"]==compoundname].index[0]
                    #check_result=check_mag_satisfaction(compoundname,df.at[item_index,"spinD"],df.at[item_index,"spin_cell"])
                    #if check_result==-1:
                    #    continue
                    compoundname_list.append(compoundname)
                    filepath_list.append(dataset_dir+j)
                    appendix_dict_list.append({k:df.at[item_index,k] for k in appendix_list})
                    spin_list.append(df.at[item_index,"spin_cell"])
                    if len(filepath_list)==para_thread or count_num==total_len-1:
                        site_rep_and_vol_list=pmap(site_rep,filepath_list)
                        selected_args=[i for i in range(len(site_rep_and_vol_list)) if isinstance(site_rep_and_vol_list[i],tuple)]
                        site_rep_array=np.array([site_rep_and_vol_list[i][0] for i in selected_args])
                        vol_array=np.array([site_rep_and_vol_list[i][1] for i in selected_args])
                        sub_elementtables,sub_mag_atom,scale_factor=data2elementtable([compoundname_list,spin_list],return_scale=True,appendix=pd.DataFrame(appendix_dict_list))
                        sub_elementtables=np.concatenate([sub_elementtables[selected_args],site_rep_array],axis=1)
                        sub_mag_atom=(sub_mag_atom/scale_factor)[selected_args]
                        mag_atom.append(sub_mag_atom)
                        elementtables.append(sub_elementtables)
                        vol_list.append(vol_array)
                        for compoundname in np.array(compoundname_list)[selected_args]:
                            space_group.append(df.at[df[df["compound"]==compoundname].index[0],"spacegroup_relax"])
                            energy_atom.append(df.at[df[df["compound"]==compoundname].index[0],"energy_atom"])
                        filepath_list=[]
                        compoundname_list=[]
                        appendix_dict_list=[]
                        spin_list=[]
                    if count_num%1000==0:
                        print(count_num/total_len,time.time()-time0,i)

            elementtables=np.concatenate(elementtables,axis=0)
            mag_atom=np.absolute(np.concatenate(mag_atom,axis=0))
            vol_list=np.concatenate(vol_list,axis=0)
            space_group=np.array(space_group)
            energy_atom=np.array(energy_atom)

            np.savez(save_dataname,elementtables=elementtables,\
                     values_list=mag_atom,space_group=space_group,\
                        energy_atom=energy_atom,vol_list=vol_list)

        else:
            data=np.load(save_dataname)
            elementtables=data["elementtables"]
            mag_atom=data["values_list"]
            space_group=data["space_group"]
            energy_atom=data["energy_atom"]
            vol_list=data["vol_list"]
        
        a=((2)**(1/2)*vol_list)**(1/3) # the lattice constant of the primitive cell
        #a=a-np.mean(a)
        if search:
            remain_index=((mag_atom>=0.01) & (mag_atom<4) & (elementtables[:,0,2,4]==0).reshape(-1,1)).reshape(-1)
            #remain_index=((mag_atom>=0.01) & (mag_atom<4)).reshape(-1)
        else:
            if task=='Mag' or task=='lattice_constant':
                #remain_index=((a<6).reshape(-1,1) & (mag_atom<4) & (mag_atom>0.001)).reshape(-1)
                #remain_index=((mag_atom<4) & (mag_atom>0.01)).reshape(-1)
                remain_index=((mag_atom>=0.01) & (mag_atom<4) & (elementtables[:,0,2,4]==0).reshape(-1,1)).reshape(-1)
            elif task=='E0':
                remain_index=((energy_atom<=0) & (a<6)).reshape(-1)
        elementtables=elementtables[remain_index]
        mag_atom=mag_atom[remain_index]
        space_group=space_group[remain_index.squeeze()]
        elementtables=np.concatenate([elementtables[:,0:3,:,:],\
                                      np.expand_dims(elementtables[:,3,:,:]-elementtables[:,4,:,:],axis=1),\
                                        elementtables[:,5:]],axis=1)
        energy_atom=energy_atom[remain_index.squeeze()]
        vol_list=vol_list[remain_index.squeeze()]
        a=a[remain_index.squeeze()]

        assert len(elementtables)==len(mag_atom)==len(space_group)==len(vol_list)==len(a), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.mag_atom=torch.tensor(mag_atom,dtype=torch.float32,device=device)
        self.space_group=torch.tensor(space_group,dtype=torch.int32,device=device)
        self.energy_atom=torch.tensor(energy_atom,dtype=torch.float32,device=device).reshape(-1,1)
        self.vol_list=torch.tensor(vol_list,dtype=torch.float32,device=device).reshape(-1,1)
        self.a=torch.tensor(a,dtype=torch.float32,device=device).reshape(-1,1)

        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx],self.mag_atom[idx],\
                self.space_group[idx],self.a[idx],self.energy_atom[idx]]


class elementtables_structure_full_heusler(torch.utils.data.Dataset):
    # this dataset contains only full heusler alloys with formation energy, stability and volume
    def __init__(self,device="cuda"):
        super().__init__()
        save_dataname="elementtables_structure_full_heusler.npz"
        if not os.path.isfile(save_dataname):
            path="datasets/L2_1_FullHeusler_Cu2MnAl/"
            elementtables=[]
            volumes=[]
            formation_energy=[]
            stability=[]
            scale=[]
            space_group=[]
            for file_path in glob.glob(path+'*.json'):
                with open(file_path, 'r') as f:
                    f_json=json.load(f)
                data=f_json['data']
                for j in range(len(data)):
                    site_rep=site_rep_from_OQMD(data[j]['sites'])
                    if isinstance(site_rep,int):
                        continue
                    elementtable_tmp,scale_tmp=get_component_vector([data[j]['name']],return_scale=True)
                    volumes_tmp=data[j]['volume']
                    formation_energy_tmp=data[j]['delta_e']
                    stability_tmp=data[j]['stability']
                    space_group_tmp=data[j]['spacegroup']
                    if formation_energy_tmp is None or stability_tmp is None or \
                        volumes_tmp is None or space_group_tmp is None:
                        continue
                    elementtables.append(np.concatenate([elementtable_tmp.reshape(-1,10,10),np.zeros((3,10,10)),site_rep],axis=0))
                    volumes.append(volumes_tmp)
                    formation_energy.append(formation_energy_tmp)
                    stability.append(stability_tmp)
                    scale.append(scale_tmp)
                    if space_group_tmp=="Fm-3m":
                        space_group.append(225)
                    elif space_group_tmp=="F-43m":
                        space_group.append(216)
                    else:
                        space_group.append(1)
            elementtables=np.array(elementtables)
            volumes=np.array(volumes)
            formation_energy=np.array(formation_energy)
            stability=np.array(stability)
            scale=np.array(scale)
            space_group=np.array(space_group)
            np.savez(save_dataname,elementtables=elementtables,volumes=volumes,\
                     formation_energy=formation_energy,stability=stability,scale=scale,space_group=space_group)
        else:
            data=np.load(save_dataname)
            elementtables=data["elementtables"]
            volumes=data["volumes"]
            formation_energy=data["formation_energy"]
            stability=data["stability"]
            scale=data["scale"]
            space_group=data["space_group"]
        
        remain_index=((formation_energy<4)&(stability<3.3)).reshape(-1)
        elementtables=elementtables[remain_index]
        volumes=volumes[remain_index]
        formation_energy=formation_energy[remain_index]
        stability=stability[remain_index]
        scale=scale[remain_index]
        space_group=space_group[remain_index]

        self.elementtables=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.volumes=torch.tensor(volumes.reshape(-1,1),dtype=torch.float32,device=device)
        self.lattice_constant=torch.tensor((((2)**(1/2)*volumes)**(1/3)).reshape(-1,1),dtype=torch.float32,device=device) # the lattice constant of the primitive cell
        self.formation_energy=torch.tensor(formation_energy.reshape(-1,1),dtype=torch.float32,device=device)
        self.stability=torch.tensor(stability.reshape(-1,1),dtype=torch.float32,device=device)
        self.scale=torch.tensor(scale,dtype=torch.float32,device=device)
        self.space_group=torch.tensor(space_group,dtype=torch.int32,device=device)

    def __len__(self):
        return len(self.elementtables)
    
    def __getitem__(self, idx):
        return [self.elementtables[idx],self.formation_energy[idx],self.space_group[idx],self.stability[idx]]


def check_mag_satisfaction(component,spinD,spin_cell,principal_element_mag_max=0.2,
                           spinD_spin_cell_delta=0.08):
    # make sure principal element magnetic moment is less than principal_element_mag_max
    # make sure abs((sum(spinD)-spin_cell)/spin_cell) is less than spinD_spin_cell_delta
    # make sure not all spinD is zero
    # input: component, spinD, spin_cell
    # output: 1 for True or -1 for False
    name,num=readComponent([component])
    name=name[0]
    num=num[0]
    if spinD==0:
        return -1
    elif isinstance(spinD,str):
        spinD=ast.literal_eval(spinD)
    if isinstance(spin_cell,str):
        spin_cell=float(spin_cell)
    if np.sum(spinD)==0:
        return -1
    if spin_cell==0:
        return -1
    if np.abs((np.sum(spinD)-spin_cell)/np.sum(num))>spinD_spin_cell_delta:
        return -1
    count=0
    for i in range(len(name)):
        for j in range(int(num[i])):
            if name[i] in PrincipalElements and spinD[count]>principal_element_mag_max:
                return -1
            count+=1
    return 1

def site_rep(file_path,delta=0.01):
    # read the site representation of the structure,
    # for heusler alloys, there are 4 sites, each site corresponds to a channel
    # if the site is occupied, the occupied element of corresponding channel is 1, otherwise 0
    # input: file_path, the path of the structure file
    # output: four channels of site representation, and volume of the structure
    poscar = Poscar.from_file(file_path)
    cposcar_structure=SpacegroupAnalyzer(poscar.structure).get_conventional_standard_structure()
    pposcar_structure=cposcar_structure.get_primitive_structure()
    volume=pposcar_structure.volume
    atom_number=pposcar_structure.atomic_numbers
    frac_coords=pposcar_structure.frac_coords
    if len(atom_number)>4:
        return -1
    site_rep=site_rep_only(atom_number,frac_coords,delta)
    return site_rep, volume

def site_rep_only(atom_number,frac_coords,delta=0.01):
    site_rep=np.zeros((4,100))
    for i in range(len(atom_number)):
        if abs(np.sum(frac_coords[i]-[0,0,0]))<delta or abs(np.sum(frac_coords[i]-[1,1,1]))<delta:
            site_rep[0,atom_number[i]-1]=1
        elif abs(np.sum(frac_coords[i]-[0.25,0.25,0.25]))<delta:
            site_rep[1,atom_number[i]-1]=1
        elif abs(np.sum(frac_coords[i]-[0.5,0.5,0.5]))<delta:
            site_rep[2,atom_number[i]-1]=1
        elif abs(np.sum(frac_coords[i]-[0.75,0.75,0.75]))<delta:
            site_rep[3,atom_number[i]-1]=1
        else:
            return -1
    return site_rep.reshape(4,10,10)

def site_rep_from_OQMD(site,delta=0.01):
    # the site parameter like ['Lu @ 0.25 0.25 0.25','Os @ 0.75 0.75 0.75'
    # , 'Sc @ 0 0 0', 'Sc @ 0.5 0.5 0.5']
    elements=[i.split(' @ ')[0] for i in site]
    frac_coords=[np.array([float(j) for j in i.split(' @ ')[1].split()]) for i in site]
    atom_number=[np.argwhere(ChemicalSymbols==i)[0][0]+1 for i in elements]
    return site_rep_only(atom_number,frac_coords,delta)

class adjacent_matrix_datasets(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=aflow_prototype,
                  device="cuda", transform=None,cutoff_radius=15,
                  sigma=0.1,contain_near_numbers=5,elements_list=ChemicalSymbols):
        super(adjacent_matrix_datasets, self).__init__()
        

        if not (os.path.isfile("adjacent_matrix.npy") and os.path.isfile("adjacent_matrix_label.npy")):
            elementtables=[]
            values_list=[]
            #datafile=['ABC_cF12_216_c_b_a']
            for i in datafile:
                df=pd.read_csv("datasets/"+i+"/"+i+'.txt',sep='\t',header=0)
                dataset_dir="datasets/"+i+"/structure/"
                total_len=len(os.listdir(dataset_dir))
                time0=time.time()
                filepath_list=[]
                filename_list=[]
                for count_num,j in enumerate(os.listdir(dataset_dir)):
                    if j.split(".")[-1]=="cif":
                        continue
                    filepath_list.append(dataset_dir+j)
                    filename_list.append(j)
                    if len(filepath_list)==para_thread or count_num==total_len-1:
                        poscar_list=pmap(read_poscar,filepath_list)
                        adajcent_matrix_list=pmap(degenerate_adjacent_matrix,poscar_list,(cutoff_radius,sigma,contain_near_numbers,elements_list,))
                        for k in range(len(poscar_list)):
                            if isinstance(adajcent_matrix_list[k],int):
                                continue
                            poscar=poscar_list[k]
                            filename=filename_list[k]
                            adajcent_matrix=adajcent_matrix_list[k]
                            total_spin=df.at[df[df["compound"]==filename.split(".")[0]].index[0],"spin_cell"]
                            total_spin=torch.tensor(np.abs(np.array(total_spin))/len(poscar.structure.atomic_numbers),dtype=torch.float32)
                            elementtables.append(adajcent_matrix)
                            values_list.append(total_spin)
                        filepath_list=[]
                        filename_list=[]
                    if count_num%1000==0:
                        print(count_num/total_len,time.time()-time0,i)

            elementtables=np.array(elementtables)
            values_list=np.absolute(values_list)

            np.save("adjacent_matrix.npy",elementtables)
            np.save("adjacent_matrix_label.npy",values_list)

        else:
            elementtables=np.load("adjacent_matrix.npy")
            values_list=np.load("adjacent_matrix_label.npy")

        remain_index=((values_list>0.1) &(values_list<4)).reshape(-1)
        elementtables=elementtables[remain_index]
        values_list=values_list[remain_index]

        #elementtables=elementtables[:,:4,:,:]

        #for i in range(len(elementtables)):
        #    for j in range(contain_near_numbers*2):
        #        for row in range(len(elements_list)):
        #            elementtables[i,j,row,row]=-elementtables[i,j,row,row]
                    #for col in range(len(elements_list)):
                    #    if row>col and row<len(elements_list)-1-col:
                    #        tmp=elementtables[i,j,row,col]
                    #        elementtables[i,j,row,col]=-elementtables[i,j,len(elements_list)-col-1,len(elements_list)-row-1]
                    #        elementtables[i,j,len(elements_list)-col-1,len(elements_list)-row-1]=-tmp

        assert len(elementtables)==len(values_list), 'The length of elementtables and values_list should be equal'
        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx]]
    
class TcDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir='data/', datafile=["Tc_superconductor_data.txt","Tc_insulator_data.txt"],
                  device="cuda", transform=None):
        super(TcDataset, self).__init__()
        elementtables=[]
        values_list=[]
        
        # read the superconductor data
        with open("data/Tc_superconductor_data.txt", "r") as f:
            data = f.readlines()
        sub_elementtables,sub_values_list=data2elementtable([[i.split()[0] for i in data],[float(i.split()[1]) for i in data]])
        elementtables.append(sub_elementtables)
        values_list.append(sub_values_list)

        # read the insulator data, ie non-superconductor data
        with open("data/Tc_insulator_data.txt", "r") as f:
            data = f.readlines()
        sub_elementtables,sub_values_list=data2elementtable([[i.split()[1] for i in data],[float(i.split()[3]) for i in data]])
        elementtables.append(sub_elementtables)
        values_list.append(sub_values_list)
        
        elementtables=np.concatenate(elementtables,axis=0)
        values_list=np.concatenate(values_list,axis=0)
        assert len(elementtables)==len(values_list), 'The length of elementtables and values_list should be equal'

        self.datafile=torch.tensor(elementtables,dtype=torch.float32,device=device)
        self.labels=torch.tensor(values_list,dtype=torch.float32,device=device)
        self.root_dir=root_dir

    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx):
        return [self.datafile[idx], self.labels[idx]]


def extract_cartcoords(file_path,len_max=6):
    # extract the cartesian coordinates from the vasp file
    # input: file path
    # output: cartesian coordinates elementtable
    cart_coords_list=[]
    cart_elementtable=np.zeros((len_max,100))-1
    poscar = Poscar.from_file(file_path)
    pstructure=Structure.get_primitive_structure(poscar.structure)
    for atom_number in set(pstructure.atomic_numbers):
        cart_coords_temp=np.array(pstructure.cart_coords[np.array(pstructure.atomic_numbers)==atom_number]).reshape(-1)
        if len(cart_coords_temp)>len_max:
            return 0
        cart_elementtable[:len(cart_coords_temp),atom_number-1]=cart_coords_temp
    return cart_elementtable.reshape(len_max,10,10)


def degenerate_adjacent_matrix(poscar,cutoff_radius=7,sigma=0.1,contain_near_numbers=2,elements_list=ChemicalSymbols):
    center_indices, points_indices, offset_vectors, distances = \
        poscar.structure.get_neighbor_list(r=cutoff_radius, numerical_tol=1e-8, exclude_self=True)
    # this part is to reduce the size of the adjacent matrix, for example, original shape is (*,100,100)
    # but some elements may not be used, so we only keep the used elements, where used elements is from elements_list
    # before we calculate the adjacent matrix, we need to transfer the atomic numbers to the index of elements_list
    atomnames=ChemicalSymbols[np.array(poscar.structure.atomic_numbers)-1]
    elements_list_arg=[]
    for i in atomnames:
        if i not in elements_list:
            return 1
        elements_list_arg.append(np.argwhere(np.array(elements_list)==i)[0][0])
    elements_list_arg=np.array(elements_list_arg)
    trans_center_indices=elements_list_arg[center_indices]
    trans_points_indices=elements_list_arg[points_indices]
    dim=len(elements_list)
    adjacent_matrix=np.zeros((contain_near_numbers*2,dim,dim))
    for i in set(elements_list_arg):
        for j in set(elements_list_arg):
            threshold=sigma
            arg=(trans_center_indices==i) & (trans_points_indices==j) &(distances>threshold)
            distance_sec=distances[arg]
            for k in range(contain_near_numbers):
                distance_sec=distance_sec[distance_sec>=threshold]
                if len(distance_sec)==0:
                    break
                else:
                    adjacent_matrix[k*2,i,j]=1/np.min(distance_sec)
                    adjacent_matrix[k*2+1,i,j]=np.count_nonzero(distance_sec-np.min(distance_sec)<sigma)
                    threshold=np.min(distance_sec+sigma)
    return adjacent_matrix

def random_split(length,p=[0.8,0.2]):
    # random split the dataset into training set and test set
    # input: length of the dataset, p is split proportion
    # output: mask of training set and test set, may contain validation set
    if isinstance(p, float):
        rand_split=np.random.rand(length)
        train_mask=rand_split<p
        test_mask=rand_split>=p
        return train_mask, test_mask
    assert len(p) in [2,3], 'The length of p should be 2 or 3'
    assert np.sum(p)==1, 'The sum of p should be 1'
    if len(p)==2:
        rand_split=np.random.rand(length)
        train_mask=rand_split<p[0]
        test_mask=rand_split>=p[0]
        return train_mask, test_mask
    elif len(p)==3:
        rand_split=np.random.rand(length)
        train_mask=rand_split<p[0]
        val_mask=(rand_split>=p[0])&(rand_split<p[0]+p[1])
        test_mask=rand_split>=p[0]+p[1]
        return train_mask, val_mask, test_mask

def K_fold_split(length,K=5):
    # K-fold split the dataset into training set and test set
    # input: length of the dataset, K is the number of fold
    # output: mask of test set
    rand_split=K*np.random.rand(length)
    fold_mask=[]
    for i in range(K):
        fold_mask.append(np.logical_and(rand_split>=i,rand_split<i+1))
    return fold_mask

if __name__ == "__main__":

    """测试"""
    os.chdir("..")
    b = elementtables_structure_new()

    print(b[0])