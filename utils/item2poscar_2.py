import os
import pandas as pd
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import re
import ase
from pymatgen.core.composition import Composition
import warnings
#warnings.filterwarnings("error")
try:
    from .folder_names import aflow_prototype
except ImportError:
    from folder_names import aflow_prototype
import ast
from parfor import pmap
from time import sleep

para_thread=200

ChemicalSymbols = np.array([ 'H',  'He', 'Li', 'Be','B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si','P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr','Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se','Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W','Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po','At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu','Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr'])

PrincipalElements = np.array([ 'H',  'He', 'Li', 'Be','B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si','P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Ga', 'Ge', 'As', 'Se','Br', 'Kr', 'Rb', 'Sr', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe','Cs', 'Ba', 'Tl', 'Pb', 'Bi', 
                    'Po','At', 'Rn', 'Fr', 'Ra'])

RareearthElements = np.array(['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                              'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                              'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                              'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'])

abundance20=['Be', 'Na', 'Mg', 'K', 'Ca', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Ag',
           'Cd', 'Sn', 'Ba', 'La', 'Hf', 'Hg', 'Tl', 'Pb', 'Bi']

def readComponent_old(comps: list):
    # find all the elements and percentage in a compound
    # input: compounds in string format
    # output: a list of elements and a list of corresponding percentage
    namelist=[]
    numlist=[]
    for i in comps:
        namelist.append(re.findall(r'[A-Z][a-z]*', i))
        # find all the numbers corresponding to the elements because some elements occur in string without numbers
        numlist_tmp=[]
        s=i
        for j in range(len(namelist[-1])):
            s=s.replace(namelist[-1][j],"",1) # discard the first element
            if len(s)==0:
                numlist_tmp.append(1) # suit the case that the last element has no number
            elif s[0].isupper():
                numlist_tmp.append(1) # suit the case some elements with no specific number
            else:
                first_number = re.search(r'\d+\.\d+|\d+', s).group() # find the first number
                numlist_tmp.append(float(first_number))
                s=s.replace(first_number,"",1) # discard the first number

        if len(numlist_tmp)!=len(namelist[-1]): # check if the number of elements and numbers are equal
            print(i)
            raise ValueError("The number of elements and numbers are not equal!")

        numlist.append(numlist_tmp)
    
    return namelist, numlist

def readComponent(comps:list):
    # find all the elements and percentage in a compound
    # input: compounds in list format
    # output: a list of elements and a list of corresponding percentage
    namelist=[]
    numlist=[]
    for i in comps:
        namelist_tmp=[]
        numlist_tmp=[]
        if len(i.split(','))==2:   # for the case that the compound is in the format of "compound, space_group"
            for name,num in Composition(i.split(',')[0]).as_dict().items():
                namelist_tmp.append(name)
                numlist_tmp.append(num)
            namelist.append(namelist_tmp)
            numlist.append(numlist_tmp)
            continue
        for name,num in Composition(i).as_dict().items():
            namelist_tmp.append(name)
            numlist_tmp.append(num)
        namelist.append(namelist_tmp)
        numlist.append(numlist_tmp)
    
    return namelist, numlist

def get_component_vector(compound:list,return_scale=False,appendix=None):
    # get the component vector of compounds
    # input: compounds in string format, return_scale returns normalization factor if True, 
    # appendix pass something like space group number or hubbard U
    # output: component vectors in numpy array format with shape (*, 10, 10) or (*, channels, 10, 10)
    namelist,numlist=readComponent(compound)
    elementtables=[]
    for i in range(len(namelist)):
        elementtable_tmp=np.zeros(100) if appendix is None else np.zeros((100,len(appendix.columns)+1))
        for j in range(len(namelist[i])):
            site=np.argwhere(ChemicalSymbols==namelist[i][j])[0]
            if appendix is None:
                elementtable_tmp[site]+=numlist[i][j]
            else:
                elementtable_tmp[site,0]+=numlist[i][j]
                for k in range(elementtable_tmp.shape[1]-1):
                    if isinstance(appendix.iloc[i,k],str):
                        elementtable_tmp[site,k+1]=ast.literal_eval(appendix.iloc[i,k])[j]
        elementtables.append(elementtable_tmp)
    elementtables=np.array(elementtables)
    if appendix is None:
        if return_scale:
            scale_factor=np.sum(elementtables,axis=1).reshape(-1,1)
        elementtables=(np.array(elementtables)/np.sum(elementtables,axis=1).reshape(-1,1)).reshape(-1,10,10)
        elementtables=np.expand_dims(elementtables,axis=1)
    else:
        if return_scale:
            scale_factor=np.sum(elementtables[:,:,0],axis=1).reshape(-1,1)
        elementtables[:,:,0]=elementtables[:,:,0]/scale_factor
        elementtables=np.transpose(elementtables,(0,2,1)).reshape(-1,len(appendix.columns)+1,10,10)
    if return_scale:
        return elementtables, scale_factor
    return np.array(elementtables)

def data2elementtable(data:list,return_scale=False,appendix=None):
    # convert the datafile to 10*10 element table
    # input: data, contain string of compound and corresponding value
    # output: component vectors in numpy array format with shape (*, 1, 10, 10)
    assert len(data)==2, 'The length of each line in data should be 2'
    if return_scale:
        elementtables,scale_factor=get_component_vector(data[0],return_scale=True,appendix=appendix)
    else:
        elementtables=get_component_vector(data[0],appendix=appendix)
    values_list=np.array(data[1],dtype=float).reshape(-1,1)
    assert len(elementtables)==len(values_list), 'The length of elementtables and values_list should be equal'
    if return_scale:
        return elementtables, values_list, scale_factor
    return elementtables, values_list


def modify_structure(args):
    # modify a vasp4 structure to a vasp5 structure
    # input: structure in string format, composition in string format
    # output: structure in string format
    structure,comp=args
    namelist, numlist = readComponent([comp])
    s=structure.split("\n")
    if namelist[0]==s[5].split():
        return 0 # mean thereis no need to change, original file is already vasp5 format
    if any([c.isalpha() for c in s[5]]):
        s.remove(s[5])
    if [str(int(num_tmp)) for num_tmp in numlist[0]]!=s[5].split():
        raise ValueError("The number of elements in composition and in structure are not equal!")
    s.insert(5," ".join(namelist[0]))
    if "He" in "\n".join(s):
        return 1 # there is discrepancy in the structure
    return "\n".join(s)

def read_structure(filepath):
    # read the structure from a vasp file
    # input: filepath in string format
    # output: structure in string format
    with open(filepath) as f:
        structure=f.read()
    return structure

def update_structure(args):
    filepath, structure=args
    with open(filepath,"w") as f:
        f.write(structure)



if __name__=="__main__":

    os.chdir("../datasets/")

    for i in range(len(aflow_prototype)):
        name_list=[]
        total_num=len(os.listdir(aflow_prototype[i]+"/structure/"))
        for count_index,j in enumerate(os.listdir(aflow_prototype[i]+"/structure/")):
            if j.split(".")[-1] == "cif" or "SFConflict" in j:
                os.remove(j)
                continue
            name_list.append([aflow_prototype[i]+"/structure/"+j,j.split(".")[0]])
            if len(name_list)==para_thread or count_index==total_num-1:
                structure_list=pmap(read_structure,[sub_name_list[0] for sub_name_list in name_list])
                modify_structure_list=pmap(modify_structure,[[structure_list[k],name_list[k][1]] for k in range(len(name_list))])
                modify_args=[]
                for k in range(len(name_list)):
                    if modify_structure_list[k]==1:
                        with open("../utils/item2poscar_error.txt", "a") as f:
                            f.write(aflow_prototype[i]+"  "+j+"\n")
                            f.writelines(structure_list[k]+"\n")
                    if modify_structure_list[k]!=0 and modify_structure_list[k]!=1:
                        modify_args.append(k)
                if modify_args!=[]:
                    pmap(update_structure,[[name_list[k][0],modify_structure_list[k]] for k in modify_args])
                name_list=[]

