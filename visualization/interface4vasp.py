import numpy as np
import os
import glob

def generate_arrays(input, current, index, output):
    if index == len(input):
        output.append(current[:])
        return
    current[index] = input[index]
    generate_arrays(input, current, index + 1, output)
    if input[index] > 0:
        current[index] = -input[index]
        generate_arrays(input, current, index + 1, output)
    
def discard_equivalent_arrays(output):
    output_simplify=[]
    append_sign=True
    for i in output:
        for j in output_simplify:
            if np.all(np.array(i)==-np.array(j)):
                append_sign=False
                break
        if append_sign:
            output_simplify.append(i)
        append_sign=True
    return output_simplify

def get_all_possible_arrays(input):
    output = []
    generate_arrays(input, input[:], 0, output)
    return discard_equivalent_arrays(output)

PrincipalElements = np.array([ 'H',  'He', 'Li', 'Be','B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si','P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Ga', 'Ge', 'As', 'Se','Br', 'Kr', 'Rb', 'Sr', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe','Cs', 'Ba', 'Tl', 'Pb', 'Bi', 
                    'Po','At', 'Rn', 'Fr', 'Ra'])

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

vaspfiles=glob.glob("*.vasp")

for i in vaspfiles:
    with open(i) as f:
        poscar=f.readlines()
    os.system("mkdir %s"%(i.split('.')[0]))
    os.chdir("%s"%(i.split('.')[0]))
    os.system("cp ../%s ./POSCAR"%(i))
    os.system("echo -e '102\n2\n0.02' | vaspkit")
    os.system("cp ../INCAR ./")

    elements=poscar[5].strip().split()
    number=poscar[6].strip().split()
    print(number,elements)
    magmom=[]
    for numberi in range(len(number)):
        for numberj in range(int(number[numberi])):
            magmom.append(0 if elements[numberi] in PrincipalElements else 5)
    os.system("cat >> INCAR <<EOF\nMAGMOM = %s\nEOF"%(" ".join([str(i) for i in magmom])))
    
    element_index=[np.where(element==ChemicalSymbols)[0][0]+1 for element in elements]
    zval=[int(float(j.split()[5])) for j in os.popen("grep ZVAL POTCAR").read().split("\n") if j!=""]
    contain_f=np.any(((np.array(element_index)>=57) & (np.array(element_index)-np.array(zval)<=71)) | ((np.array(element_index)>=89) & (np.array(element_index)-np.array(zval)<=103)))
    os.system("cat >> INCAR <<EOF\nLMAXMIX = %d\nEOF"%(6 if contain_f else 4))
    
    os.system('cp ../vasp.sh ./')
    #os.system('sbatch vasp.sh')
    os.chdir("..")
