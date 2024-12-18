# Attention-CNN
CNN integrates with Transformer attention, spatial attention to investigate Heusler alloys

## file and folder
1. main.py, this is the main program to run
2. untils, contains files to download data entry, transform poscar and create datasets
   1. down_all_heusler_1.py, download all heusler alloys from aflow, and all subcatgories are listed in folder_names.py
   2. folder_names.py, list all subcatgozies of Heusler alloys in aflow
   3. item2poscar_2.py, modify poscar in vasp version 4.6 to that of format in vasp version 5.4, and also transform cif to poscar
   4. dataset_construct_3.py, construct datasets function with data in datasets folder and save to elementtables_structure_new.py, etc
3. models
   1. GCNmodel.py, contains all the function used or not used in Attention CNN, including spatial attention, transformer, multi-kernel, etc.
4. visualazation, all the post process files are here
   1. attention_show.ipynb, contains how to process and plot attention matrix, part of plotting attention matrix is moved to GCNmodel.py
   2. generate_datasets.py, search all the materials not in the datasets, use Attention CNN to predict and ranking, the final results of these ranking are written to folder results
   3. interface4vasp.py, this is a file to submit job in linux, it create necessary files to start first-principles calculation by vasp
   4. stablility_contrast.ipynb, plot the contrast figure of $E_0$
   5. stable_material_mag_contrast.ipynb, filtering stable materials with high saturated $m_s$ in datasets
   6. visualazation.ipynb, plot the embeddings and PCA of embeddings, part of this function is moved to GCNmodel.py
5. result, most are figures shown in text, some specific files are mentioned below
   1. top_data.txt, filtering high saturated $m_s$ materials not in datasets, the $E_{hull}$ are listed also in this file
   2. top_data_mag.txt, furthermore filter stable materials in top_data.txt, with the criterion $E_{hull} < 0.01 eV/atom$
   3. top_data_mag_metastable.txt, like top_data_mag.txt but with the criterion $E_{hull} < 0.1 eV/atom$
   4. lattice_constant_E0_Ef_stability.pth, trained model parameters for lattice constant $a$, etc
   5. Mag_model.pth, trained model parameters for magnetic moment
6. elementtables_structure_new.npy and elementtables_structure_full_heusler.npz, these two files are pre created datasets file by utils/dataset_construct_3.py
7. datasets contains two many files, including poscar and entry of every items, so we upload its zip file, datasets.zip

## running
direct run main.py. if you want to reconstruct datasets, unzip the datasets.zip to root folder and rerun, the datasets construct process may consume much time
