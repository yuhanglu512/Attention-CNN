import json
import pandas as pd
from urllib.request import urlopen
import os
try:
    from folder_names import aflow_prototype
except ImportError:
    from .folder_names import aflow_prototype
from parfor import pmap


API='http://aflow.org/API/aflux/'


def download_function(request):
    try:
        response=urlopen(request,timeout=1000).read().decode('UTF-8')
    except:
        return 0
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        return response

def download_dataframe(aflow_prototype):
    para_thread=1
    os.chdir("datasets/")
    for i in range(len(aflow_prototype)):
        MATCHBOOK='$auid,compound,energy_atom,species_pp_ZVAL,ldau_l,ldau_u,ldau_j,spinD,spin_cell,bader_net_charges'
        MATCHBOOK+=',$aflow_prototype_label_relax('+aflow_prototype[i]+'),positions_fractional'
        os.chdir(aflow_prototype[i])
        stop_sign=False

        for j in range(208,400):
            #DIRECTIVE='$paging('+str(page)+',500),format(json)'
            #REQUEST=API+"?"+MATCHBOOK+","+DIRECTIVE
            REQUEST=[API+"?"+MATCHBOOK+","+'$paging('+str(j*para_thread+k)+',500),format(json)' for k in range(1,para_thread+1)]
            response_list=pmap(download_function,REQUEST)
            for k in range(para_thread):
                response=response_list[k]
                if response==0:
                    with open('error_download_datframe.txt', 'a') as f:
                        f.write(REQUEST[k]+'\n')
                    continue
                if response==[]:
                    stop_sign=True
                    break
                df=pd.DataFrame(response)
                if j==0 and k==0:
                    with open(aflow_prototype[i]+'.txt', 'w') as f:
                        f.write(df.to_csv(header=True, index=False, sep='\t'))
                else:
                    with open(aflow_prototype[i]+'.txt', 'a') as f:
                        f.write(df.to_csv(header=False, index=False, sep='\t'))

            if stop_sign:
                break

        os.chdir("..")
    os.chdir("..")

def download_structure(aflow_prototype):
    para_thread=200
    os.chdir("datasets/")
    for i in range(len(aflow_prototype)):
        os.chdir(aflow_prototype[i])
        df=pd.read_csv(aflow_prototype[i]+'.txt',sep='\t',header=0)
        downloaded=[item.split(".")[0] for item in os.listdir("structure/")]
        url_list=[]
        structure_name_list=[]
        for j in range(len(df)):
            item=df.iloc[j]
            if item["compound"] in downloaded:
                continue
            url_list.append("http://"+item["aurl"].replace(":AFLOWDATA","/AFLOWDATA")+"/CONTCAR.relax")
            structure_name_list.append(item["compound"])
            if len(url_list)==para_thread or j==len(df)-1:
                response_list=pmap(download_function,url_list)
                for k in range(len(response_list)):
                    response=response_list[k]
                    if response==0:
                        with open('error_download_structure.txt', 'a') as f:
                            f.write(url_list[k]+'\n')
                        continue
                    with open('structure/'+structure_name_list[k]+'.vasp', 'w') as f:
                        f.write(response)
                url_list=[]
                structure_name_list=[]
            #try:
            #    response="http://"+item["aurl"].replace(":AFLOWDATA","/AFLOWDATA")+"/CONTCAR.relax"
            #    response_restapi=urlopen(response,timeout=200).read().decode('UTF-8')
            #    with open('structure/'+item["compound"]+'.vasp', 'w') as f:
            #        f.write(response_restapi)
            #    #response="http://"+item["aurl"].replace(":AFLOWDATA","/AFLOWDATA")+"/POSCAR.relax"
            #except:
            #    with open('../../utils/download_error.txt', 'a') as f:
            #        f.write(item["aurl"]+"\n")
            #    continue
        os.chdir("..")
    os.chdir("..")


if __name__=="__main__":
    os.chdir("../")
    #download_dataframe(aflow_prototype)
    download_structure(aflow_prototype)