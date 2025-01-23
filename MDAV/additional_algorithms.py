
import numpy as np
import pandas as pd
import scipy
import os
import re

def extract_m_and_Monefile(path):
    numbers=re.findall(r"-?\d+\.?\d*", path)
    m=float(numbers[0])
    M=float(numbers[1])
    return m, M

def deleted_element_0(path):
    df=pd.read_csv(path)
    df=df[df.iloc[:, 1]!=0]
    df.to_csv(path, index=False)


def generate_probabilities_csv(m, M, path_distances, path_probabilities):
    df=pd.read_csv(path_distances)
    name= "probabilities m=" + str(m)+ " M=" +str(M)
    df.columns=[name]
    # df.rename(columns={"Average distances": "Roll_no"}, inplace=True)
    average_distance=df.iloc[:, 0]
    p=m+(M-m)*average_distance
    probability_of_being_sampled=1-p
    probability_of_being_sampled.to_csv(path_probabilities, index=False)


# extract_m_and_M_List(path: str ):
    



def extract_m_and_MofPath(path): 
    numbers=[]
    for i in range(len(path)):
        for j in re.findall(r"-?\d+\.?\d*", path[i]):
            numbers.append(float(j))
    half_total_element=int(len(numbers)/2)
    m_and_M=np.reshape(numbers, (half_total_element,-1))
    return m_and_M

def deleted_element0_of_path(path):
    list_dir=os.listdir(path)
    for i in range(len(list_dir)):
        path_file=path + "\\" + list_dir[i]
        deleted_element_0(path_file)

def join_csv_of_carpet(path_of_carpet, path_of_archive):
    "This function joins all the CSVs in a folder into one, to make it easier to get your work done with pandas"
    list_dir=os.listdir(path_of_carpet)
    df_probe=pd.read_csv(path_of_carpet + "\\" + list_dir[0])
    list_columns=list(df_probe.columns)
    df=pd.DataFrame(columns=list_columns)
    for i in range(len(list_dir)):
        df1=pd.read_csv(path_of_carpet + "\\" + list_dir[i])
        df=pd.concat([df, df1])
    df.to_csv(path_of_archive, index=None)