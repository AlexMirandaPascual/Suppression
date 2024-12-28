
import numpy as np
import pandas as pd
import os
import re

def extract_m_and_MofPath(path): 
    numbers=[]
    for i in range(len(path)):
        for j in re.findall(r"-?\d+\.?\d*", path[i]):
            numbers.append(float(j))
    half_total_element=int(len(numbers)/2)
    m_and_M=np.reshape(numbers, (half_total_element,-1))
    return m_and_M

def extract_m_and_Monefile(path):
    numbers=re.findall(r"-?\d+\.?\d*", path)
    m=float(numbers[0])
    M=float(numbers[1])
    return m, M

def deleted_element_0(path):
    df=pd.read_csv(path)
    df=df[df.iloc[:, 1]!=0]
    df.to_csv(path, index=False)

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


def generate_probabilities_cvs(m, M, path_distances, path_probabilidades):
    df=pd.read_csv(path_distances)
    name= "probabilities m=" + str(m)+ " M=" +str(M)
    df.columns=[name]
    # df.rename(columns={"Average distances": "Roll_no"}, inplace=True)
    promedio=df.iloc[:, 0]
    p=m+(M-m)*promedio
    probabilidad_de_que_este=1-p
    probabilidad_de_que_este.to_csv(path_probabilidades, index=False)




# extract_m_and_M_List(path: str ):
    
def calculate_delta_prima(delta, m):
    delta_prima=delta*(1-m)
    return delta_prima

def calculate_L1(m: float, M: float, eps: float)-> float:
    if eps==0:
         V1=((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m))
    else:
        F=np.exp(eps)
        a1=(F-1) * (M/m) * (np.power((M-m), 2))
        b1=-((M-m)/m) * ((np.power(m,2)-4*M*m +2*M)*(F-1) + F*M)
        c1= ((1-m)/m)* ( (F-1) * (2* np.power(m, 2)-4*M*m-m) + (3*F-1)*M )
        d1=-(1-m) *( (F-1) * (m-2) + (F/m) )
    
        D10=np.power(b1,2)-3*a1*c1
        D11= (2*np.power(b1,3)) - (9*a1*b1*c1) + (27*np.power(a1, 2)* d1)
        R1=np.sqrt(np.power(D10, 3))
       #Opciones D1
        caso=np.power(D11, 2)-4*np.power(D10, 3)
        if caso>=0:
            cubic_pos=(D11 + np.sqrt(caso))/2
            cubic_neg=(D11 - np.sqrt(caso))/2
            V1= -(1/(3*a1))*(b1+np.cbrt(cubic_pos)+np.cbrt(cubic_neg))  
        else:
            V1=-(1/(3*a1))*(b1+2*np.sqrt(D10)*np.cos( (1/3) * np.arccos(D11/(2*R1)) ))  
    maxV1=np.amax([V1, 0]) 
    p=np.amin([1, maxV1])              
    L1=np.log(  F-(F-1)*(p*M+ (1-p)*m)  ) + p*(M/m) + (1-p)*(1-m)/(1-(p*M+(1-p)*m))-1
    
    return L1  

def calculate_L2(m: float, M: float, eps: float)-> float:
    if eps==0:
        V2= 2-(np.sqrt(m*(1-M)))/(1-M)
    else:
        F=np.exp(eps)
        a2 = (F-(F-1)*m)/m
        b2 = -(6*F-(F-1)*(M+5*m))/m
        c2 = (1/(m*(1-M))) * (m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
        d2 = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
        D20 = np.power(b2, 2)-3*a2*c2
        D21 = 2*np.power(b2, 3) - 9*a2*b2*c2+27*np.power(a2, 2)*d2
        R2 = np.sqrt(np.power(D20,3))
        V2=-1/(3*a2)*(b2+2*np.sqrt(D20)*np.cos((1/3)*np.arccos(D21/(2*R2))))
    maxV2=np.amax([V2, 0])
    p=np.amin([1, maxV2])
    dentro_del_log=F-(F-1)*(p*M+(1-p)*(M+m-p*M)/(2-p))
    L2=np.log(dentro_del_log) + p*(M/m) +(1-p)*(1-((M+m-p*M)/(2-p)))/(1-M)-1
    return L2

def calculate_L3(m: float, M: float, eps: float)-> float:
    F=np.exp(eps)
    L3= -(np.log( F + (1-F)*M))  + (1- (1-M)/(1-m))
    return L3

def calculate_eps_prima(m: float, M: float, eps: float)-> float:
    if (m==M):
        F=np.exp(eps)
        A=F-(F-1)*m
        B=1/(F + (1-F)*M)
        max=np.amax([A, B])
        eps_prima= np.log(max)
        return eps_prima
    else: 
        L1=calculate_L1(m, M, eps)
        L2=calculate_L2(m, M, eps)
        L3=calculate_L3(m, M, eps)
        eps_prima=np.amax([L1, L2, L3])
        return eps_prima
# def calcL2(m: float, M: float, eps: float)-> float:
#         F=np.exp(eps)
#         a2 = (F-(F-1)*m)/m
#         b2 = -(6*F-(F-1)*(M+5*m))/m
#         c2 = 1/(m*(1-M))*(m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
#         d2 = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
#         D20 = np.power(b2, 2)-3*a*c
#         D21 = 2*np.power(b2, 3) - 9*a2*b2*c2+27*np.power(a2, 2)*d2
#         R2 = np.sqrt(D20^3)
#         V2=-1/(3*a2)*(b2+2*np.sqrt(D20)*np.cos((1/3)*np.arccos(D21/(2*R2))))


# def calculate_epsilon_prima(M: float, m: float, eps: float):
#     #para L1
#     a1=(np.exp(eps)-1) * (M/m) * (np.power((M-m), 2))
#     b1=-((M-m)/m) * ((np.power(m,2)-4*M*m +2*M)*(np.exp(eps)-1) + np.exp(eps)*M)
#     c1= ((1-m)/m)*( (np.exp(eps)-1) * (2* np.power(m, 2)-4*M*m-m) + (3*np.exp(eps)-1)*M)
#     d1=-(1-m) *( (np.exp(eps)-1) * (m-2) + (np.exp(eps)/m) )
    
#     D10=np.power(b1,2)-3*a1*c1
#     D11= (2*np.power(b1,3)) - (9*a1*b1*c1) + (27*np.power(a1, 2)* d1)
#     R1=np.sqrt(np.power(D10, 3))
  
#     #Opciones D1
#     np.power(D11, 2)-4*np.power(D10, 3)

#     #para L2

# print(calculate_eps_prima(0.1, 0.1, 0.15))

# archivo="Files m and M\\file m=0.1 M=0.2.csv"
# numeros=extract_m_and_Monefile(archivo)
# print(numeros[1])