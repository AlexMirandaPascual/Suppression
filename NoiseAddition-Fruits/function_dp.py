import pandas as pd
import numpy as np
from alexalgoritmo_funcion import *
from generate_distances import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

# satisfies 1-differential privacy
def F1(epsilon1=1, sensitivity=1, loc=0):
    return np.random.laplace(loc, scale=sensitivity/epsilon1)
# satisfies 1-differential privacy
def F2(epsilon2 = 1, sensitivity=1, loc=0):
    return np.random.laplace(loc, scale=sensitivity/epsilon2)
# satisfies 2-differential privacy
def F3(total_epsilon = 2, sensitivity=1, loc=0):
    return np.random.laplace(loc, scale=sensitivity/total_epsilon)
# satisfies 2-differential privacy, by sequential composition
def F_combined(epsilon1 = 1, epsilon2 = 1, sensitivity=1, loc=0):
    return (F1(epsilon1=epsilon1, sensitivity=sensitivity, loc=loc) + F2(epsilon2=epsilon2, sensitivity=1, loc=loc)) / 2

def Gaussian_p(delta, epsilon=1, sensitivity=1):
     sigma=(2*np.power(sensitivity,2)*np.log(1.25/delta))/(np.power(epsilon, 2))
    #  sigma_elevate2=np.power(sigma, 2)
     sigma=np.sqrt(sigma)
     gauss=np.random.normal(0, sigma)
     return gauss
def sensitivityAverage(upper, lower):
    return np.abs(upper - lower)

def sensitivitySummation(upper, lower):
    return np.abs(upper - lower)


# filename="irishn_train.csv"

# df=pd.read_csv(filename)
# Age=df["HighestEducationCompleted"]
# max_valor_age=Age.max()
# Age_norm=Age/max_valor_age
# # print(Age_norm)
# a=np.linalg.norm([1,0.2, 2])
# # print(np.sqrt(1**1+ 0.2**2+2**2))
# cuadrado=Age_norm[0::]*Age_norm[0::]
# # print(cuadrado)

# dist=np.linalg.norm(Age_norm[0::], Age_norm[0])
# print(dist)
# promdistancias=[]

# countable_elements=len(Age_norm)
# for i in range(countable_elements): 
#     suma_total=0
#     for j in range(countable_elements): 
#         dist=np.linalg.norm([Age_norm[i], Age_norm[j]])
#         suma_total=suma_total + dist
#     promedio=suma_total/(countable_elements)
#     promdistancias.append(promedio)
# dataframenumpy=np.array(promdistancias)
# # dataframe.to_csv(path="algo.csv", header=["Average distances"], index=None)
# np.savetxt("HighestEducationCompleted.csv", dataframenumpy)
 
def generate_files_m_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        path_average_distances="Appledistances.csv",
                        path_of_file="Files m and M\\Apple\\"):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""
    
    if not os.path.exists(path_of_file):
    # If no exist, create the carpet
        os.makedirs(path_of_file)

    longitud=len(m)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        print("m[i]= ", m[i] )

        for j in range(longitud):
            if m[longitud-j-1]>=m[i]:
                print("m[j]= ", m[longitud-j-1])
                name_file= path_of_file +"file m=" + str(m[i]) + " M=" + str(m[longitud-j-1]) + ".csv"
                generar_cvs_probabilidad(m=m[i],M=m[longitud-j-1], path_distances=path_average_distances, path_probabilidades=name_file)
            else:
                break
    name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    generar_cvs_probabilidad(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilidades=name_file)        
#End of generate_files_m_M 

def generatem_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    m_and_M=[]
    longitud=len(m)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        print("m[i]= ", m[i] )
        for j in range(longitud):
            if m[longitud-j-1]>=m[i]:
                m_and_M.append([m[i],m[longitud-j-1]])
            else:
                break
    m_and_M.append([m[-1],m[-1]])
    return m_and_M
# generate_files_m_M()
# generate_files_m_M(path_average_distances="HighestEducationCompleteddistances.csv",
#                         path_of_file="Files m and M\\HighestEducationCompleted\\")

def prob_dataset(probabilitic, dataset, fruit_name):
    """"Dado un dataset lo reduce segun una probabilidad dada de que escoja 
    o no cada elemento, probabilitic y dataset deben ser dataframe"""
    # if probabilitic<0 or probabilitic>1:
    #     return "La probabilidad debe estar entre 0 y 1"
    new_dataset=[]
    probabilitic_df=pd.read_csv(probabilitic)
    dataset_df=pd.read_csv(dataset)
    element=dataset_df[dataset_df["Fruit Name"]==fruit_name]
    element.reindex()
    vector_size=element.shape
    total_element=vector_size[0]

    for i in range(total_element-1):
        x=np.random.random(1)
        if (x<=probabilitic_df.iloc[i, 0]):
            new_dataset.append(element.iloc[i])
    df_new_dataset=pd.DataFrame(new_dataset)
    return df_new_dataset
# dato=prob_dataset(probabilitic="Files m and M\\Age\\file m=0.1 M=0.1.csv", dataset="irishn_train.csv", column_name="Age")

# def supression_dataset(path_of_files_m_M="Files m and M\\Age\\file m=0.1 M=0.1.csv", path_of_supression="Age_suprresed"):

def generate_supression_df(path_m_M="Files m and M\\Apple\\", original_dataset_path="fruits.csv", column_name="Age", numberofrepeat: int=10):
     
    list_element=os.listdir("Files m and M\\Age\\")
    header=["average", "total_sum", "total_element", "m", "M"]
    column_name="Age"
    element=[[0]*5]

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=prob_dataset(probabilitic=path_m_M + list_element[i], dataset=original_dataset_path, column_name=column_name)
            data=np.array(data)
            total_sum=data.sum()
            total_elemnt=data.size
            average=total_sum/total_elemnt
            m, M=extract_m_and_Monefile(path_m_M + list_element[i])
            element.append([average, total_sum, total_elemnt, m, M])  
    df=pd.DataFrame(element, columns=header)
    return df

def generate_supression_file(path_m_M="Files m and M\\Apple\\", original_dataset_path="fruits.csv", fruit_name="Apple", numberofrepeat: int=1):
    """Generate a file containing the base of element of datasets with supression"""
    list_element=os.listdir(path_m_M)
    header=["average weight", "total_sum weight", "average volume", "total_sum volume", "total_element", "m", "M"]
    element=[[0]*7]
    
    carpet="File_graphic\\"

    if not os.path.exists(carpet):
        # If no exist, create the carpet
            os.makedirs(carpet)


    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=prob_dataset(probabilitic=path_m_M + list_element[i], dataset=original_dataset_path, fruit_name=fruit_name)
            total_sum_weight=data["Weight"].sum()
            total_sum_volume=data["Volume"].sum()
            total_elemnt=data.shape
            average_weight=total_sum_weight/total_elemnt[0]
            average_volume=total_sum_volume/total_elemnt[0]
            m, M=extract_m_and_Monefile(path_m_M + list_element[i])
            element.append([average_weight , total_sum_weight, average_volume, total_sum_volume,  total_elemnt[0], m, M])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(carpet + fruit_name +"_onlysupression.csv", index=False)
    deleted_element_0(carpet + fruit_name +"_onlysupression.csv")

# generate_supression_file(numberofrepeat=100)

def agregateLaplaceandGaussian(file="Apple_onlysupression.csv", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, epsilon=1, sensitivity=1, delta=None):
    carpet="File_graphic\\"
    df=pd.read_csv(carpet + file)
    totalelement=df.shape
    average_laplacian_weight_list=[]
    average_laplacian_volume_list=[]
    average_gaussian_weight_list=[]
    average_gaussian_volume_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal_weight=new_df["total_sum weight"]
        sumtotal_volume=new_df["total_sum volume"]
        total_element=new_df["total_element"]
        if (delta==None):
            delta=np.power((1/total_element), 2)
        else:
            delta=delta
        #Laplacian weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element +F1(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        average_laplacian_weight_list.append(average_laplacian_weight)
        #Laplacian volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element +F1(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        average_laplacian_volume_list.append(average_laplacian_volume)
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element + Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        average_gaussian_weight_list.append(average_gaussian_weight)
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume+ Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_gaussian_volume=total_element + Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        average_gaussian_volume_list.append(average_gaussian_volume)
    df["average_laplacian weight"]=average_laplacian_weight_list
    df["average_gaussian weight"]=average_gaussian_weight_list
    df["average_laplacian volume"]=average_laplacian_volume_list
    df["average_gaussian volume"]=average_gaussian_volume_list
    df.to_csv(carpet + file, index=False)


def agregateLaplaceandGaussianPrima(file="Apple_onlysupression.csv", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, epsilon=1, sensitivity=1):
    carpet="File_graphic\\"
    df=pd.read_csv(carpet + file)
    totalelement=df.shape
    average_laplacian_weight_list=[]
    average_gaussian_weight_list=[]
    average_laplacian_volume_list=[]
    average_gaussian_volume_list=[]
    delta_prima_list=[]
    epsilon_prima_list=[] 
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal_weight=new_df["total_sum weight"]
        sumtotal_volume=new_df["total_sum volume"]
        total_element=new_df["total_element"]
        delta_prima=calculate_delta_prima(delta=np.power((1/totalelement[0]), 2) , m=new_df["m"])
        epsilon_prima=calculate_eps_prima(m=float(new_df["m"]), M=float(new_df["M"]), eps=epsilon)
        delta_prima_list.append(delta_prima)
        epsilon_prima_list.append(epsilon_prima)
        #Laplacian weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        average_laplacian_weight_list.append(average_laplacian_weight)
        #Laplacian volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        average_laplacian_volume_list.append(average_laplacian_volume)
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        average_gaussian_weight_list.append(average_gaussian_weight)
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume)) 
        total_element_gaussian_volume=total_element + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        average_gaussian_volume_list.append(average_gaussian_volume)
    df["delta_prima"]=delta_prima_list
    df["epsilon_prima"]=epsilon_prima_list
    df["average_laplacian prima weight"]=average_laplacian_weight_list
    df["average_gaussian prima weight"]=average_gaussian_weight_list
    df["average_laplacian prima volume"]=average_laplacian_volume_list
    df["average_gaussian prima volume"]=average_gaussian_volume_list
    file.replace(".csv", "")
    df.to_csv(carpet + file + "_Lapl_Gauss_Prima.csv", index=False)
    
def calculateAverageofelement(file="File_graphic\\Apple_onlysupression.csv", File_name="File_graphic\\AverageApple.csv"):
    """This function selects all the elements according to M and M and calculates the average of these, example:
       m=0.1 and M=0.1 laplacian average= 42; m=0.1 m=0.1 laplacian average= 42; you get m=0.1 m=0.1 laplacian average= 42.5
       m=0.1 and M=0.2 Gaussian average= 40; m=0.1 m=0.2 Gaussian average= 50; you get m=0.2 m=0.2 Gaussian average= 45"""
    df=pd.read_csv(file)
    header=["m", "M", "average weight", "average_laplacian weight","average_gaussian weight", "average volume", "average_laplacian volume","average_gaussian volume"]
    element=[[0]*8]
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average_weight=average_df["average weight"]
        average_laplacian_weight=average_df["average_laplacian weight"]
        average_gaussian_weight=average_df["average_gaussian weight"]
        average_volume=average_df["average volume"]
        average_laplacian_volume=average_df["average_laplacian volume"]
        average_gaussian_volume=average_df["average_gaussian volume"]
        element.append([m, M, average_weight, average_laplacian_weight, average_gaussian_weight, average_volume, average_laplacian_volume, average_gaussian_volume])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(File_name, index=False)
    deleted_element_0(File_name)


def agregatefileoriginalPrima(path="fruits.csv", name_of_newfile="original", fruit_name="Apple", epsilon=1, delta=None, upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, sensitivity=1):
    path_of_file="File_graphic"
    if not os.path.exists(path_of_file):
    # If no exist, create the carpet
        os.makedirs(path_of_file)

    df=pd.read_csv(path)
    df_fruit=df[df["Fruit Name"]==fruit_name]
    sumtotal_weight=df_fruit["Weight"].sum()
    average_weight= df_fruit["Weight"].mean()
    sumtotal_volume=df_fruit["Volume"].sum()
    average_volume= df_fruit["Volume"].mean()
    total_element=df_fruit.shape
    if delta==None:
        delta=np.power((1/total_element[0]), 2)
    else:
        delta=delta
    header=["m", "M", "delta prima", "epsilon prima", "average weight", "average_laplacian weight", "average_gaussian weight", "average volume", "average_laplacian volume", "average_gaussian volume" ]
    element=[[0]*10]
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m,)
        print("M=", M)
        delta_prima=calculate_delta_prima(delta=delta , m=m)
        epsilon_prima=calculate_eps_prima(m=m, M=M, eps=epsilon)
    
        #Laplacian weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element[0] +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        #Laplacian volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element[0] +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element[0] + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume)) 
        total_element_gaussian_volume=total_element[0] + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
    
        element.append([m, M, delta_prima, epsilon_prima, average_weight, average_laplacian_weight, average_gaussian_weight, average_volume, average_laplacian_volume, average_gaussian_volume])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(path_of_file +"\\" + name_of_newfile + fruit_name +" prima.csv", index=False)
    deleted_element_0(path_of_file +"\\" + name_of_newfile + fruit_name +" prima.csv")
    
def combining_averages(path_average_supression="File_graphic\\AverageApple.csv", path_average_prima="File_graphic\\originalApple prima.csv",
                       file="File_graphic\\CombiningApple.csv", real_weight=100, real_volume=80):
    average_supression=pd.read_csv(path_average_supression)
    average_prima=pd.read_csv(path_average_prima)
    header=["m", "M", "delta prima", "epsilon prima", "real_weight", "real_volume", "metric_laplacian", "metric_gaussian"]
    element=[[0]*8]
   
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m)
        print("M=", M)
        ave_supre=average_supression[(average_supression["m"]==m) & (average_supression["M"]==M)]
        ave_prima=average_prima[(average_prima["m"]==m) & (average_prima["M"]==M)]
        metric_laplacian=np.linalg.norm([ave_prima["average_laplacian weight"]-real_weight,ave_prima["average_laplacian volume"]-real_volume] )-np.linalg.norm([ave_supre["average_laplacian weight"]-real_weight, ave_supre["average_laplacian volume"]-real_volume])
        metric_gaussian=np.linalg.norm([ave_prima["average_gaussian weight"]-real_weight,ave_prima["average_gaussian volume"]-real_volume] )-np.linalg.norm([ave_supre["average_gaussian weight"]-real_weight, ave_supre["average_gaussian volume"]-real_volume])
        element.append([m, M, float(ave_prima["delta prima"]), float(ave_prima["epsilon prima"]), float(real_weight), float(real_volume), float(metric_laplacian), float(metric_gaussian)])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(file, index=False)
    deleted_element_0(file)

# combining_averages()

