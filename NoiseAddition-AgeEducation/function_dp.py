import pandas as pd
import numpy as np
from alexalgoritmo_funcion import *
from generate_distances import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

# satisfies 1-differential privacy
def F1(epsilon1: float=1, sensitivity=1, loc=0):
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
                        path_average_distances="Agedistances.csv",
                        path_of_file="Files m and M\\Age\\"):      
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

def generatem_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])-> list:
    "generate  values of m and M in list"
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

def prob_dataset(probabilitic, dataset, column_name):
    """"Given a dataset reduces it according to a given probability that it chooses 
    or not each element, probabilitic and dataset are where the csv file is located"""
    # if probabilitic<0 or probabilitic>1:
    #     return "La probabilidad debe estar entre 0 y 1"
    new_dataset=[]
    probabilitic_df=pd.read_csv(probabilitic)
    dataset_df=pd.read_csv(dataset)
    element=dataset_df[column_name]
    vector_size=dataset_df.shape
    total_element=vector_size[0]

    for i in range(total_element-1):
        x=np.random.random(1)
        if (x<=probabilitic_df.iloc[i, 0]):
            new_dataset.append(element[i])
    return new_dataset
# dato=prob_dataset(probabilitic="Files m and M\\Age\\file m=0.1 M=0.1.csv", dataset="irishn_train.csv", column_name="Age")

# def supression_dataset(path_of_files_m_M="Files m and M\\Age\\file m=0.1 M=0.1.csv", path_of_supression="Age_suprresed"):

def generate_supression_df(path_m_M: str="Files m and M\\Age\\", original_dataset_path: str="irishn_train.csv", column_name: str="Age", numberofrepeat: int=100):
     
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

def generate_supression_file(path_m_M="Files m and M\\Age\\", original_dataset_path="irishn_train.csv", column_name="Age", numberofrepeat: int=1):
    """Generate a file containing the base of element of datasets with supression"""
    list_element=os.listdir(path_m_M)
    header=["average", "total_sum", "total_element", "m", "M"]
    element=[[0]*5]
    
    carpet="File_graphic\\"

    if not os.path.exists(carpet):
        # If no exist, create the carpet
            os.makedirs(carpet)


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
    df.to_csv(carpet + column_name +"_onlysupression.csv", index=False)
    deleted_element_0(carpet + column_name +"_onlysupression.csv")

# generate_supression_file(numberofrepeat=100)

def agregateLaplaceandGaussian(file="Age_onlysupression.csv", upper=100, lower=0, epsilon=1, sensitivity=1, delta=None):
    """Add the elements with the Laplacian and Gaussian noise to the file"""
    carpet="File_graphic\\"
    df=pd.read_csv(carpet + file)
    totalelement=df.shape
    average_laplacian_list=[]
    average_gaussian_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal=new_df["total_sum"]
        total_element=new_df["total_element"]
        if (delta==None):
            delta=np.power((1/totalelement[0]), 2)
        else:
            delta=delta
        #Laplacian
        sumtotal_laplacian=sumtotal+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +F1(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian=sumtotal_laplacian/total_element_laplacian
        average_laplacian_list.append(average_laplacian)
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        average_gaussian_list.append(average_gaussian)
    file.replace(".csv","")
    df["average_laplacian"]=average_laplacian_list
    df["average_gaussian"]=average_gaussian_list
    df.to_csv(carpet + file, index=False)


def agregateLaplaceandGaussianPrima(file: str="Age_onlysupression.csv", upper=100, lower=0, epsilon: float=1, sensitivity=1):
    carpet="File_graphic\\"
    df=pd.read_csv(carpet + file)
    totalelement=df.shape
    average_laplacian_list=[]
    average_gaussian_list=[]
    delta_prima_list=[]
    epsilon_prima_list=[] 
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal=new_df["total_sum"]
        total_element=new_df["total_element"]
        delta_prima=calculate_delta_prima(delta=np.power((1/totalelement[0]), 2) , m=new_df["m"])
        epsilon_prima=calculate_eps_prima(m=float(new_df["m"]), M=float(new_df["M"]), eps=epsilon)
        delta_prima_list.append(delta_prima)
        epsilon_prima_list.append(epsilon_prima)
        #Laplacian
        sumtotal_laplacian=sumtotal+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian=sumtotal_laplacian/total_element_laplacian
        average_laplacian_list.append(average_laplacian)
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        average_gaussian_list.append(average_gaussian)
    df["delta_prima"]=delta_prima_list
    df["epsilon_prima"]=epsilon_prima_list
    df["average_laplacian"]=average_laplacian_list
    df["average_gaussian"]=average_gaussian_list
    df.to_csv(carpet + "Age_onlysupression_Lapl_Gauss_Prima.csv", index=False)
    
def calculateAverageofelement(file: str="File_graphic\\Age_onlysupression.csv", File_name: str="File_graphic\\AverageAge.csv"):
    """This function selects all the elements according to M and M and calculates the average of these, example:
       m=0.1 and M=0.1 laplacian average= 42; m=0.1 m=0.1 laplacian average= 42; you get m=0.1 m=0.1 laplacian average= 42.5
       m=0.1 and M=0.2 Gaussian average= 40; m=0.1 m=0.2 Gaussian average= 50; you get m=0.2 m=0.2 Gaussian average= 45"""
    df=pd.read_csv(file)
    header=["m", "M", "average", "average_laplacian", "average_gaussian"]
    element=[[0]*5]
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["average"]
        average_laplacian=average_df["average_laplacian"]
        average_gaussian=average_df["average_gaussian"]
        element.append([m, M, average, average_laplacian, average_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(File_name, index=False)
    deleted_element_0(File_name)


def agregatefileoriginalPrima(path="irishn_train.csv", name_of_newfile="original", column_name="Age", epsilon=1, delta=None, sensitivity=1, upper=100, lower=0):
    path_of_file="File_graphic"
    if not os.path.exists(path_of_file):
    # If no exist, create the carpet
        os.makedirs(path_of_file)
    
    df=pd.read_csv(path)
    sumtotal=df[column_name].sum()
    average= df[column_name].mean()
    total_element=df[column_name].size
    if delta==None:
        delta=np.power((1/total_element), 2)
    else:
        delta=delta
    header=["m", "M", "delta prima", "epsilon prima", "average", "average_laplacian", "average_gaussian"]
    element=[[0]*7]
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m,)
        print("M=", M)
        delta_prima=calculate_delta_prima(delta=delta , m=m)
        epsilon_prima=calculate_eps_prima(m=m, M=M, eps=epsilon)
    
        #Laplacian
        sumtotal_laplacian=sumtotal+F1(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +F1(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian=sumtotal_laplacian/total_element_laplacian
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        element.append([m, M, delta_prima, epsilon_prima, average, average_laplacian, average_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(path_of_file +"\\" + name_of_newfile + column_name +" prima.csv", index=False)
    deleted_element_0(path_of_file +"\\" + name_of_newfile + column_name +" prima.csv")
    
def combining_averages(path_average_supression="File_graphic\\AverageAge.csv", path_average_prima="File_graphic\\originalAge prima.csv",
                       file="File_graphic\\CombiningAge.csv"):
    average_supression=pd.read_csv(path_average_supression)
    average_prima=pd.read_csv(path_average_prima)
    header=["m", "M", "delta prima", "epsilon prima", "average", "metric_laplacian", "metric_gaussian"]
    element=[[0]*7]
    
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m)
        print("M=", M)
        ave_supre=average_supression[(average_supression["m"]==m) & (average_supression["M"]==M)]
        ave_prima=average_prima[(average_prima["m"]==m) & (average_prima["M"]==M)]
        metric_laplacian=np.abs(ave_prima["average_laplacian"]-ave_prima["average"]) - np.abs(ave_supre["average_laplacian"]-ave_prima["average"])
        metric_gaussian=np.abs(ave_prima["average_gaussian"]-ave_prima["average"])- np.abs(ave_supre["average_gaussian"]-ave_prima["average"])
        element.append([m, M, float(ave_prima["delta prima"]), float(ave_prima["epsilon prima"]), float(ave_prima["average"]) , float(metric_laplacian), float(metric_gaussian)])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(file, index=False)
    deleted_element_0(file)

# combining_averages()

# def generateFileandGraphHighestEducationCompleteddistances():
#     filedistanceL2(filename="irishn_train.csv", column="HighestEducationCompleted")
#     generate_files_m_M(m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         path_average_distances = "HighestEducationCompleteddistances.csv",
#         path_of_file = "Files m and M\\HighestEducationCompleted\\")
#     generate_supression_file(path_m_M="Files m and M\\HighestEducationCompleted\\", column_name="HighestEducationCompleted", numberofrepeat=100)
#     agregateLaplaceandGaussian(file="HighestEducationCompleted_onlysupression.csv", upper=10, lower=1)
#     agregatefileoriginalPrima(column_name="HighestEducationCompleted", upper = 10, lower = 1)
#     calculateAverageofelement(file="File_graphic\\HighestEducationCompleted_onlysupression.csv", File_name= "File_graphic\\AverageHighestEducationCompleted.csv")
#     combining_averages(path_average_supression = "File_graphic\\AverageHighestEducationCompleted.csv",
#         path_average_prima = "File_graphic\\originalHighestEducationCompleted prima.csv",
#         file= "File_graphic\\CombiningHighestEducationCompleted.csv")
#     generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="gaussian")
#     generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="laplacian")
    
# def generateFileandGraphHighestEducationCompleteddistances():
#     filedistanceL2()
#     generate_files_m_M()
#     generate_supression_file(numberofrepeat=100)
#     agregateLaplaceandGaussian()
#     agregatefileoriginalPrima()
#     calculateAverageofelement()
#     combining_averages()