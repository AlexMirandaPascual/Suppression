import pandas as pd
import numpy as np
from alexalgoritmo_funcion import *
from generate_distances import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

# satisfies 1-differential privacy
def Laplacian(epsilon1: float=1, sensitivity=1, loc=0):
    return np.random.laplace(loc, scale=sensitivity/epsilon1)


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
                generate_probabilities_cvs(m=m[i],M=m[longitud-j-1], path_distances=path_average_distances, path_probabilidades=name_file)
            else:
                break
    name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    generate_probabilities_cvs(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilidades=name_file)        
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

def prob_dataset(probabilitic, dataset, column_name):
    """"Given a dataset reduces it according to a given probability that it chooses 
    or not each element, probabilitic and dataset are where the csv file is located"""
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

#This function is useful, as it creates a dataframe, perhaps someone can take advantage of it
def generate_supression_df(path_m_M: str="Files m and M\\Age\\", original_dataset_path: str="irishn_train.csv", column_name: str="Age", numberofrepeat: int=100):
     
    list_element=os.listdir(path_m_M)
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
    header=["original average", "supression average", "total_sum", "total_element", "m", "M"]
    element=[[0]*6]
    original_data=pd.read_csv(original_dataset_path)
    original_average=float(original_data[column_name].mean())
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
            element.append([original_average, average, total_sum, total_elemnt, m, M])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(carpet + column_name +"_onlysupression.csv", index=False)
    deleted_element_0(carpet + column_name +"_onlysupression.csv")


def agregateLaplaceandGaussian(file="Age_onlysupression.csv", upper=100, lower=0, epsilon=1, sensitivity=1, delta=None):
    """Add the elements with the Laplacian and Gaussian noise to the file"""
    carpet="File_graphic\\"
    df=pd.read_csv(carpet + file)
    totalelement=df.shape
    average_laplacian_list=[]
    average_gaussian_list=[]
    difference_laplacian_list=[]
    difference_gaussian_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal=new_df["total_sum"]
        total_element=new_df["total_element"]
        original_average=new_df["original average"]
        if (delta==None):
            delta=np.power((1/totalelement[0]), 2)
        else:
            delta=delta
        #Laplacian
        sumtotal_laplacian=sumtotal+Laplacian(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +Laplacian(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian=sumtotal_laplacian/total_element_laplacian
         #Difference between laplacian and average real
        difference_laplacian_supression=np.abs(original_average-average_laplacian)
        
        average_laplacian_list.append(average_laplacian)
        difference_laplacian_list.append(difference_laplacian_supression)
        
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        #Difference between gaussian and average real
        difference_gaussian_supression=np.abs(original_average-average_gaussian)

        average_gaussian_list.append(average_gaussian)
        difference_gaussian_list.append(difference_gaussian_supression)

    file.replace(".csv","")
    df["average_laplacian"]=average_laplacian_list
    df["average_gaussian"]=average_gaussian_list
    df["difference_laplacian_supression"]=difference_laplacian_list
    df["difference_gaussian_supression"]=difference_gaussian_list
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
        sumtotal_laplacian=sumtotal+Laplacian(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +Laplacian(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
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
    header=["m", "M", "original average", "difference_laplacian_supression", "difference_gaussian_supression"]
    element=[[0]*5]
    myM=generatem_M()
    for i in range(len(myM)):
        # print(myM[i])
        m=myM[i][0]
        M=myM[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["original average"]
        average_difference_laplacian=average_df["difference_laplacian_supression"]
        average_difference_gaussian=average_df["difference_gaussian_supression"]
        element.append([m, M, average, average_difference_laplacian, average_difference_gaussian])
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
    header=["m", "M", "delta prima", "epsilon prima", "average", "average_laplacian", "average_gaussian", "difference_laplacian_prima", "difference_gaussian_prima"]
    element=[[0]*9]
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
        sumtotal_laplacian=sumtotal+Laplacian(epsilon1=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplacian=total_element +Laplacian(epsilon1=epsilon_prima/2, sensitivity=sensitivity)
        average_laplacian=sumtotal_laplacian/total_element_laplacian
        difference_laplacian_prima=np.abs((sumtotal/total_element)-average_laplacian)
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_p(delta=delta_prima, epsilon=epsilon_prima/2, sensitivity=sensitivity)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        difference_gaussian_prima=np.abs((sumtotal/total_element)-average_gaussian)

        element.append([m, M, delta_prima, epsilon_prima, average, average_laplacian, average_gaussian, difference_laplacian_prima, difference_gaussian_prima])
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
        metric_laplacian=ave_prima["difference_laplacian_prima"] - ave_supre["difference_laplacian_supression"]
        metric_gaussian= ave_prima["difference_gaussian_prima"]- ave_supre["difference_gaussian_supression"]
        element.append([m, M, float(ave_prima["delta prima"]), float(ave_prima["epsilon prima"]), float(ave_prima["average"]) , float(metric_laplacian), float(metric_gaussian)])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(file, index=False)
    deleted_element_0(file)

