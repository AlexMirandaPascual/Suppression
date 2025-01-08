import pandas as pd
import numpy as np
from additional_algorithms import *
from generate_average_distance_list import *
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

def Gaussian_noise(delta, epsilon=1, sensitivity=1):
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
                        path_average_distances="Appledistances.csv",
                        path_of_file="Files m and M\\Apple\\"):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""
    
    if not os.path.exists(path_of_file):
    # If no exist, create the folder
        os.makedirs(path_of_file)

    length=len(m)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        print("m[i]= ", m[i])

        for j in range(length):
            if m[length-j-1]>=m[i]:
                print("m[j]= ", m[length-j-1])
                name_file= path_of_file +"file m=" + str(m[i]) + " M=" + str(m[length-j-1]) + ".csv"
                generate_probabilities_csv(m=m[i],M=m[length-j-1], path_distances=path_average_distances, path_probabilidades=name_file)
            else:
                break
    name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    generate_probabilities_csv(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilidades=name_file)        
#End of generate_files_m_M 

def generate_list_m_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    m_and_M=[]
    length=len(m)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        print("m[i]= ", m[i] )
        for j in range(length):
            if m[length-j-1]>=m[i]:
                m_and_M.append([m[i],m[length-j-1]])
            else:
                break
    m_and_M.append([m[-1],m[-1]])
    return m_and_M


def suppressed_dataset(probabilities, dataset, fruit_name):
    """"Given a dataset, it is reduced according to a given probability of choosing
or not each element, probabilities and dataset must be dataframe"""
    # if probabilities<0 or probabilities>1:
    #     return "La probabilidad debe estar entre 0 y 1"
    new_dataset=[]
    probabilities_df=pd.read_csv(probabilities)
    dataset_df=pd.read_csv(dataset)
    element=dataset_df[dataset_df["Fruit Name"]==fruit_name]
    element.reindex()
    vector_size=element.shape
    total_element=vector_size[0]

    for i in range(total_element-1):
        x=np.random.random(1)
        if (x<=probabilities_df.iloc[i, 0]):
            new_dataset.append(element.iloc[i])
    df_new_dataset=pd.DataFrame(new_dataset)
    return df_new_dataset


def generate_iterations_suppressed_database(path_m_M="Files m and M\\Apple\\", original_dataset_path="fruits.csv", fruit_name="Apple", numberofrepeat: int=1):
    """Generate a file containing the base of element of datasets with supression"""
    list_element=os.listdir(path_m_M)
    header=["average weight", "total_sum weight", "average volume", "total_sum volume", "total_element", "m", "M"]
    element=[[0]*7]
    df=pd.DataFrame(element, columns=header)
    folder="File_graphic\\"

    if not os.path.exists(folder):
        # If no exist, create the folder
            os.makedirs(folder)

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=path_m_M + list_element[i], dataset=original_dataset_path, fruit_name=fruit_name)
            total_sum_weight=data["Weight"].sum()
            total_sum_volume=data["Volume"].sum()
            total_element=data.shape
            average_weight=total_sum_weight/total_element[0]
            average_volume=total_sum_volume/total_element[0]
            m, M=extract_m_and_Monefile(path_m_M + list_element[i])
            element.append([average_weight , total_sum_weight, average_volume, total_sum_volume,  total_element[0], m, M])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(folder + fruit_name +"_base.csv", index=False)
    deleted_element_0(folder + fruit_name +"_base.csv")

# generate_iterations_suppressed_database(numberofrepeat=100)

def MoS_Laplace_and_Gaussian(file="Apple_base.csv", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, epsilon=1, sensitivity=1, delta=None, real_weight=100, real_volume=80):
    folder="File_graphic\\"
    df=pd.read_csv(folder + file)
    totalelement=df.shape
    average_laplacian_weight_list=[]
    average_laplacian_volume_list=[]
    average_gaussian_weight_list=[]
    average_gaussian_volume_list=[]
    L2_laplacian_list=[]
    L2_gaussian_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal_weight=new_df["total_sum weight"]
        sumtotal_volume=new_df["total_sum volume"]
        total_element=new_df["total_element"]
        if (delta==None):
            delta=np.power((1/total_element), 2)
        else:
            delta=delta
        #Laplace_noise weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element +F1(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        average_laplacian_weight_list.append(average_laplacian_weight)
        #Laplace_noise volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element +F1(epsilon1=epsilon/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        average_laplacian_volume_list.append(average_laplacian_volume)
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element + Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        average_gaussian_weight_list.append(average_gaussian_weight)
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume+ Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_gaussian_volume=total_element + Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        average_gaussian_volume_list.append(average_gaussian_volume)
        #difference L2 with weight and volume Laplace_noise
        L2_laplacian=np.linalg.norm([average_laplacian_weight-real_weight,average_laplacian_volume-real_volume])
        #difference L2 with weight and volume Laplace_noise
        L2_gaussian=np.linalg.norm([average_gaussian_weight-real_weight,average_gaussian_volume-real_volume])
        L2_laplacian_list.append(L2_laplacian)
        L2_gaussian_list.append(L2_gaussian)
    df["average_laplace weight"]=average_laplacian_weight_list
    df["average_gaussian weight"]=average_gaussian_weight_list
    df["average_laplace volume"]=average_laplacian_volume_list
    df["average_gaussian volume"]=average_gaussian_volume_list
    df["L2_laplacian_supression"]=L2_laplacian_list
    df["L2_gaussian_supression"]=L2_gaussian_list
    df.to_csv(folder + file, index=False)


def aggregateLaplaceandGaussiansuppression(file="Apple_base.csv", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, epsilon=1, sensitivity=1, delta=None, real_weight=100, real_volume=80):
    folder="File_graphic\\"
    df=pd.read_csv(folder + file)
    totalelement=df.shape
    average_laplacian_weight_list=[]
    average_gaussian_weight_list=[]
    average_laplacian_volume_list=[]
    average_gaussian_volume_list=[]
    L2_laplacian_list=[]
    L2_gaussian_list=[]
    delta_suppression_list=[]
    epsilon_suppression_list=[] 
    if delta==None:
        delta=np.power((1/totalelement[0]), 2)
    else:
        delta=delta
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal_weight=new_df["total_sum weight"]
        sumtotal_volume=new_df["total_sum volume"]
        total_element=new_df["total_element"]
        delta_suppression=calculate_delta_suppression(delta=delta , m=new_df["m"])
        epsilon_suppression=calculate_eps_suppression(m=float(new_df["m"]), M=float(new_df["M"]), eps=epsilon)
        delta_suppression_list.append(delta_suppression)
        epsilon_suppression_list.append(epsilon_suppression)
        #Laplace_noise weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element +F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        average_laplacian_weight_list.append(average_laplacian_weight)
        #Laplace_noise volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element +F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        average_laplacian_volume_list.append(average_laplacian_volume)
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        average_gaussian_weight_list.append(average_gaussian_weight)
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume+ Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume)) 
        total_element_gaussian_volume=total_element + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        average_gaussian_volume_list.append(average_gaussian_volume)
    df["delta_suppression"]=delta_suppression_list
    df["epsilon_suppression"]=epsilon_suppression_list
    df["average_laplace suppression weight"]=average_laplacian_weight_list
    df["average_gaussian suppression weight"]=average_gaussian_weight_list
    df["average_laplace suppression volume"]=average_laplacian_volume_list
    df["average_gaussian suppression volume"]=average_gaussian_volume_list
    file.replace(".csv", "")
    df.to_csv(folder + file + "_Lapl_Gauss_suppression.csv", index=False)
    
def calculateAverageofelement(file="File_graphic\\Apple_base.csv", File_name="File_graphic\\AverageApple.csv"):
    """This function selects all the elements according to M and M and calculates the average of these, example:
       m=0.1 and M=0.1 Laplace_noise average= 42; m=0.1 m=0.1 Laplace_noise average= 42; you get m=0.1 m=0.1 Laplace_noise average= 42.5
       m=0.1 and M=0.2 Gaussian average= 40; m=0.1 m=0.2 Gaussian average= 50; you get m=0.2 m=0.2 Gaussian average= 45"""
    df=pd.read_csv(file)
    header=["m", "M", "average weight", "average_laplace weight","average_gaussian weight", "average volume", "average_laplace volume","average_gaussian volume", "average_L2_laplacian_supression", "average_L2_gaussian_supression"]
    element=[[0]*10]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average_weight=average_df["average weight"]
        average_laplacian_weight=average_df["average_laplace weight"]
        average_gaussian_weight=average_df["average_gaussian weight"]
        average_volume=average_df["average volume"]
        average_laplacian_volume=average_df["average_laplace volume"]
        average_gaussian_volume=average_df["average_gaussian volume"]
        average_L2_laplacian_supression=average_df["L2_laplacian_supression"]
        average_L2_gaussian_supression=average_df["L2_gaussian_supression"]
        element.append([m, M, average_weight, average_laplacian_weight, average_gaussian_weight, average_volume, average_laplacian_volume, average_gaussian_volume, average_L2_laplacian_supression, average_L2_gaussian_supression])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(File_name, index=False)
    deleted_element_0(File_name)


def M_Laplace_and_Gaussian_change_of_parameters(path="fruits.csv", name_of_newfile="original", fruit_name="Apple", epsilon=1, delta=None, upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, sensitivity=1, real_weight=100, real_volume=80 ):
    path_of_file="File_graphic"
    if not os.path.exists(path_of_file):
    # If no exist, create the folder
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
    header=["m", "M", "delta_suppression", "epsilon_suppression", "average weight", "average_laplace weight", "average_gaussian weight", "average volume", "average_laplace volume", "average_gaussian volume", "L2_laplacian_suppression", "L2_gaussian_suppression"]
    element=[[0]*12]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m)
        print("M=", M)
        delta_suppression=calculate_delta_suppression(delta=delta , m=m)
        epsilon_suppression=calculate_eps_suppression(m=m, M=M, eps=epsilon)
    
        #Laplace_noise weight
        sumtotal_laplacian_weight=sumtotal_weight+F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight))
        total_element_laplacian_weight=total_element[0] +F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivity)
        average_laplacian_weight=sumtotal_laplacian_weight/total_element_laplacian_weight
        #Laplace_noise volume
        sumtotal_laplacian_volume=sumtotal_volume+F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume))
        total_element_laplacian_volume=total_element[0] +F1(epsilon1=epsilon_suppression/2, sensitivity=sensitivity)
        average_laplacian_volume=sumtotal_laplacian_volume/total_element_laplacian_volume
        #Gaussian weight
        sumtotal_gaussian_weight=sumtotal_weight+ Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_weight, lower=lower_weight)) 
        total_element_gaussian_weight=total_element[0] + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivity)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        #Gaussian volume
        sumtotal_gaussian_volume=sumtotal_volume + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper=upper_volume, lower=lower_volume)) 
        total_element_gaussian_volume=total_element[0] + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivity)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
         #difference L2 with weight and volume Laplace_noise
        L2_laplacian_suppression=np.linalg.norm([average_laplacian_weight-real_weight,average_laplacian_volume-real_volume])
        #difference L2 with weight and volume Gaussian
        L2_gaussian_suppression=np.linalg.norm([average_gaussian_weight-real_weight,average_gaussian_volume-real_volume]) 
        
        element.append([m, M, delta_suppression, epsilon_suppression, average_weight, average_laplacian_weight, average_gaussian_weight, average_volume, average_laplacian_volume, average_gaussian_volume, L2_laplacian_suppression, L2_gaussian_suppression])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(path_of_file +"\\" + name_of_newfile + fruit_name +" suppression.csv", index=False)
    deleted_element_0(path_of_file +"\\" + name_of_newfile + fruit_name +" suppression.csv")
    
def combining_averages(path_average_supression="File_graphic\\AverageApple.csv", path_average_suppression="File_graphic\\originalApple suppression.csv",
                       file="File_graphic\\CombiningApple.csv", real_weight=100, real_volume=80):
    average_supression=pd.read_csv(path_average_supression)
    average_suppression=pd.read_csv(path_average_suppression)
    header=["m", "M", "delta_suppression", "epsilon_suppression", "real_weight", "real_volume", "metric_laplacian", "metric_gaussian"]
    element=[[0]*8]
   
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m)
        print("M=", M)
        ave_supre=average_supression[(average_supression["m"]==m) & (average_supression["M"]==M)]
        ave_suppression=average_suppression[(average_suppression["m"]==m) & (average_suppression["M"]==M)]
        metric_laplacian=ave_suppression["L2_laplacian_suppression"]-ave_supre["average_L2_laplacian_supression"]
        metric_gaussian=ave_suppression["L2_gaussian_suppression"]-ave_supre["average_L2_gaussian_supression"]
        element.append([m, M, float(ave_suppression["delta_suppression"]), float(ave_suppression["epsilon_suppression"]), float(real_weight), float(real_volume), float(metric_laplacian), float(metric_gaussian)])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(file, index=False)
    deleted_element_0(file)



