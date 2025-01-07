import pandas as pd
import numpy as np
from additional_algorithms import *
from generate_average_distance_list import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

# satisfies 1-differential privacy
def Laplace_noise(epsilon1: float=1, sensitivity=1, loc=0):
    return np.random.laplace(loc, scale=sensitivity/epsilon1)


def Gaussian_noise(delta, epsilon=1, sensitivity=1):
    variance=(2*np.power(sensitivity,2)*np.log(1.25/delta))/(np.power(epsilon, 2))
    sd=np.sqrt(variance)
    noise=np.random.normal(0, sd)
    return noise

def sensitivitySummation(upper, lower):
    return np.abs(upper - lower)


def generate_files_m_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        path_average_distances="Appledistances.csv",
                        path_of_file="Files m and M\\Apple\\"):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""
    
    # If it does not exist, create the folder
    if not os.path.exists(path_of_file):
        os.makedirs(path_of_file)

    length=len(m)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        #print("m[i]= ", m[i] )

        for j in range(length):
            if m[length-j-1]>=m[i]:
                #print("m[j]= ", m[length-j-1])
                name_file= path_of_file +"file m=" + str(m[i]) + " M=" + str(m[length-j-1]) + ".csv"
                generate_probabilities_csv(m=m[i],M=m[length-j-1], path_distances=path_average_distances, path_probabilidades=name_file)
            else:
                break
    name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    generate_probabilities_csv(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilidades=name_file)
    for i in range(len(m)):
        if m[i] ==m[-1]:
            break
        #print("m[i]= ", m[i])

        for j in range(length):
            if m[length-j-1]>=m[i]:
                #print("m[j]= ", m[length-j-1])
                name_file= path_of_file +"file m=" + str(m[i]/10) + " M=" + str(m[length-j-1]/10) + ".csv"
                generate_probabilities_csv(m=m[i]/10,M=m[length-j-1]/10, path_distances=path_average_distances, path_probabilidades=name_file)
            else:
                break
    name_file= path_of_file +"file m=" + str(m[-1]/10) + " M=" + str(m[-1]/10) + ".csv"
    generate_probabilities_csv(m=m[-1]/10, M=m[-1]/10, path_distances=path_average_distances, path_probabilidades=name_file)        
#End of generate_files_m_M 

def generate_list_m_M(m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])-> list:
    "generate values of m and M in list"
    m_and_M=[]
    length=len(m)
    for i in range(len(m)):
        if m[i] == m[-1]:
            break
        #print("m[i]= ", m[i] )
        for j in range(length):
            if m[length-j-1]>=m[i]:
                m_and_M.append([m[i],m[length-j-1]])
            else:
                break
    m_and_M.append([m[-1],m[-1]]) #Append m=M=last element of list
    for i in range(len(m)):
        if m[i] == m[-1]:
            break
        #print("m[i]= ", m[i] )
        for j in range(length):
            if m[length-j-1]>=m[i]:
                m_and_M.append([m[i]/10,m[length-j-1]/10])
            else:
                break
    m_and_M.append([m[-1]/10,m[-1]/10]) #Append m=M=last element of list
    return m_and_M

def suppressed_dataset(probabilities, dataset, fruit_name):
    """"Given a dataset, it suppresses elements according to the outlier scores in the probabilities file"""
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

def generate_iterations_suppressed_database(path_m_M="Files m and M\\Apple\\", original_dataset_path="fruits.csv", fruit_name="Apple", numberofrepeat: int=500):
    """Generate a file containing the base statistics the of datasets with supression, repeated a number of times"""
    list_element=os.listdir(path_m_M)
    header=[ "m", "M","original_average_weight","original_average_volume","supressed_database_average_weight", "supressed_database_sum_weight","supressed_database_average_volume", "supressed_database_sum_volume", "supressed_database_number_of_elements"]
    element=[[0]*9]
    original_dataset=pd.read_csv(original_dataset_path)
    original_average_weight=float(original_dataset["Weight"].mean())
    original_average_volume=float(original_dataset["Volume"].mean())
    folder="File_graphic\\"

    df=pd.DataFrame(element, columns=header)

    # If folder does not exist, create the folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=path_m_M + list_element[i], dataset=original_dataset_path, fruit_name=fruit_name)

            m, M=extract_m_and_Monefile(path_m_M + list_element[i])


            supressed_database_sum_weight=data["Weight"].sum()
            supressed_database_sum_volume=data["Volume"].sum()
            supressed_database_number_of_elements=data.shape
            supressed_database_average_weight=supressed_database_sum_weight/supressed_database_number_of_elements[0]
            supressed_database_average_volume=supressed_database_sum_volume/supressed_database_number_of_elements[0]

            element.append([ m, M, original_average_weight, original_average_volume, supressed_database_average_weight, supressed_database_sum_weight, supressed_database_average_volume, supressed_database_sum_volume, supressed_database_number_of_elements[0]])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(folder + fruit_name +"_base.csv", index=False)
    deleted_element_0(folder + fruit_name +"_base.csv")

"""Compute MoS for Laplace and Gaussian"""
def MoS_Laplace_and_Gaussian(file="Apple_base.csv", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, epsilon=1, delta=None, real_weight=100, real_volume=80):
    folder="File_graphic\\"
    df=pd.read_csv(folder + file)
    totalelement=df.shape
    average_laplace_weight_list=[]
    average_gaussian_weight_list=[]
    difference_laplace_weight_list=[]
    difference_gaussian_weight_list=[]
    difference_realmean_laplace_weight_list=[]
    difference_realmean_gaussian_weight_list=[]
    average_laplace_volume_list=[]
    average_gaussian_volume_list=[]
    difference_laplace_volume_list=[]
    difference_gaussian_volume_list=[]
    difference_realmean_laplace_volume_list=[]
    difference_realmean_gaussian_volume_list=[]
    L2_laplace_list=[]
    L2_gaussian_list=[]
    L2_realvalue_laplace_list=[]
    L2_realvalue_gaussian_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        supressed_database_sum_weight=new_df["supressed_database_sum_weight"]
        supressed_database_sum_volume=new_df["supressed_database_sum_volume"]
        supressed_database_number_of_elements=new_df["supressed_database_number_of_elements"]
        original_average_weight=new_df["original_average_weight"]
        original_average_volume=new_df["original_average_volume"]
        if (delta==None):
            delta=np.power((1/supressed_database_number_of_elements), 2)
        else:
            delta=delta
        #Laplace weight
        sumtotal_laplace_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
        total_element_laplace_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
        average_laplace_weight=sumtotal_laplace_weight/total_element_laplace_weight
        #Difference between Laplace_noise and averages
        difference_laplace_supression_weight=np.abs(original_average_weight-average_laplace_weight)
        difference_realmean_laplace_supression_weight=np.abs(real_weight-average_laplace_weight)

        #Laplace volume
        sumtotal_laplace_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
        total_element_laplace_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
        average_laplace_volume=sumtotal_laplace_volume/total_element_laplace_volume
        #Difference between Laplace_noise and average real
        difference_laplace_supression_volume=np.abs(original_average_volume-average_laplace_volume)
        difference_realmean_laplace_supression_volume=np.abs(real_volume-average_laplace_volume)
        
        average_laplace_weight_list.append(average_laplace_weight)
        difference_laplace_weight_list.append(difference_laplace_supression_weight)
        difference_realmean_laplace_weight_list.append(difference_realmean_laplace_weight_list)
        average_laplace_volume_list.append(average_laplace_volume)
        difference_laplace_volume_list.append(difference_laplace_supression_volume)
        difference_realmean_laplace_volume_list.append(difference_realmean_laplace_volume_list)
        
        #Gaussian weight
        sumtotal_gaussian_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
        total_element_gaussian_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        #Difference between Laplace_noise and averages
        difference_gaussian_supression_weight=np.abs(original_average_weight-average_gaussian_weight)
        difference_realmean_gaussian_supression_weight=np.abs(real_weight-average_gaussian_weight)

        #Gaussian volume
        sumtotal_gaussian_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
        total_element_gaussian_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        #Difference between Gaussian_noise and average real
        difference_gaussian_supression_volume=np.abs(original_average_volume-average_gaussian_volume)
        difference_realmean_gaussian_supression_volume=np.abs(real_volume-average_gaussian_volume)
        
        average_gaussian_weight_list.append(average_gaussian_weight)
        difference_gaussian_weight_list.append(difference_laplace_gaussian_weight)
        difference_realmean_gaussian_weight_list.append(difference_realmean_gaussian_weight_list)
        average_gaussian_volume_list.append(average_gaussian_volume)
        difference_gaussian_volume_list.append(difference_gaussian_supression_volume)
        difference_realmean_gaussian_volume_list.append(difference_realmean_gaussian_volume_list)

        #difference L2 with weight and volume Laplace
        L2_laplace=np.linalg.norm([difference_laplace_supression_weight,difference_laplace_supression_volume])
        L2_realvalue_laplace=np.linalg.norm([difference_realvalue_laplace_supression_weight,difference_realvalue_laplace_supression_volume])
        #difference L2 with weight and volume Laplace
        L2_gaussian=np.linalg.norm([difference_gaussian_supression_weight,difference_gaussian_supression_volume])
        L2_realvalue_gaussian=np.linalg.norm([difference_realvalue_gaussian_supression_weight,difference_realvalue_gaussian_supression_volume])
        
        L2_laplace_list.append(L2_laplace)
        L2_realvalue_laplace_list.append(L2_realvalue_laplace)
        L2_gaussian_list.append(L2_gaussian)
        L2_realvalue_gaussian_list.append(L2_realvalue_gaussian)
    file_mod=file.replace(".csv"," ")
    new_file_to_replace = folder + file_mod + "eps=" + str(epsilon) + " delta=" + str(delta) + " MoS.csv"
    df["average_laplace_weight"]=average_laplace_weight_list
    df["average_gaussian_weight"]=average_gaussian_weight_list
    df["difference_laplace_weight"]=difference_laplace_weight_list
    df["difference_gaussian_weight"]=difference_gaussian_weight_list
    df["difference_realmean_laplace_weight"]=difference_realmean_laplace_weight_list
    df["difference_realmean_gaussian_weight"]=difference_realmean_gaussian_weight_list
    df["average_laplace_volume"]=average_laplace_volume_list
    df["average_gaussian_volume"]=average_gaussian_volume_list
    df["difference_laplace_volume"]=difference_laplace_volume_list
    df["difference_gaussian_volume"]=difference_gaussian_volume_list
    df["difference_realmean_laplace_volume"]=difference_realmean_laplace_volume_list
    df["difference_realmean_gaussian_volume"]=difference_realmean_gaussian_volume_list
    df["L2_laplace"]=L2_laplace_list
    df["L2_gaussian"]=L2_gaussian_list
    df["L2_realvalue_laplace"]=L2_realvalue_laplace_list
    df["L2_realvalue_gaussian"]=L2_realvalue_gaussian_list

    df.to_csv(new_file_to_replace, index=False)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "original_average_weight","original_average_volume", "L2_laplace", "L2_gaussian","L2_realvalue_laplace","L2_realvalue_gaussian"]
    element=[[0]*8]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average_weight=average_df["original_average_weight"]
        average_volume=average_df["original_average_volume"]
        average_L2_laplace=average_df["L2_laplace"]
        average_L2_gaussian=average_df["L2_gaussian"]
        average_L2_realvalue_laplace=average_df["L2_realvalue_laplace"]
        average_L2_realvalue_gaussian=average_df["L2_realvalue_gaussian"]
        element.append([m, M, average_weight, average_volume, average_L2_laplace, average_L2_gaussian, average_L2_realvalue_laplace, average_L2_realvalue_gaussian])
    new_df=pd.DataFrame(element, columns=header2)
    new_file_to_replace = folder + file + "eps=" + str(epsilon) + "delta=" + str(delta) + "MoS_Average.csv"
    new_df.to_csv(new_file_to_replace, index=False)
    deleted_element_0(new_file_to_replace)

"""Compute M with epsilon^S and delta^S for Laplace and Gaussian"""
def M_Laplace_and_Gaussian_change_of_parameters(path="fruits.csv", name_of_newfile="original", fruit_name="Apple", epsilon=1, delta=None, upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, numberofrepeat: int=500):
    path_of_file="File_graphic"
    if not os.path.exists(path_of_file):
    # If it does not exist, create the folder
        os.makedirs(path_of_file)
    
    df=pd.read_csv(path)
    df_fruit=df[df["Fruit Name"]==fruit_name]
    sumtotal_weight=df["Weight"].sum()
    sumtotal_volume=df["Volume"].sum()
    average_weight= df["Weight"].mean()
    average_volume= df["Volume"].mean()
    total_element=df_fruit.shape
    if delta==None:
        delta=np.power((1/total_element[0]), 2)
    else:
        delta=delta
    header=["m", "M", "delta_suppression", "epsilon_suppression", "original_average_weight","original_average_volume", "L2_laplace", "L2_gaussian","L2_realvalue_laplace","L2_realvalue_gaussian"]
    element=[[0]*9]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        epsilon_orig = epsilon
        delta_orig = delta
        delta=calculate_delta_suppression(delta=delta_orig, m=m)
        epsilon=calculate_eps_suppression(m=m, M=M, eps=epsilon_orig)
    
        for iteration in range(numberofrepeat):
            #Laplace weight
            sumtotal_laplace_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_laplace_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_laplace_weight=sumtotal_laplace_weight/total_element_laplace_weight
            #Difference between Laplace_noise and averages
            difference_laplace_supression_weight=np.abs(original_average_weight-average_laplace_weight)
            difference_realmean_laplace_supression_weight=np.abs(real_weight-average_laplace_weight)

            #Laplace volume
            sumtotal_laplace_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_laplace_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_laplace_volume=sumtotal_laplace_volume/total_element_laplace_volume
            #Difference between Laplace_noise and average real
            difference_laplace_supression_volume=np.abs(original_average_volume-average_laplace_volume)
            difference_realmean_laplace_supression_volume=np.abs(real_volume-average_laplace_volume)
            
            #Gaussian weight
            sumtotal_gaussian_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_gaussian_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
            #Difference between Laplace_noise and averages
            difference_gaussian_supression_weight=np.abs(original_average_weight-average_gaussian_weight)
            difference_realmean_gaussian_supression_weight=np.abs(real_weight-average_gaussian_weight)

            #Gaussian volume
            sumtotal_gaussian_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_gaussian_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
            #Difference between Gaussian_noise and average real
            difference_gaussian_supression_volume=np.abs(original_average_volume-average_gaussian_volume)
            difference_realmean_gaussian_supression_volume=np.abs(real_volume-average_gaussian_volume)

            #difference L2 with weight and volume Laplace
            L2_laplace=np.linalg.norm([difference_laplace_supression_weight,difference_laplace_supression_volume])
            L2_realvalue_laplace=np.linalg.norm([difference_realvalue_laplace_supression_weight,difference_realvalue_laplace_supression_volume])
            #difference L2 with weight and volume Laplace
            L2_gaussian=np.linalg.norm([difference_gaussian_supression_weight,difference_gaussian_supression_volume])
            L2_realvalue_gaussian=np.linalg.norm([difference_realvalue_gaussian_supression_weight,difference_realvalue_gaussian_supression_volume])
            
            element.append([m, M, delta, epsilon, average_weight, average_volume, L2_laplace, L2_gaussian, L2_realvalue_laplace, L2_realvalue_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_file_to_replace=path_of_file +"\\" + name_of_newfile + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta.csv"
    new_df.to_csv(new_file_to_replace, index=False)
    deleted_element_0(new_file_to_replace)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "original_average_weight","original_average_volume", "L2_laplace", "L2_gaussian","L2_realvalue_laplace","L2_realvalue_gaussian"]
    newelement=[[0]*8]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average_weight=average_df["original_average_weight"]
        average_volume=average_df["original_average_volume"]
        average_L2_laplace=average_df["L2_laplace"]
        average_L2_gaussian=average_df["L2_gaussian"]
        average_L2_realvalue_laplace=average_df["L2_realvalue_laplace"]
        average_L2_realvalue_gaussian=average_df["L2_realvalue_gaussian"]
        element.append([m, M, average_weight, average_volume, average_L2_laplace, average_L2_gaussian, average_L2_realvalue_laplace, average_L2_realvalue_gaussian])
    new_df=pd.DataFrame(newelement, columns=header2)
    new_file_to_replace=path_of_file +"\\" + name_of_newfile + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta_Average.csv"
    new_df.to_csv(new_file_to_replace, index=False)
    deleted_element_0(new_file_to_replace)

"""Compute M with epsilon and delta for Laplace and Gaussian"""
def M_Laplace_and_Gaussian(path="fruits.csv", name_of_newfile="original", fruit_name="Apple", epsilon=1, delta=None, upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, numberofrepeat: int=500):
    path_of_file="File_graphic"
    if not os.path.exists(path_of_file):
    # If it does not exist, create the folder
        os.makedirs(path_of_file)
    
    df=pd.read_csv(path)
    df_fruit=df[df["Fruit Name"]==fruit_name]
    sumtotal_weight=df["Weight"].sum()
    sumtotal_volume=df["Volume"].sum()
    average_weight= df["Weight"].mean()
    average_volume= df["Volume"].mean()
    total_element=df_fruit.shape
    if delta==None:
        delta=np.power((1/total_element[0]), 2)
    else:
        delta=delta
    header=["m", "M", "delta_suppression", "epsilon_suppression", "original_average_weight","original_average_volume", "L2_laplace", "L2_gaussian","L2_realvalue_laplace","L2_realvalue_gaussian"]
    element=[[0]*9]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
    
        for iteration in range(numberofrepeat):
            #Laplace weight
            sumtotal_laplace_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_laplace_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_laplace_weight=sumtotal_laplace_weight/total_element_laplace_weight
            #Difference between Laplace_noise and averages
            difference_laplace_supression_weight=np.abs(original_average_weight-average_laplace_weight)
            difference_realmean_laplace_supression_weight=np.abs(real_weight-average_laplace_weight)

            #Laplace volume
            sumtotal_laplace_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_laplace_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_laplace_volume=sumtotal_laplace_volume/total_element_laplace_volume
            #Difference between Laplace_noise and average real
            difference_laplace_supression_volume=np.abs(original_average_volume-average_laplace_volume)
            difference_realmean_laplace_supression_volume=np.abs(real_volume-average_laplace_volume)
            
            #Gaussian weight
            sumtotal_gaussian_weight=supressed_database_sum_weight+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_gaussian_weight=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
            #Difference between Laplace_noise and averages
            difference_gaussian_supression_weight=np.abs(original_average_weight-average_gaussian_weight)
            difference_realmean_gaussian_supression_weight=np.abs(real_weight-average_gaussian_weight)

            #Gaussian volume
            sumtotal_gaussian_volume=supressed_database_sum_volume+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_gaussian_volume=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
            #Difference between Gaussian_noise and average real
            difference_gaussian_supression_volume=np.abs(original_average_volume-average_gaussian_volume)
            difference_realmean_gaussian_supression_volume=np.abs(real_volume-average_gaussian_volume)

            #difference L2 with weight and volume Laplace
            L2_laplace=np.linalg.norm([difference_laplace_supression_weight,difference_laplace_supression_volume])
            L2_realvalue_laplace=np.linalg.norm([difference_realvalue_laplace_supression_weight,difference_realvalue_laplace_supression_volume])
            #difference L2 with weight and volume Laplace
            L2_gaussian=np.linalg.norm([difference_gaussian_supression_weight,difference_gaussian_supression_volume])
            L2_realvalue_gaussian=np.linalg.norm([difference_realvalue_gaussian_supression_weight,difference_realvalue_gaussian_supression_volume])
            
            element.append([m, M, delta, epsilon, average_weight, average_volume, L2_laplace, L2_gaussian, L2_realvalue_laplace, L2_realvalue_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(path_of_file +"\\" + name_of_newfile + fruit_name + "eps=" + str(epsilon) + "delta=" + str(delta) + "MChangeEpsDelta.csv", index=False)
    deleted_element_0(path_of_file +"\\" + name_of_newfile + fruit_name + "eps=" + str(epsilon) + "delta=" + str(delta) + "MChangeEpsDelta.csv")

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "original_average_weight","original_average_volume", "L2_laplace", "L2_gaussian","L2_realvalue_laplace","L2_realvalue_gaussian"]
    newelement=[[0]*8]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average_weight=average_df["original_average_weight"]
        average_volume=average_df["original_average_volume"]
        average_L2_laplace=average_df["L2_laplace"]
        average_L2_gaussian=average_df["L2_gaussian"]
        average_L2_realvalue_laplace=average_df["L2_realvalue_laplace"]
        average_L2_realvalue_gaussian=average_df["L2_realvalue_gaussian"]
        element.append([m, M, average_weight, average_volume, average_L2_laplace, average_L2_gaussian, average_L2_realvalue_laplace, average_L2_realvalue_gaussian])
    new_df=pd.DataFrame(newelement, columns=header2)
    new_file_to_replace=path_of_file +"\\" + name_of_newfile + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta_Average.csv"
    new_df.to_csv(new_file_to_replace, index=False)
    deleted_element_0(new_file_to_replace)


###Obsolete functions

#This function is useful, as it creates a dataframe, perhaps someone can take advantage of it
def generate_supression_df(path_m_M: str="Files m and M\\Age\\", original_dataset_path: str="irishn_train.csv", column_name: str="Age", numberofrepeat: int=500):
     
    list_element=os.listdir(path_m_M)
    header=["average", "total_sum", "total_element", "m", "M"]
    element=[[0]*5]

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=path_m_M + list_element[i], dataset=original_dataset_path, column_name=column_name)
            data=np.array(data)
            total_sum=data.sum()
            total_element=data.size
            average=total_sum/total_element
            m, M=extract_m_and_Monefile(path_m_M + list_element[i])
            element.append([average, total_sum, total_element, m, M])  
    df=pd.DataFrame(element, columns=header)
    return df

"""Compute MoS for Laplace and Gaussian, but changing the parameters epsilon and delta for epsilon^S and delta^S"""
def aggregateLaplaceandGaussiansuppression(file: str="Age_base.csv", upper=100, lower=0, epsilon: float=1,delta=None):
    folder="File_graphic\\"
    df=pd.read_csv(folder + file)
    totalelement=df.shape
    average_laplace_list=[]
    average_gaussian_list=[]
    delta_suppression_list=[]
    epsilon_suppression_list=[] 
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        sumtotal=new_df["total_sum"]
        total_element=new_df["total_element"]
        original_average=new_df["original average"]

        if (delta==None):
            delta=np.power((1/totalelement[0]), 2)
        else:
            delta=delta

        #Compute epsilon and delta
        delta_suppression=calculate_delta_suppression(delta=delta , m=new_df["m"])
        epsilon_suppression=calculate_eps_suppression(m=float(new_df["m"]), M=float(new_df["M"]), eps=epsilon)
        delta_suppression_list.append(delta_suppression)
        epsilon_suppression_list.append(epsilon_suppression)
        
        #Laplace
        sumtotal_laplace=sumtotal+Laplace_noise(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplace=total_element +Laplace_noise(epsilon1=epsilon_suppression/2, sensitivity=1)
        average_laplace=sumtotal_laplace/total_element_laplace
        #Difference between Laplace_noise and average real
        difference_laplace_supression=np.abs(original_average-average_laplace)
        
        average_laplace_list.append(average_laplace)
        difference_laplace_list.append(difference_laplace_supression)
        
        #Gaussian
        sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=total_element + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=1)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        #Difference between gaussian and average real
        difference_gaussian_supression=np.abs(original_average-average_gaussian)

        average_gaussian_list.append(average_gaussian)
        difference_gaussian_list.append(difference_gaussian_supression)
    df["delta_suppression"]=delta_suppression_list
    df["epsilon_suppression"]=epsilon_suppression_list
    df["average_laplace"]=average_laplace_list
    df["average_gaussian"]=average_gaussian_list
    df["difference_laplace_supression"]=difference_laplace_list
    df["difference_gaussian_supression"]=difference_gaussian_list
    df.to_csv(folder + "Age_base_Lapl_Gauss_suppression.csv", index=False)

def calculateAverageofelement(file: str="", File_name: str="File_graphic\\AverageAge.csv"):
    """This function selects all the elements according to M and M and calculates the average of these, example:
       m=0.1 and M=0.1 Laplace_noise average= 42; m=0.1 m=0.1 Laplace_noise average= 42; you get m=0.1 m=0.1 Laplace_noise average= 42.5
       m=0.1 and M=0.2 Gaussian average= 40; m=0.1 m=0.2 Gaussian average= 50; you get m=0.2 m=0.2 Gaussian average= 45"""
    df=pd.read_csv(file)
    header=["m", "M", "original average", "difference_laplace_supression", "difference_gaussian_supression"]
    element=[[0]*5]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m,)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["original average"]
        average_difference_laplacian=average_df["difference_laplace_supression"]
        average_difference_gaussian=average_df["difference_gaussian_supression"]
        element.append([m, M, average, average_difference_laplacian, average_difference_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(File_name, index=False)
    deleted_element_0(File_name)