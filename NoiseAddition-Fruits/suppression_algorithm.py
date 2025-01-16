import pandas as pd
import numpy as np
from additional_algorithms import *
from generate_average_distance_list import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

# satisfies 1-differential privacy
def Laplace_noise(epsilon: float=1, sensitivity=1, loc=0):
    if epsilon<=0:
        return float('nan')
    else: 
        return np.random.laplace(loc, scale=sensitivity/epsilon)

def Gaussian_noise(delta, epsilon=1, sensitivity=1):
    if(epsilon<=0 or epsilon>=1):
        return float('nan')
    else:
        variance=(2*np.power(sensitivity,2)*np.log(1.25/delta))/(np.power(epsilon, 2))
        sd=np.sqrt(variance)
        noise=np.random.normal(0, sd)
        return noise

def sensitivitySummation(upper, lower):
    return np.abs(upper - lower)

def weighted_norm(vector, norm_weights=[1/100,1/80]):
    return np.sqrt( (vector[0]*norm_weights[0])**2 + (vector[1]*norm_weights[1])**2 )


def generate_files_m_M(list_m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        path_average_distances="Appledistances.csv",
                        path_of_file=os.path.join("Files m and M","Apple")):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""

    m_and_M = generate_list_m_M(list_m)

    for i in range(len(m_and_M)):
        print("m=",m_and_M[i][0]," M=",m_and_M[i][1])
        name_file= os.path.join(path_of_file,"file m=" + str(m_and_M[i][0]) + " M=" + str(m_and_M[i][1]) + ".csv")
        generate_probabilities_csv(m=m_and_M[i][0],M=m_and_M[i][1], path_distances=path_average_distances, path_probabilities=name_file)

    #length=len(m)
    #for i in range(len(m)):
    #    if m[i] ==m[-1]:
    #        break
    #    #print("m[i]= ", m[i] )

    #    for j in range(length):
    #        if m[length-j-1]>=m[i]:
    #            #print("m[j]= ", m[length-j-1])
    #            name_file= path_of_file +"file m=" + str(m[i]) + " M=" + str(m[length-j-1]) + ".csv"
    #            generate_probabilities_csv(m=m[i],M=m[length-j-1], path_distances=path_average_distances, path_probabilities=name_file)
    #        else:
    #            break
    #name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    #generate_probabilities_csv(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilities=name_file)
    #for i in range(len(m)):
    #    if m[i] ==m[-1]:
    #        break
    #    #print("m[i]= ", m[i])

    #    for j in range(length):
    #        if m[length-j-1]>=m[i]:
    #            #print("m[j]= ", m[length-j-1])
    #            name_file= path_of_file +"file m=" + str(m[i]/10) + " M=" + str(m[length-j-1]/10) + ".csv"
    #            generate_probabilities_csv(m=m[i]/10,M=m[length-j-1]/10, path_distances=path_average_distances, path_probabilities=name_file)
    #        else:
    #            break
    #name_file= path_of_file +"file m=" + str(m[-1]/10) + " M=" + str(m[-1]/10) + ".csv"
    #generate_probabilities_csv(m=m[-1]/10, M=m[-1]/10, path_distances=path_average_distances, path_probabilities=name_file)        
#End of generate_files_m_M 

def generate_list_m_M(list_m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])-> list:
    "generate values of m and M in list"
    m_and_M=[]
    
    length=len(list_m)
    for i in range(length):
        if list_m[i] == list_m[-1]:
            break
        #print("list_m[i]= ", list_m[i] )
        for j in range(length):
            if list_m[length-j-1]>=list_m[i]:
                m_and_M.append([list_m[i],list_m[length-j-1]])
            else:
                break
    m_and_M.append([list_m[-1],list_m[-1]]) #Append m=M=last element of list
    
    list_m = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    length=len(list_m)
    for i in range(length):
        if list_m[i] == list_m[-1]:
            break
        #print("list_m[i]= ", list_m[i] )
        for j in range(length):
            if list_m[length-j-1]>=list_m[i]:
                m_and_M.append([list_m[i],list_m[length-j-1]])
            else:
                break
    m_and_M.append([list_m[-1],list_m[-1]]) #Append m=M=last element of list
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

def generate_iterations_suppressed_database(output_file_name=os.path.join("File_graphic","Apple_base.csv"), path_m_M=os.path.join("Files m and M","Apple"), original_dataset_path="fruits.csv", fruit_name="Apple", numberofrepeat: int=500):
    """Generate a file containing the base statistics the of datasets with suppression, repeated a number of times"""
    list_element=os.listdir(path_m_M)
    header=[ "m", "M","original_average_weight","original_average_volume","supressed_database_average_weight", "supressed_database_sum_weight","supressed_database_average_volume", "supressed_database_sum_volume", "supressed_database_number_of_elements"]
    element=[[0]*9]
    original_dataset=pd.read_csv(original_dataset_path)
    fruit_only_dataset=original_dataset[original_dataset["Fruit Name"]==fruit_name]
    fruit_only_dataset.reindex()
    original_average_weight=float(fruit_only_dataset["Weight"].mean())
    original_average_volume=float(fruit_only_dataset["Volume"].mean())

    df=pd.DataFrame(element, columns=header)

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=os.path.join(path_m_M,list_element[i]), dataset=original_dataset_path, fruit_name=fruit_name)

            m, M=extract_m_and_Monefile(os.path.join(path_m_M,list_element[i]))


            supressed_database_sum_weight=data["Weight"].sum()
            supressed_database_sum_volume=data["Volume"].sum()
            supressed_database_number_of_elements=data.shape
            supressed_database_average_weight=supressed_database_sum_weight/supressed_database_number_of_elements[0]
            supressed_database_average_volume=supressed_database_sum_volume/supressed_database_number_of_elements[0]

            element.append([ m, M, original_average_weight, original_average_volume, supressed_database_average_weight, supressed_database_sum_weight, supressed_database_average_volume, supressed_database_sum_volume, supressed_database_number_of_elements[0]])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)
    #df.to_csv(folder + fruit_name +"_base.csv", index=False)
    #deleted_element_0(folder + fruit_name +"_base.csv")

"""Compute MoS for Laplace and Gaussian"""
def MoS_Laplace_and_Gaussian(output_file_name=os.path.join("File_graphic","Apple_eps=1_delta=SQR_MoS.csv"), file=os.path.join("File_graphic","Apple_base.csv"), upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, real_weight=100, real_volume=80, norm_weights = [1/100,1/80], epsilon=1, delta=None, EpsDeltaChange=True):
    df=pd.read_csv(file)
    totalelement=df.shape
    epsilon_of_M_list=[]
    delta_of_M_list=[]
    L2_original_list=[]
    L2_realmean_list=[]
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
    L2_realmean_laplace_list=[]
    L2_realmean_gaussian_list=[]
    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        supressed_database_sum_weight=new_df["supressed_database_sum_weight"]
        supressed_database_sum_volume=new_df["supressed_database_sum_volume"]
        supressed_database_number_of_elements=new_df["supressed_database_number_of_elements"]

        if (delta==None):
            delta=np.power((1/supressed_database_number_of_elements), 2)
        else:
            delta=delta

        if EpsDeltaChange==True:
            m = df["m"].iloc[i]
            M = df["M"].iloc[i]

            epsilon_of_M = calculate_eps_suppression_inverse(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression_inverse(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta

        epsilon_of_M_list.append(epsilon_of_M)
        delta_of_M_list.append(delta_of_M)

        #Original average
        original_average_weight=new_df["original_average_weight"]
        original_average_volume=new_df["original_average_volume"]
        L2_original = weighted_norm(vector=[original_average_weight,original_average_volume],norm_weights=norm_weights)
        L2_realmean = weighted_norm(vector=[real_weight,real_volume],norm_weights=norm_weights)
        L2_original_list.append(L2_original)
        L2_realmean_list.append(L2_realmean)

        #Laplace weight
        sumtotal_laplace_weight=supressed_database_sum_weight+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
        total_element_laplace_weight=supressed_database_number_of_elements+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
        average_laplace_weight=sumtotal_laplace_weight/total_element_laplace_weight
        #Difference between Laplace_noise and averages
        difference_laplace_suppression_weight=np.abs(original_average_weight-average_laplace_weight)
        difference_realmean_laplace_suppression_weight=np.abs(real_weight-average_laplace_weight)

        #Laplace volume
        sumtotal_laplace_volume=supressed_database_sum_volume+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
        total_element_laplace_volume=supressed_database_number_of_elements +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
        average_laplace_volume=sumtotal_laplace_volume/total_element_laplace_volume
        #Difference between Laplace_noise and average real
        difference_laplace_suppression_volume=np.abs(original_average_volume-average_laplace_volume)
        difference_realmean_laplace_suppression_volume=np.abs(real_volume-average_laplace_volume)
        
        average_laplace_weight_list.append(average_laplace_weight)
        difference_laplace_weight_list.append(difference_laplace_suppression_weight)
        difference_realmean_laplace_weight_list.append(difference_realmean_laplace_suppression_weight)
        average_laplace_volume_list.append(average_laplace_volume)
        difference_laplace_volume_list.append(difference_laplace_suppression_volume)
        difference_realmean_laplace_volume_list.append(difference_realmean_laplace_suppression_volume)
        
        #Gaussian weight
        sumtotal_gaussian_weight=supressed_database_sum_weight+Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=sensitivitySummation(upper_weight, lower_weight))
        total_element_gaussian_weight=supressed_database_number_of_elements +Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=1)
        average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
        #Difference between Laplace_noise and averages
        difference_gaussian_suppression_weight=np.abs(original_average_weight-average_gaussian_weight)
        difference_realmean_gaussian_suppression_weight=np.abs(real_weight-average_gaussian_weight)

        #Gaussian volume
        sumtotal_gaussian_volume=supressed_database_sum_volume + Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=sensitivitySummation(upper_volume, lower_volume))
        total_element_gaussian_volume=supressed_database_number_of_elements + Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=1)
        average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
        #Difference between Gaussian_noise and average real
        difference_gaussian_suppression_volume=np.abs(original_average_volume-average_gaussian_volume)
        difference_realmean_gaussian_suppression_volume=np.abs(real_volume-average_gaussian_volume)
        
        average_gaussian_weight_list.append(average_gaussian_weight)
        difference_gaussian_weight_list.append(difference_gaussian_suppression_weight)
        difference_realmean_gaussian_weight_list.append(difference_realmean_gaussian_suppression_weight)
        average_gaussian_volume_list.append(average_gaussian_volume)
        difference_gaussian_volume_list.append(difference_gaussian_suppression_volume)
        difference_realmean_gaussian_volume_list.append(difference_realmean_gaussian_suppression_volume)

        #difference L2 with weight and volume Laplace
        L2_laplace=weighted_norm(vector=[difference_laplace_suppression_weight,difference_laplace_suppression_volume],norm_weights=norm_weights)
        L2_realmean_laplace=weighted_norm(vector=[difference_realmean_laplace_suppression_weight,difference_realmean_laplace_suppression_volume],norm_weights=norm_weights)
        #difference L2 with weight and volume Laplace
        L2_gaussian=weighted_norm(vector=[difference_gaussian_suppression_weight,difference_gaussian_suppression_volume],norm_weights=norm_weights)
        L2_realmean_gaussian=weighted_norm(vector=[difference_realmean_gaussian_suppression_weight,difference_realmean_gaussian_suppression_volume],norm_weights=norm_weights)
        
        L2_laplace_list.append(L2_laplace)
        L2_realmean_laplace_list.append(L2_realmean_laplace)
        L2_gaussian_list.append(L2_gaussian)
        L2_realmean_gaussian_list.append(L2_realmean_gaussian)

    df["epsilon_of_M"]=epsilon_of_M_list
    df["delta_of_M"]=delta_of_M_list
    df["L2_original"]=L2_original_list
    df["L2_realmean"]=L2_realmean_list
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
    df["L2_realmean_laplace"]=L2_realmean_laplace_list
    df["L2_realmean_gaussian"]=L2_realmean_gaussian_list

    df.to_csv(output_file_name, index=False)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    df=pd.read_csv(output_file_name)
    header2=["m", "M", "epsilon_of_M", "delta_of_M", "L2_original", "L2_realmean", "L2_laplace", "L2_gaussian","L2_realmean_laplace","L2_realmean_gaussian"]
    element=[[0]*10]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)        
        #print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        epsilon_of_M=average_df["epsilon_of_M"]
        delta_of_M=average_df["delta_of_M"]
        average_L2_original=average_df["L2_original"]
        average_L2_realmean=average_df["L2_realmean"]
        original_average_volume=average_df["original_average_volume"]
        average_L2_laplace=average_df["L2_laplace"]
        average_L2_gaussian=average_df["L2_gaussian"]
        average_L2_realmean_laplace=average_df["L2_realmean_laplace"]
        average_L2_realmean_gaussian=average_df["L2_realmean_gaussian"]
        element.append([m, M, epsilon_of_M, delta_of_M, average_L2_original, average_L2_realmean, average_L2_laplace, average_L2_gaussian, average_L2_realmean_laplace, average_L2_realmean_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    new_df=pd.DataFrame(element, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

"""Compute M with epsilon and delta for Laplace and Gaussian"""
def M_Laplace_and_Gaussian(output_file_name=os.path.join("File_graphic","Apple_eps=1_delta=SQR_M.csv"), path="fruits.csv", fruit_name="Apple", epsilon=1, delta=None, upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, real_weight=100, real_volume=80, norm_weights = [1/100,1/80], EpsDeltaChange=True, numberofrepeat: int=500):
    #path_of_file="File_graphic"
    #if not os.path.exists(path_of_file):
    ## If it does not exist, create the folder
    #    os.makedirs(path_of_file)
    
    df=pd.read_csv(path)
    df_fruit=df[df["Fruit Name"]==fruit_name]
    sumtotal_weight=df_fruit["Weight"].sum()
    sumtotal_volume=df_fruit["Volume"].sum()
    original_average_weight= df_fruit["Weight"].mean()
    original_average_volume= df_fruit["Volume"].mean()
    total_element=df_fruit.shape[0]

    L2_original = weighted_norm(vector=[original_average_weight,original_average_volume],norm_weights=norm_weights)
    L2_realmean = weighted_norm(vector=[real_weight,real_volume],norm_weights=norm_weights)

    if delta==None:
        delta=np.power((1/total_element), 2)
    else:
        delta=delta
    header=["m", "M", "epsilon_of_M", "delta_of_M", "L2_original", "L2_realmean", "L2_laplace", "L2_gaussian","L2_realmean_laplace","L2_realmean_gaussian"]
    element=[[0]*10]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        
        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression_inverse(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression_inverse(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta
    
        for iteration in range(numberofrepeat):
            #Laplace weight
            sumtotal_laplace_weight=sumtotal_weight+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_laplace_weight=total_element +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
            average_laplace_weight=sumtotal_laplace_weight/total_element_laplace_weight
            #Difference between Laplace_noise and averages
            difference_laplace_suppression_weight=np.abs(original_average_weight-average_laplace_weight)
            difference_realmean_laplace_suppression_weight=np.abs(real_weight-average_laplace_weight)

            #Laplace volume
            sumtotal_laplace_volume=sumtotal_volume+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_laplace_volume=total_element +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
            average_laplace_volume=sumtotal_laplace_volume/total_element_laplace_volume
            #Difference between Laplace_noise and average real
            difference_laplace_suppression_volume=np.abs(original_average_volume-average_laplace_volume)
            difference_realmean_laplace_suppression_volume=np.abs(real_volume-average_laplace_volume)
            
            #Gaussian weight
            sumtotal_gaussian_weight=sumtotal_weight+Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=sensitivitySummation(upper_weight, lower_weight))
            total_element_gaussian_weight=total_element +Gaussian_noise(epsilon=epsilon_of_M/2, delta=delta_of_M, sensitivity=1)
            average_gaussian_weight=sumtotal_gaussian_weight/total_element_gaussian_weight
            #Difference between Laplace_noise and averages
            difference_gaussian_suppression_weight=np.abs(original_average_weight-average_gaussian_weight)
            difference_realmean_gaussian_suppression_weight=np.abs(real_weight-average_gaussian_weight)

            #Gaussian volume
            sumtotal_gaussian_volume=sumtotal_volume+Gaussian_noise(epsilon=epsilon/2, delta=delta_of_M, sensitivity=sensitivitySummation(upper_volume, lower_volume))
            total_element_gaussian_volume=total_element +Gaussian_noise(epsilon=epsilon/2, delta=delta_of_M, sensitivity=1)
            average_gaussian_volume=sumtotal_gaussian_volume/total_element_gaussian_volume
            #Difference between Gaussian_noise and average real
            difference_gaussian_suppression_volume=np.abs(original_average_volume-average_gaussian_volume)
            difference_realmean_gaussian_suppression_volume=np.abs(real_volume-average_gaussian_volume)

            #difference L2 with weight and volume Laplace
            L2_laplace=weighted_norm(vector=[difference_laplace_suppression_weight,difference_laplace_suppression_volume],norm_weights=norm_weights)
            L2_realmean_laplace=weighted_norm(vector=[difference_realmean_laplace_suppression_weight,difference_realmean_laplace_suppression_volume],norm_weights=norm_weights)
            #difference L2 with weight and volume Laplace
            L2_gaussian=weighted_norm(vector=[difference_gaussian_suppression_weight,difference_gaussian_suppression_volume],norm_weights=norm_weights)
            L2_realmean_gaussian=weighted_norm(vector=[difference_realmean_gaussian_suppression_weight,difference_realmean_gaussian_suppression_volume],norm_weights=norm_weights)

            element.append([m, M, epsilon_of_M, delta_of_M, L2_original, L2_realmean, L2_laplace, L2_gaussian, L2_realmean_laplace, L2_realmean_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    #new_file_to_replace=os.path.join(path_of_file, column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta.csv")
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "epsilon_of_M", "delta_of_M", "L2_original", "L2_realmean", "L2_laplace", "L2_gaussian","L2_realmean_laplace","L2_realmean_gaussian"]
    newelement=[[0]*8]
    m_and_M=generate_list_m_M()
    df=pd.read_csv(output_file_name)
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)        
        #print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        epsilon_of_M=average_df["epsilon_of_M"]
        delta_of_M=average_df["delta_of_M"]
        average_L2_original=average_df["L2_original"]
        average_L2_realmean=average_df["L2_realmean"]
        average_L2_laplace=average_df["L2_laplace"]
        average_L2_gaussian=average_df["L2_gaussian"]
        average_L2_realmean_laplace=average_df["L2_realmean_laplace"]
        average_L2_realmean_gaussian=average_df["L2_realmean_gaussian"]
        newelement.append([m, M, epsilon_of_M, delta_of_M, average_L2_original, average_L2_realmean, average_L2_laplace, average_L2_gaussian, average_L2_realmean_laplace, average_L2_realmean_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    #new_file_to_replace=os.path.join(path_of_file, column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta_Average.csv")
    new_df=pd.DataFrame(newelement, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

def DifferenceBetweenMetrics(path_MoS_Average=os.path.join("File_graphic","Apple_eps=1_delta=SQR_MoS_Average.csv"), 
                        path_MoS_ChangeEpsDelta_Average=os.path.join("File_graphic","Apple_eps=1_delta=SQR_MoS_ChangeEpsDelta_Average.csv"),
                        path_M_Average=os.path.join("File_graphic","Apple_eps=1_delta=SQR_M_Average.csv"),
                        path_M_ChangeEpsDelta_Average=os.path.join("File_graphic","Apple_eps=1_delta=SQR_MChangeEpsDelta_Average.csv"),
                       output_file_name=os.path.join("File_graphic","Apple_eps=1_delta=SQR_combined.csv")):
    df_MoS=pd.read_csv(path_MoS_Average)
    df_MoS_ChangeEpsDelta=pd.read_csv(path_MoS_ChangeEpsDelta_Average)
    df_M=pd.read_csv(path_M_Average)
    df_M_ChangeEpsDelta=pd.read_csv(path_M_ChangeEpsDelta_Average)
    header=["m", "M", "epsilon_of_M", "delta_of_M", "L2_original", "L2_realmean", 
            "difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS",
            "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta", 
            "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS",
            "difference_laplace_M_minus_MoS_realmean", "difference_gaussian_M_minus_MoS_realmean",
            "difference_laplace_M_minus_MoSChangeEpsDelta_realmean", "difference_gaussian_M_minus_MoSChangeEpsDelta_realmean",
            "difference_laplace_MChangeEpsDelta_minus_MoS_realmean", "difference_gaussian_MChangeEpsDelta_minus_MoS_realmean"]
    element=[[0]*18]
    
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        df_MoS_instance=df_MoS[(df_MoS["m"]==m) & (df_MoS["M"]==M)]
        df_MoS_ChangeEpsDelta_instance=df_MoS_ChangeEpsDelta[(df_MoS_ChangeEpsDelta["m"]==m) & (df_MoS_ChangeEpsDelta["M"]==M)]
        df_M_instance=df_M[(df_M["m"]==m) & (df_M["M"]==M)]
        df_M_ChangeEpsDelta_instance=df_M_ChangeEpsDelta[(df_M_ChangeEpsDelta["m"]==m) & (df_M_ChangeEpsDelta["M"]==M)]

        difference_laplace_M_minus_MoS = df_M_instance["L2_laplace"] - df_MoS_instance["L2_laplace"]
        difference_gaussian_M_minus_MoS = df_M_instance["L2_gaussian"] - df_MoS_instance["L2_gaussian"]
        difference_laplace_M_minus_MoSChangeEpsDelta = df_M_instance["L2_laplace"] - df_MoS_ChangeEpsDelta_instance["L2_laplace"]
        difference_gaussian_M_minus_MoSChangeEpsDelta = df_M_instance["L2_gaussian"] - df_MoS_ChangeEpsDelta_instance["L2_gaussian"]
        difference_laplace_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["L2_laplace"] - df_MoS_instance["L2_laplace"]
        difference_gaussian_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["L2_gaussian"] - df_MoS_instance["L2_gaussian"]

        difference_laplace_M_minus_MoS_realmean = df_M_instance["L2_realmean_laplace"] - df_MoS_instance["L2_realmean_laplace"]
        difference_gaussian_M_minus_MoS_realmean = df_M_instance["L2_realmean_gaussian"] - df_MoS_instance["L2_realmean_gaussian"]
        difference_laplace_M_minus_MoSChangeEpsDelta_realmean = df_M_instance["L2_realmean_laplace"] - df_MoS_ChangeEpsDelta_instance["L2_realmean_laplace"]
        difference_gaussian_M_minus_MoSChangeEpsDelta_realmean = df_M_instance["L2_realmean_gaussian"] - df_MoS_ChangeEpsDelta_instance["L2_realmean_gaussian"]
        difference_laplace_MChangeEpsDelta_minus_MoS_realmean = df_M_ChangeEpsDelta_instance["L2_realmean_laplace"] - df_MoS_instance["L2_realmean_laplace"]
        difference_gaussian_MChangeEpsDelta_minus_MoS_realmean = df_M_ChangeEpsDelta_instance["L2_realmean_gaussian"] - df_MoS_instance["L2_realmean_gaussian"]
        element.append([m, M, float(df_M_ChangeEpsDelta_instance["epsilon_of_M"].iloc[0]), float(df_M_ChangeEpsDelta_instance["delta_of_M"].iloc[0]), 
                        float(df_MoS_instance["L2_original"].iloc[0]), float(df_MoS_instance["L2_realmean"].iloc[0]), 
                        float(difference_laplace_M_minus_MoS.iloc[0]), float(difference_gaussian_M_minus_MoS.iloc[0]), 
                        float(difference_laplace_M_minus_MoSChangeEpsDelta.iloc[0]), float(difference_gaussian_M_minus_MoSChangeEpsDelta.iloc[0]),
                        float(difference_laplace_MChangeEpsDelta_minus_MoS.iloc[0]), float(difference_gaussian_MChangeEpsDelta_minus_MoS.iloc[0]),
                        float(difference_laplace_M_minus_MoS_realmean.iloc[0]), float(difference_gaussian_M_minus_MoS_realmean.iloc[0]), 
                        float(difference_laplace_MChangeEpsDelta_minus_MoS_realmean.iloc[0]), float(difference_gaussian_MChangeEpsDelta_minus_MoS_realmean.iloc[0]),
                        float(difference_laplace_MChangeEpsDelta_minus_MoS_realmean.iloc[0]), float(difference_gaussian_MChangeEpsDelta_minus_MoS_realmean.iloc[0])])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)


###Obsolete functions

#This function is useful, as it creates a dataframe, perhaps someone can take advantage of it
def generate_suppression_df(path_m_M: str=os.path.join("Files m and M","Apple"), original_dataset_path: str="irishn_train.csv", column_name: str="Apple", numberofrepeat: int=500):
     
    list_element=os.listdir(path_m_M)
    header=["average", "total_sum", "total_element", "m", "M"]
    element=[[0]*5]

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=os.path.join(path_m_M,list_element[i]), dataset=original_dataset_path, column_name=column_name)
            data=np.array(data)
            total_sum=data.sum()
            total_element=data.size
            average=total_sum/total_element
            m, M=extract_m_and_Monefile(os.path.join(path_m_M,list_element[i]))
            element.append([average, total_sum, total_element, m, M])  
    df=pd.DataFrame(element, columns=header)
    return df