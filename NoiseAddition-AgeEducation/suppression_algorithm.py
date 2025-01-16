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


def generate_files_m_M(list_m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        path_average_distances="Agedistances.csv",
                        path_of_file=os.path.join("Files m and M","Age")):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""

    m_and_M = generate_list_m_M(list_m)

    for i in range(len(m_and_M)):
        #print("m=",m_and_M[i][0]," M=",m_and_M[i][1])
        name_file= os.path.join(path_of_file, "file m=" + str(m_and_M[i][0]) + " M=" + str(m_and_M[i][1]) + ".csv")
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

def suppressed_dataset(probabilities, dataset, column_name):
    """"Given a dataset, it suppresses elements according to the outlier scores in the probabilities file"""
    new_dataset=[]
    probabilities_df=pd.read_csv(probabilities)
    dataset_df=pd.read_csv(dataset)
    element=dataset_df[column_name]
    vector_size=dataset_df.shape
    total_element=vector_size[0]

    for i in range(total_element-1):
        x=np.random.random(1)
        if (x<=probabilities_df.iloc[i, 0]):
            new_dataset.append(element[i])
    return new_dataset

def generate_iterations_suppressed_database(output_file_name=os.path.join("File_graphic","Age_base.csv"), path_m_M=os.path.join("Files m and M","Age"), original_dataset_path="irishn_train.csv", column_name="Age", numberofrepeat: int=500):
    """Generate a file containing the base statistics the of datasets with suppression, repeated a number of times"""
    list_element=os.listdir(path_m_M)
    header=[ "m", "M","original_average", "supressed_database_average", "supressed_database_sum", "supressed_database_number_of_elements"]
    element=[[0]*6]
    original_dataset=pd.read_csv(original_dataset_path)
    original_average=float(original_dataset[column_name].mean())

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=os.path.join(path_m_M,list_element[i]), dataset=original_dataset_path, column_name=column_name)
            data=np.array(data)

            m, M=extract_m_and_Monefile(os.path.join(path_m_M,list_element[i]))

            supressed_database_sum=data.sum()
            supressed_database_number_of_elements=data.size
            supressed_database_average=supressed_database_sum/supressed_database_number_of_elements

            element.append([m, M, original_average, supressed_database_average, supressed_database_sum, supressed_database_number_of_elements])  
    df=pd.DataFrame(element, columns=header)
    df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)
    #df.to_csv(folder + column_name +"_base.csv", index=False)
    #deleted_element_0(folder + column_name +"_base.csv")

"""Compute MoS for Laplace and Gaussian"""
def MoS_Laplace_and_Gaussian(output_file_name=os.path.join("File_graphic","Age_eps=1_delta=SQR_MoS.csv"), file=os.path.join("File_graphic","Age_base.csv"), upper=100, lower=0, epsilon=1, delta=None, EpsDeltaChange=True):
    df=pd.read_csv(file)
    totalelement=df.shape

    epsilon_of_M_list=[]
    delta_of_M_list=[]
    average_laplace_list=[]
    average_gaussian_list=[]
    difference_laplace_list=[]
    difference_gaussian_list=[]

    for i in range(totalelement[0]):
        new_df=df.iloc[i]
        supressed_database_sum=new_df["supressed_database_sum"]
        supressed_database_number_of_elements=new_df["supressed_database_number_of_elements"]
        original_average=new_df["original_average"]

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

        #Laplace
        sumtotal_laplace=supressed_database_sum+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplace=supressed_database_number_of_elements +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
        average_laplace=sumtotal_laplace/total_element_laplace
        #Difference between Laplace_noise and average real
        difference_laplace_suppression=np.abs(original_average-average_laplace)
        
        average_laplace_list.append(average_laplace)
        difference_laplace_list.append(difference_laplace_suppression)
        
        #Gaussian
        sumtotal_gaussian=supressed_database_sum+ Gaussian_noise(delta=delta_of_M, epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=supressed_database_number_of_elements + Gaussian_noise(delta=delta_of_M, epsilon=epsilon_of_M/2, sensitivity=1)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        #Difference between gaussian and average real
        difference_gaussian_suppression=np.abs(original_average-average_gaussian)

        average_gaussian_list.append(average_gaussian)
        difference_gaussian_list.append(difference_gaussian_suppression)

    df["epsilon_of_M"]=epsilon_of_M_list
    df["delta_of_M"]=delta_of_M_list
    df["average_laplace"]=average_laplace_list
    df["average_gaussian"]=average_gaussian_list
    df["difference_laplace_suppression"]=difference_laplace_list
    df["difference_gaussian_suppression"]=difference_gaussian_list
    df.to_csv(output_file_name, index=False)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    df=pd.read_csv(output_file_name)
    header2=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "difference_laplace_suppression", "difference_gaussian_suppression"]
    element=[[0]*7]
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
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace_suppression"]
        average_difference_gaussian=average_df["difference_gaussian_suppression"]
        element.append([m, M, epsilon_of_M, delta_of_M, average, average_difference_laplace, average_difference_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    new_df=pd.DataFrame(element, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

"""Compute M with epsilon and delta for Laplace and Gaussian"""
def M_Laplace_and_Gaussian(output_file_name=os.path.join("File_graphic","Age_eps=1_delta=SQR_M_ChangeEpsDelta.csv"),path="irishn_train.csv", column_name="Age", epsilon=1, delta=None, upper=100, lower=0, EpsDeltaChange=True, numberofrepeat: int=500):
    #path_of_file="File_graphic"
    #if not os.path.exists(path_of_file):
    ## If it does not exist, create the folder
    #    os.makedirs(path_of_file)
    
    df=pd.read_csv(path)
    sumtotal=df[column_name].sum()
    original_average= df[column_name].mean()
    total_element=df[column_name].size
    if delta==None:
        delta=np.power((1/total_element), 2)
    else:
        delta=delta

    header=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "average_laplace", "average_gaussian", "difference_laplace", "difference_gaussian"]
    element=[[0]*9]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        m=m_and_M[i][0]
        M=m_and_M[i][1]

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta
    
        for iteration in range(numberofrepeat):
            #Laplace
            sumtotal_laplace=sumtotal+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper, lower))
            total_element_laplace=total_element +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
            average_laplace=sumtotal_laplace/total_element_laplace
            difference_laplace=np.abs((sumtotal/total_element)-average_laplace)
            #Gaussian
            sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta_of_M, epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(upper, lower)) 
            total_element_gaussian=total_element + Gaussian_noise(delta=delta_of_M, epsilon=epsilon_of_M/2, sensitivity=1)
            average_gaussian=sumtotal_gaussian/total_element_gaussian
            difference_gaussian=np.abs((sumtotal/total_element)-average_gaussian)

            element.append([m, M, epsilon_of_M, delta_of_M, original_average, average_laplace, average_gaussian, difference_laplace, difference_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    #new_file_to_replace=os.path.join(path_of_file, column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_M.csv")
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "difference_laplace_suppression", "difference_gaussian_suppression"]
    newelement=[[0]*7]
    m_and_M=generate_list_m_M()
    df=pd.read_csv(output_file_name)
    for i in range(len(m_and_M)):
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        epsilon_of_M=average_df["epsilon_of_M"]
        delta_of_M=average_df["delta_of_M"]
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace"]
        average_difference_gaussian=average_df["difference_gaussian"]
        newelement.append([m, M, epsilon_of_M, delta_of_M, original_average, average_difference_laplace, average_difference_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    new_df=pd.DataFrame(newelement, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

def DifferenceBetweenMetrics(path_MoS_Average=os.path.join("File_graphic","Age_eps=1_delta=SQR_MoS_Average.csv"), 
                        path_MoS_ChangeEpsDelta_Average=os.path.join("File_graphic","Age_eps=1_delta=SQR_MoS_ChangeEpsDelta_Average.csv"),
                        path_M_Average=os.path.join("File_graphic","Age_eps=1_delta=SQR_M_Average.csv"),
                        path_M_ChangeEpsDelta_Average=os.path.join("File_graphic","Age_eps=1_delta=SQR_MChangeEpsDelta_Average.csv"),
                       output_file_name=os.path.join("File_graphic","Age_eps=1_delta=SQR_combined.csv")):
    df_MoS=pd.read_csv(path_MoS_Average)
    df_MoS_ChangeEpsDelta=pd.read_csv(path_MoS_ChangeEpsDelta_Average)
    df_M=pd.read_csv(path_M_Average)
    df_M_ChangeEpsDelta=pd.read_csv(path_M_ChangeEpsDelta_Average)
    header=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", 
            "difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", 
            "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta", 
            "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS"]
    element=[[0]*11]
    
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
        original_average = df_MoS_instance["original_average"]
        difference_laplace_M_minus_MoS = df_M_instance["difference_laplace_suppression"] - df_MoS_instance["difference_laplace_suppression"]
        difference_gaussian_M_minus_MoS = df_M_instance["difference_gaussian_suppression"] - df_MoS_instance["difference_gaussian_suppression"]
        difference_laplace_M_minus_MoSChangeEpsDelta = df_M_instance["difference_laplace_suppression"] - df_MoS_ChangeEpsDelta_instance["difference_laplace_suppression"]
        difference_gaussian_M_minus_MoSChangeEpsDelta = df_M_instance["difference_gaussian_suppression"] - df_MoS_ChangeEpsDelta_instance["difference_gaussian_suppression"]
        difference_laplace_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["difference_laplace_suppression"] - df_MoS_instance["difference_laplace_suppression"]
        difference_gaussian_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["difference_gaussian_suppression"] - df_MoS_instance["difference_gaussian_suppression"]
        element.append([m, M, float(df_M_ChangeEpsDelta_instance["epsilon_of_M"].iloc[0]), float(df_M_ChangeEpsDelta_instance["delta_of_M"].iloc[0]), float(original_average.iloc[0]),
                        float(difference_laplace_M_minus_MoS.iloc[0]), float(difference_gaussian_M_minus_MoS.iloc[0]), 
                        float(difference_laplace_M_minus_MoSChangeEpsDelta.iloc[0]), float(difference_gaussian_M_minus_MoSChangeEpsDelta.iloc[0]),
                        float(difference_laplace_MChangeEpsDelta_minus_MoS.iloc[0]), float(difference_gaussian_MChangeEpsDelta_minus_MoS.iloc[0])])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)





###Obsolete functions

#This function is useful, as it creates a dataframe, perhaps someone can take advantage of it
def generate_suppression_df(path_m_M: str=os.path.join("Files m and M","Age"), original_dataset_path: str="irishn_train.csv", column_name: str="Age", numberofrepeat: int=500):
     
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