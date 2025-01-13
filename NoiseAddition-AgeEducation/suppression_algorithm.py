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


def generate_files_m_M(list_m: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        path_average_distances="Agedistances.csv",
                        path_of_file="Files m and M\\Age\\"):      
    """Generates files of probabilities that a file is not deleted 
    with combinations of m and M where m 
    is combined with all elements in a valid way"""
    
    # If it does not exist, create the folder
    if not os.path.exists(path_of_file):
        os.makedirs(path_of_file)

    m_and_M = generate_list_m_M(list_m)

    for i in range(len(m_and_M)):
        print("m=",m_and_M[i][0]," M=",m_and_M[i][1])
        name_file= path_of_file +"file m=" + str(m_and_M[i][0]) + " M=" + str(m_and_M[i][1]) + ".csv"
        generate_probabilities_csv(m=m_and_M[i][0],M=m_and_M[i][1], path_distances=path_average_distances, path_probabilidades=name_file)

    #length=len(m)
    #for i in range(len(m)):
    #    if m[i] ==m[-1]:
    #        break
    #    #print("m[i]= ", m[i] )

    #    for j in range(length):
    #        if m[length-j-1]>=m[i]:
    #            #print("m[j]= ", m[length-j-1])
    #            name_file= path_of_file +"file m=" + str(m[i]) + " M=" + str(m[length-j-1]) + ".csv"
    #            generate_probabilities_csv(m=m[i],M=m[length-j-1], path_distances=path_average_distances, path_probabilidades=name_file)
    #        else:
    #            break
    #name_file= path_of_file +"file m=" + str(m[-1]) + " M=" + str(m[-1]) + ".csv"
    #generate_probabilities_csv(m=m[-1], M=m[-1], path_distances=path_average_distances, path_probabilidades=name_file)
    #for i in range(len(m)):
    #    if m[i] ==m[-1]:
    #        break
    #    #print("m[i]= ", m[i])

    #    for j in range(length):
    #        if m[length-j-1]>=m[i]:
    #            #print("m[j]= ", m[length-j-1])
    #            name_file= path_of_file +"file m=" + str(m[i]/10) + " M=" + str(m[length-j-1]/10) + ".csv"
    #            generate_probabilities_csv(m=m[i]/10,M=m[length-j-1]/10, path_distances=path_average_distances, path_probabilidades=name_file)
    #        else:
    #            break
    #name_file= path_of_file +"file m=" + str(m[-1]/10) + " M=" + str(m[-1]/10) + ".csv"
    #generate_probabilities_csv(m=m[-1]/10, M=m[-1]/10, path_distances=path_average_distances, path_probabilidades=name_file)        
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

def generate_iterations_suppressed_database(output_file_name="File_graphic\\Age_base.csv", path_m_M="Files m and M\\Age\\", original_dataset_path="irishn_train.csv", column_name="Age", numberofrepeat: int=500):
    """Generate a file containing the base statistics the of datasets with supression, repeated a number of times"""
    list_element=os.listdir(path_m_M)
    header=[ "m", "M","original_average", "supressed_database_average", "supressed_database_sum", "supressed_database_number_of_elements"]
    element=[[0]*6]
    original_dataset=pd.read_csv(original_dataset_path)
    original_average=float(original_dataset[column_name].mean())
    #folder="File_graphic\\"

    # If folder does not exist, create the folder
    #if not os.path.exists(folder):
    #    os.makedirs(folder)

    for k in range(numberofrepeat):
        for i in range(len(list_element)):
            data=suppressed_dataset(probabilities=path_m_M + list_element[i], dataset=original_dataset_path, column_name=column_name)
            data=np.array(data)

            m, M=extract_m_and_Monefile(path_m_M + list_element[i])

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
def MoS_Laplace_and_Gaussian(output_file_name="File_graphic\\Age_eps=1_delta=SQR_MoS.csv", file="File_graphic\\Age_base.csv", upper=100, lower=0, epsilon=1, delta=None):
    #folder="File_graphic\\"
    df=pd.read_csv(file)
    totalelement=df.shape
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
            delta=np.power((1/totalelement[0]), 2)
        else:
            delta=delta
        #Laplace
        sumtotal_laplace=supressed_database_sum+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
        total_element_laplace=supressed_database_number_of_elements +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
        average_laplace=sumtotal_laplace/total_element_laplace
        #Difference between Laplace_noise and average real
        difference_laplace_supression=np.abs(original_average-average_laplace)
        
        average_laplace_list.append(average_laplace)
        difference_laplace_list.append(difference_laplace_supression)
        
        #Gaussian
        sumtotal_gaussian=supressed_database_sum+ Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
        total_element_gaussian=supressed_database_number_of_elements + Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=1)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        #Difference between gaussian and average real
        difference_gaussian_supression=np.abs(original_average-average_gaussian)

        average_gaussian_list.append(average_gaussian)
        difference_gaussian_list.append(difference_gaussian_supression)

    #file_mod=file.replace("_base.csv","")
    #new_file_to_replace = folder + file_mod + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MoS.csv"
    df["average_laplace"]=average_laplace_list
    df["average_gaussian"]=average_gaussian_list
    df["difference_laplace_supression"]=difference_laplace_list
    df["difference_gaussian_supression"]=difference_gaussian_list
    df.to_csv(output_file_name, index=False)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    df=pd.read_csv(output_file_name)
    header2=["m", "M", "original_average", "difference_laplace_supression", "difference_gaussian_supression"]
    element=[[0]*5]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace_supression"]
        average_difference_gaussian=average_df["difference_gaussian_supression"]
        element.append([m, M, average, average_difference_laplace, average_difference_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    #new_file_to_replace = folder + file_mod + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MoS_Average.csv"
    new_df=pd.DataFrame(element, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

"""Compute M with epsilon^S and delta^S for Laplace and Gaussian"""
def M_Laplace_and_Gaussian_change_of_parameters(output_file_name="File_graphic\\Age_eps=1_delta=SQR_MChangeEpsDelta.csv", path="irishn_train.csv", column_name="Age", epsilon=1, delta=None, upper=100, lower=0, numberofrepeat: int=500):
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
    header=["m", "M", "delta_suppression", "epsilon_suppression", "original_average", "average_laplace", "average_gaussian", "difference_laplace", "difference_gaussian"]
    element=[[0]*9]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        delta_suppression=calculate_delta_suppression(delta=delta , m=m)
        epsilon_suppression=calculate_eps_suppression(m=m, M=M, eps=epsilon)
    
        for iteration in range(numberofrepeat):
            #Laplace
            sumtotal_laplace=sumtotal+Laplace_noise(epsilon1=epsilon_suppression/2, sensitivity=sensitivitySummation(upper, lower))
            total_element_laplace=total_element +Laplace_noise(epsilon1=epsilon_suppression/2, sensitivity=1)
            average_laplace=sumtotal_laplace/total_element_laplace
            difference_laplace=np.abs((sumtotal/total_element)-average_laplace)
            #Gaussian
            sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=sensitivitySummation(upper, lower)) 
            total_element_gaussian=total_element + Gaussian_noise(delta=delta_suppression, epsilon=epsilon_suppression/2, sensitivity=1)
            average_gaussian=sumtotal_gaussian/total_element_gaussian
            difference_gaussian=np.abs((sumtotal/total_element)-average_gaussian)

            element.append([m, M, delta_suppression, epsilon_suppression, original_average, average_laplace, average_gaussian, difference_laplace, difference_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    #new_file_to_replace=path_of_file +"\\" + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta.csv"
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "delta_suppression", "epsilon_suppression", "original_average", "difference_laplace_supression", "difference_gaussian_supression"]
    newelement=[[0]*7]
    m_and_M=generate_list_m_M()
    df=pd.read_csv(output_file_name)
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m)
        print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        delta_suppression=average_df["delta_suppression"]
        epsilon_suppression=average_df["epsilon_suppression"]
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace"]
        average_difference_gaussian=average_df["difference_gaussian"]
        newelement.append([m, M, delta_suppression, epsilon_suppression, original_average, average_difference_laplace, average_difference_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    #new_file_to_replace=path_of_file +"\\" + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_MChangeEpsDelta_Average.csv"
    new_df=pd.DataFrame(newelement, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)


"""Compute M with epsilon and delta for Laplace and Gaussian"""
def M_Laplace_and_Gaussian(output_file_name="File_graphic\\Age_eps=1_delta=SQR_M.csv",path="irishn_train.csv", column_name="Age", epsilon=1, delta=None, upper=100, lower=0, numberofrepeat: int=500):
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
    header=["m", "M", "delta_suppression", "epsilon_suppression", "original_average", "average_laplace", "average_gaussian", "difference_laplace", "difference_gaussian"]
    element=[[0]*9]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
    
        for iteration in range(numberofrepeat):
            #Laplace
            sumtotal_laplace=sumtotal+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
            total_element_laplace=total_element +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
            average_laplace=sumtotal_laplace/total_element_laplace
            difference_laplace=np.abs((sumtotal/total_element)-average_laplace)
            #Gaussian
            sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
            total_element_gaussian=total_element + Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=1)
            average_gaussian=sumtotal_gaussian/total_element_gaussian
            difference_gaussian=np.abs((sumtotal/total_element)-average_gaussian)

            element.append([m, M, delta, epsilon, original_average, average_laplace, average_gaussian, difference_laplace, difference_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    #new_file_to_replace=path_of_file +"\\" + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_M.csv"
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)

    """Now we group every entry with the same pair (m,M) and compute its average"""
    header2=["m", "M", "original_average", "difference_laplace_supression", "difference_gaussian_supression"]
    newelement=[[0]*5]
    m_and_M=generate_list_m_M()
    df=pd.read_csv(output_file_name)
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace"]
        average_difference_gaussian=average_df["difference_gaussian"]
        newelement.append([m, M, original_average, average_difference_laplace, average_difference_gaussian])
    output_file_name_average = output_file_name.replace(".csv","_Average.csv")
    #new_file_to_replace=path_of_file +"\\" + column_name + "_eps=" + str(epsilon) + "_delta=" + str(delta) + "_M_Average.csv"
    new_df=pd.DataFrame(newelement, columns=header2)
    new_df.to_csv(output_file_name_average, index=False)
    deleted_element_0(output_file_name_average)

def DifferenceBetweenMetrics(path_MoS_Average="File_graphic\\Age_eps=1_delta=SQR_MoS_Average.csv", 
                        path_MChangeEpsDelta_Average="File_graphic\\Age_eps=1_delta=SQR_MChangeEpsDelta.csv",
                        path_M_Average="File_graphic\\Age_eps=1_delta=SQR_MChangeEpsDelta.csv",
                       output_file_name="File_graphic\\Age_eps=1_delta=SQR_combined.csv"):
    df_MoS=pd.read_csv(path_MoS_Average)
    df_MChangeEpsDelta=pd.read_csv(path_MChangeEpsDelta_Average)
    df_M=pd.read_csv(path_M_Average)
    header=["m", "M", "delta_suppression", "epsilon_suppression", "difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS"]
    element=[[0]*8]
    
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        print("m=", m)
        print("M=", M)
        df_MoS_instance=df_MoS[(df_MoS["m"]==m) & (df_MoS["M"]==M)]
        df_MChangeEpsDelta_instance=df_MChangeEpsDelta[(df_MChangeEpsDelta["m"]==m) & (df_MChangeEpsDelta["M"]==M)]
        df_M_instance=df_M[(df_M["m"]==m) & (df_M["M"]==M)]
        difference_laplace_M_minus_MoS = df_M_instance["difference_laplace_supression"] - df_MoS_instance["difference_laplace_supression"]
        difference_gaussian_M_minus_MoS = df_M_instance["difference_gaussian_supression"] - df_MoS_instance["difference_gaussian_supression"]
        difference_laplace_MChangeEpsDelta_minus_MoS = df_MChangeEpsDelta_instance["difference_laplace_supression"] - df_MoS_instance["difference_laplace_supression"]
        difference_gaussian_MChangeEpsDelta_minus_MoS = df_MChangeEpsDelta_instance["difference_gaussian_supression"] - df_MoS_instance["difference_gaussian_supression"]
        element.append([m, M, float(df_MChangeEpsDelta_instance["delta_suppression"]), float(df_MChangeEpsDelta_instance["epsilon_suppression"]), float(difference_laplace_M_minus_MoS), float(difference_gaussian_M_minus_MoS), float(difference_laplace_MChangeEpsDelta_minus_MoS), float(difference_gaussian_MChangeEpsDelta_minus_MoS)])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)
    deleted_element_0(output_file_name)





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
        original_average=new_df["original_average"]

        if (delta==None):
            delta=np.power((1/totalelement[0]), 2)
        else:
            delta=delta

        #Compute epsilon and delta
        delta_suppression=calculate_delta_suppression(delta=np.power((1/totalelement[0]), 2) , m=new_df["m"])
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
    header=["m", "M", "original_average", "difference_laplace_supression", "difference_gaussian_supression"]
    element=[[0]*5]
    m_and_M=generate_list_m_M()
    for i in range(len(m_and_M)):
        # print(m_and_M[i])
        m=m_and_M[i][0]
        M=m_and_M[i][1]
        #print("m=", m)
        #print("M=", M)
        average_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        average=average_df["original_average"]
        average_difference_laplace=average_df["difference_laplace_supression"]
        average_difference_gaussian=average_df["difference_gaussian_supression"]
        element.append([m, M, average, average_difference_laplace, average_difference_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(File_name, index=False)
    deleted_element_0(File_name)