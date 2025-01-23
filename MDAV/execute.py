from function_algoritmia import *
from MDAV_function import *

# generate_files_m_M( m=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     path_average_distances = "irishn_train distances.csv",
#     path_of_file=os.path.join("Files m and M", "Irish Age-HEC"))
path_of_file="irish_MDAV.csv"
probabilities_path_one_file="Files m and M\\Irish Age-HECfile m=0.8 M=0.9.csv"
column_name_1="Age"
column_name_2="HighestEducationCompleted"
# df=MDAV_todf()
# df.to_csv(path_of_file)
# df=pd.read_csv("irish_MDAV.csv", index_col=0)
# print(df[:])

def generate_element_with_MDAV(probabilities_path_one_file= "Files m and M\\Irish Age-HECfile m=0.8 M=0.9.csv", column_name_1="Age", column_name_2="HighestEducationCompleted", k=5):
    """Generate error de elements supressed return in form of dataframe"""
    df=suppressed_dataset(probabilities=probabilities_path_one_file, dataset="irishn_train.csv", reset_index=True)
    principal1=(df[column_name_1]-df[column_name_1].mean())/df[column_name_1].std()
    principal2=(df[column_name_2]-df[column_name_2].mean())/df[column_name_2].std()
    df_supre=pd.concat([principal1, principal2], axis=1)

    list_dataframe=pandasMDAVwMSE2D(df2D=df, column_name_1=column_name_1, column_name_2=column_name_2, k=k, Change_value_for_the_centroid = True)
    df_MDAV=Convertlist_to_dataframe(listofdataframe= list_dataframe, sort_index = True)
    
    new_dat=(df_supre[column_name_1]-df_MDAV[column_name_1])**2 +(df_supre[column_name_2]-df_MDAV[column_name_2])**2

    return new_dat

def generate_Average_N_element_with_MDAV(probabilities_path_one_file= "Files m and M\\Irish Age-HECfile m=0.8 M=0.9.csv", column_name_1="Age", column_name_2="HighestEducationCompleted", k=5, number_of_repeat=10):
    df1=pd.DataFrame()
    for i in range(number_of_repeat):    
        df=generate_element_with_MDAV(probabilities_path_one_file=probabilities_path_one_file , column_name_1=column_name_1, column_name_2=column_name_2, k=k)
        df1=pd.concat([df1, df], ignore_index=True)
    return df1.mean().iloc[0]


def generate_df_with_m_and_M(probabilities_path=os.path.join("Files m and M"), file_name="m_M error.csv",  column_name_1="Age", column_name_2="HighestEducationCompleted", k=5, number_of_repeat=100):
    list_element=os.listdir(probabilities_path)
    del list_element[0]
    header=["m", "M", "metric error", "k cluster", "number of repeat"]
    element=[[0]*5]
    df=pd.DataFrame(element, columns=header)
    folder="File_graphic\\"
    if not os.path.exists(folder):
        # If no exist, create the folder
            os.makedirs(folder)
    for i in range(len(list_element)):
        probabilities=os.path.join(probabilities_path,list_element[i])
        metric_error=generate_Average_N_element_with_MDAV(probabilities_path_one_file= probabilities, column_name_1=column_name_1, column_name_2=column_name_2, k=k, number_of_repeat=number_of_repeat)
        m, M=extract_m_and_Monefile(probabilities)
        element.append([m , M, metric_error, k,  number_of_repeat]) 
    df=pd.DataFrame(element, columns=header)
    df.to_csv(folder + file_name, index=False)
    deleted_element_0(folder + file_name)
    # header=["average weight", "total_sum weight", "average volume", "total_sum volume", "total_element", "m", "M"]
    # element=[[0]*7]
    # df=pd.DataFrame(element, columns=header)

def generate_error_original_file(path_file_original="irishn_train.csv", file_to_data="original_error", column_name_1="Age", column_name_2="HighestEducationCompleted", k=5):
    df=pd.read_csv(path_file_original)
    principal1=(df[column_name_1]-df[column_name_1].mean())/df[column_name_1].std()
    principal2=(df[column_name_2]-df[column_name_2].mean())/df[column_name_2].std()
    df=pd.concat([principal1, principal2], axis=1)

    list_dataframe=pandasMDAVwMSE2D(df2D=df, column_name_1=column_name_1, column_name_2=column_name_2, k=k, Change_value_for_the_centroid = True)
    df_MDAV=Convertlist_to_dataframe(listofdataframe= list_dataframe, sort_index = True)
    
    new_dat=(df[column_name_1]-df_MDAV[column_name_1])**2 +(df[column_name_2]-df_MDAV[column_name_2])**2
    folder="File_graphic\\"
    if not os.path.exists(folder):
        # If no exist, create the folder
            os.makedirs(folder)
    name_file=folder + file_to_data + " k="+str(k) + ".csv" 
    new_dat.to_csv(name_file, index_label="index", header=["metric error"])
    print("The metric error with k="+ str(k)+ " is ", new_dat.mean())


















# path_of_data="irishn_train.csv"
# column_name_1="Age"
# column_name_2="HighestEducationCompleted"
# df=pd.read_csv(path_of_data)
#     # Normalize the data
# principal1=(df[column_name_1]-df[column_name_1].mean())/df[column_name_1].std()
# principal2=(df[column_name_2]-df[column_name_2].mean())/df[column_name_2].std()
# df_new=pd.concat([principal1, principal2], axis=1)
# print(df_new[:])


# df=suppressed_dataset(probabilities="C:\\MDAV\\Files m and M\\Irish Age-HECfile m=0.8 M=0.9.csv", dataset="irishn_train.csv")
# print(df[:])
