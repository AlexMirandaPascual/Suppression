import pandas as pd
import numpy as np

def generate_average_distance_list(filename="irishn_train.csv", column_name_1="Age", column_name_2="HighestEducationCompleted"):
    # Normalize Age vector
    df=pd.read_csv(filename)
    df_elements_column_1=df[column_name_1]
    df_elements_column_2=df[column_name_2]
    
    max_value_Element_1=df_elements_column_1.max()
    Element_norm_1=df_elements_column_1/max_value_Element_1
    
    max_value_Element_2=df_elements_column_2.max()
    Element_norm_2=df_elements_column_2/max_value_Element_2
    
    outlier_score_list=[]
    length_Element=len(Element_norm_1)
    #For every element in the database, compute its outlier score and add it to the list
    for i in range(length_Element): 
        total_sum=0
        for j in range(length_Element): 
            dist=np.linalg.norm([Element_norm_1.iloc[i]-Element_norm_2.iloc[j], Element_norm_2.iloc[i]-Element_norm_2.iloc[j]])
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    dataframenumpy=np.array(outlier_score_list)
    file_new=filename.replace(".csv", " ")
    file=file_new + "distances.csv"
    # dataframe.to_csv(path="name.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)
