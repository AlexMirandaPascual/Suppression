import pandas as pd
import numpy as np

def generate_average_distance_list(filename="fruits.csv", name_fruit="Apple"):
    # Normalize Age vector
    df=pd.read_csv(filename)
    df_elements=df[df["Fruit Name"]==name_fruit]
    
    max_value_Element_weight=df_elements["Weight"].max()
    Element_norm_weight=df_elements["Weight"]/max_value_Element_weight
    
    max_value_Element_Volume=df_elements["Volume"].max()
    Element_norm_Volume=df_elements["Volume"]/max_value_Element_Volume
    
    outlier_score_list=[]
    length_Element=len(Element_norm_weight)
    #For every element in the database, compute its outlier score and add it to the list
    for i in range(length_Element): 
        total_sum=0
        for j in range(length_Element): 
            dist=np.linalg.norm([Element_norm_weight.iloc[i]-Element_norm_weight.iloc[j], Element_norm_Volume.iloc[i]-Element_norm_Volume.iloc[j]])
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    dataframenumpy=np.array(outlier_score_list)
    file=name_fruit + "distances.csv"
    # dataframe.to_csv(path="name.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)
