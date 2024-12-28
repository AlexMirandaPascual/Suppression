import pandas as pd
import numpy as np


def generate_distances(filename="irishn_train.csv", column="Age"):
    # Normalize Age vector
    df=pd.read_csv(filename)
    Element=df[column]
    max_value_Element=Element.max()
    Element_norm=Element/max_value_Element
    
    outlier_score_list=[]
    length_Element=len(Element_norm)
    #For every element in the database, compute its outlier score and add it to the list
    for i in range(length_Element): 
        total_sum=0
        for j in range(length_Element): 
            dist=np.abs(Element_norm[i]-Element_norm[j])
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    dataframenumpy=np.array(outlier_score_list)
    file=column + "distances.csv"
    # dataframe.to_csv(path="name.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)
