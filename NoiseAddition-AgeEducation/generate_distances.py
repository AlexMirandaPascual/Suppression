import pandas as pd
import numpy as np


def filedistanceL2(filename="irishn_train.csv", column="Age"):

    df=pd.read_csv(filename)
    Element=df[column]
    max_value_Element=Element.max()
    Element_norm=Element/max_value_Element
    outlier_score_list=[]

    length_Element=len(Element_norm)
    for i in range(length_Element): 
        total_sum=0
        for j in range(length_Element): 
            dist=np.linalg.norm([Element_norm[i], Element_norm[j]])
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    dataframenumpy=np.array(outlier_score_list)
    file=column + "distances.csv"
    # dataframe.to_csv(path="algo.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)
