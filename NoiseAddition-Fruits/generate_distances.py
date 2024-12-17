import pandas as pd
import numpy as np


def filedistanceL2(filename="fruits.csv", name_fruit="Apple"):

    df=pd.read_csv(filename)
    df_elements=df[df["Fruit Name"]=="Apple"]
    
    max_valor_Element_weight=df_elements["Weight"].max()
    Element_norm_weight=df_elements["Weight"]/max_valor_Element_weight
    
    max_valor_Element_Volume=df_elements["Volume"].max()
    Element_norm_Volume=df_elements["Volume"]/max_valor_Element_Volume

    promdistancias=[]

    countable_elements=len(Element_norm_weight)
    for i in range(countable_elements): 
        suma_total=0
        for j in range(countable_elements): 
            dist=np.linalg.norm([Element_norm_weight[i]-Element_norm_weight[j], Element_norm_Volume[i]-Element_norm_Volume[j]])
            suma_total=suma_total + dist
        promedio=suma_total/(countable_elements)
        promdistancias.append(promedio)
    dataframenumpy=np.array(promdistancias)
    file=name_fruit + "distances.csv"
    # dataframe.to_csv(path="algo.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)

