import pandas as pd
import numpy as np


def filedistanceL2(filename="irishn_train.csv", column="Age"):

    df=pd.read_csv(filename)
    Element=df[column]
    max_valor_Element=Element.max()
    Element_norm=Element/max_valor_Element
    promdistancias=[]

    countable_elements=len(Element_norm)
    for i in range(countable_elements): 
        suma_total=0
        for j in range(countable_elements): 
            dist=np.linalg.norm([Element_norm[i], Element_norm[j]])
            suma_total=suma_total + dist
        promedio=suma_total/(countable_elements)
        promdistancias.append(promedio)
    dataframenumpy=np.array(promdistancias)
    file=column + "distances.csv"
    # dataframe.to_csv(path="algo.csv", header=["Average distances"], index=None)
    np.savetxt(file, dataframenumpy)