import pandas as pd
import numpy as np

filename="irishn_train.csv"

df=pd.read_csv(filename)
Age=df["Age"]
min_valor_age=Age.min()
print(min_valor_age)
# Age_norm=Age/max_valor_age
# # print(Age_norm)
# a=np.linalg.norm([1,0.2, 2])
# # print(np.sqrt(1**1+ 0.2**2+2**2))
# cuadrado=Age_norm[0::]*Age_norm[0::]
# # print(cuadrado)

# dist=np.linalg.norm(Age_norm[0::], Age_norm[0])
# print(dist)
# promdistancias=[]

# countable_elements=len(Age_norm)
# for i in range(countable_elements): 
#     suma_total=0
#     if i==90:
#         break
#     for j in range(countable_elements): 
#         dist=np.linalg.norm([Age_norm[i], Age_norm[j]])
#         suma_total=suma_total + dist
#     promedio=suma_total/(countable_elements)
#     promdistancias.append(promedio)
# dataframenumpy=np.array(promdistancias)
# # dataframe.to_csv(path="algo.csv", header=["Average distances"], index=None)
# np.savetxt("algo.csv", dataframenumpy)
