from  function_dp import *

def generateFile_principal(filename="irishn_train.csv", column_name="Age", epsilon=1, upper=100, lower=0):
    filename="irishn_train.csv"
    df=pd.read_csv(filename)
    sumtotal=df[column_name].sum()
    print("suma total", sumtotal)
    average= df[column_name].mean()
    totalelement=df[column_name].size
    print("totalelement", totalelement)
    
    delta=np.power((1/totalelement), 2)
    print("delta", delta)
    sumtotal_laplacian=sumtotal+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
    totalelement_laplacian=totalelement +F1(epsilon1=epsilon/2, sensitivity=1)
    average_laplacian=sumtotal_laplacian/totalelement_laplacian
    
    sumtotal_gaussian=sumtotal+ Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
    totalelement_gaussian=totalelement + Gaussian_p(delta=delta, epsilon=epsilon/2, sensitivity=1)
    average_gaussian=sumtotal_gaussian/totalelement_gaussian
    
    print("average_laplacian", average_laplacian)
    print("average_gaussian", average_gaussian)
    print("average", average)


# def generatefile_delta_epsilon_prima(file="File_graphic\\Age_onlysupression.csv", epsilon=[0.5, 0.75, 1, 1.5 , 2, 2.5, 3, 3.5, 4]):
#     df=pd.read_csv(file)
#     total_element=df.shape
#     print(df.iloc[0])
#     average
#     for i in range(total_element-1)
#         df.iloc[i]
# generatefile_delta_epsilon_prima()

