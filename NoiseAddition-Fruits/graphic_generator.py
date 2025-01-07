import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  suppression_algorithm import *
def generate3DoriginalvsSuprresion(original_path="irishn_train.csv", path_with_supression="File_graphic\\Age_base 1repeat.csv", column_name="Age"):
    df = pd.read_csv(path_with_supression)
    file_original = pd.read_csv(original_path)
    average_original=file_original[column_name].mean()
    m=df["m"]
    M=df["M"]
    
    df["accuracy_difference"]=df["average"]-float(average_original)
    x=m
    y=M
    z=df["accuracy_difference"]

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(x, y, z, cmap="turbo", linewidth=0.2, antialiased=True)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M', zlabel="accuracy_difference")
    ax.set_title("m vs M vs (average algoritm_suppresion " +str(column_name) + "- average " + str(column_name))
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    # ax.set_box_aspect((3,3,4))
    plt.show()

def generate3DoriginalvsSuprresionAverages(original_path="irishn_train.csv", path_with_supression="File_graphic\\AverageAge.csv", column_name="Age", column_average="average_laplace", epsilon=1, upper=100, lower=0):
    df = pd.read_csv(path_with_supression)
    file_original = pd.read_csv(original_path)
    
    #File original "irishn_train" 
    average_original=file_original[column_name].mean()
    sumtotal=file_original[column_name].sum()
    totalelement=file_original[column_name].size
    delta=np.power((1/totalelement), 2)

    sumtotal_laplace=sumtotal+F1(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
    totalelement_laplacian=totalelement +F1(epsilon1=epsilon/2, sensitivity=1)
    average_laplacian_original=sumtotal_laplace/totalelement_laplacian
    
    sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=sensitivitySummation(upper, lower)) 
    totalelement_gaussian=totalelement + Gaussian_noise(delta=delta, epsilon=epsilon/2, sensitivity=1)
    average_gaussian_original=sumtotal_gaussian/totalelement_gaussian
    
   
    #Choose among average, guassian average and Laplace_noise average
    if (column_average.find("average_laplace")==-1) and  (column_average.find("average_gaussian")==-1):
        df["accuracy_difference"]=np.abs(df[column_average]-float(average_original))
    elif (column_average.find("average_laplace")!=-1):
        df["accuracy_difference"]=np.abs(df[column_average]-float(average_laplacian_original))
    elif (column_average.find("average_gaussian")!=-1):
        df["accuracy_difference"]=np.abs(df[column_average]-float(average_gaussian_original))

    m=df["m"]
    M=df["M"]
    x=m
    y=M
    z=df["accuracy_difference"]

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(x, y, z, cmap="turbo", linewidth=0.2, antialiased=True)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M', zlabel="accuracy_difference")
    ax.set_title("m vs M vs (" + column_average + " algoritm_suppresion " +str(column_name) + "- " + str(column_average)+ " "+ str(column_name))
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    # ax.set_box_aspect((3,3,4))
    plt.show()

def generate3DmetricAverages(path="File_graphic\\CombiningApple.csv" , metric="Laplace_noise"):
    df = pd.read_csv(path)
    if metric=="Laplace_noise":
        m=df["m"]
        M=df["M"]
        x=m
        y=M
        z=df["metric_laplacian"]
    elif metric=="gaussian":
        df_gaussian=df[df["epsilon_suppression"]<2]
        print (df_gaussian)
        m=df_gaussian["m"]
        M=df_gaussian["M"]
        x=m
        y=M
        z=df_gaussian["metric_gaussian"]

    # fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(x, y, z, cmap="turbo", linewidth=0.2, antialiased=True)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M', zlabel="accuracy_difference " + metric + " metric")
    ax.set_title("m vs M vs  " + metric+  " metric" )
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    # ax.set_box_aspect((3,3,4))
    plt.show()
    # new_path=path.replace(".csv", "")
    # plt.savefig(fname=new_path  + metric + ".png", dpi=1000, format="png")


# generate3DoriginalvsSuprresionAverages(column_average="average")
# generate3DoriginalvsSuprresionAverages(column_average="average_laplace")
# generate3DoriginalvsSuprresionAverages(column_average="average_gaussian")

