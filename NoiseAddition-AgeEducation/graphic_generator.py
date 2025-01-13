import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  suppression_algorithm import *

def generate3DmetricAverages2D(plot_path_start="Plots\\Age_eps=1_delta=SQR", path="File_graphic\\Age_eps=1_delta=SQR_combined.csv", epsilon=1, EpsDeltaChange=False, gaussian=False, smallsuppression=False):
    df_all = pd.read_csv(path)
    Points=[]

    if smallsuppression==False:
        df = df_all[df_all["m"]>=0.1]
        file_name_end = "_10--90.pdf"
    else:
        df = df_all[df_all["m"]<0.1]
        file_name_end = "_1--9.pdf"

    if EpsDeltaChange==False:
        if gaussian==False:
            m=df["m"]
            M=df["M"]
            x=m
            y=M
            z=df["difference_laplace_M_minus_MoS"]
            file_name_middle="_difference_laplace_M_minus_MoS"
            for i in range(df.shape[0]):
                Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])
        else:
            m=df["m"]
            M=df["M"]
            x=m
            y=M
            z=df["difference_gaussian_M_minus_MoS"]
            file_name_middle = "_difference_gaussian_M_minus_MoS"
            for i in range(df.shape[0]):
                Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])
    else:
        if gaussian==False:
            m=df["m"]
            M=df["M"]
            x=m
            y=M
            z=df["difference_laplace_MChangeEpsDelta_minus_MoS"]
            file_name_middle = "_difference_laplace_MChangeEpsDelta_minus_MoS"
            for i in range(df.shape[0]):
                Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])
        else:
            df_gaussian=df[df["epsilon_suppression"]<2]
            m=df_gaussian["m"]
            M=df_gaussian["M"]
            x=m
            y=M
            z=df_gaussian["difference_gaussian_MChangeEpsDelta_minus_MoS"]
            file_name_middle = "_difference_gaussian_MChangeEpsDelta_minus_MoS"
            for i in range(df_gaussian.shape[0]):
                Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])

    fig, ax = plt.subplots()
    graph=ax.tricontourf(x, y, z, levels=30, cmap="turbo", antialiased=True)
    fig.colorbar(graph)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M')
    ax.set_title("m vs M vs "+ file_name_middle)
    
    if smallsuppression==False:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    else:
        ax.set_xlim([0, 0.1])
        ax.set_ylim([0, 0.1])

    ##Plot implicitly when eps^S(eps,m,M)=eps with respect to m and M
    #step = 0.01
    #line_xrange = np.arange(0.01, 9.99, step)
    #line_yrange = np.arange(0.01, 9.99, step)
    #X, Y = np.meshgrid(line_xrange,line_yrange)

    #f = lambda m, M, eps: calculate_eps_suppression(m,M,eps)-eps
    #Z = f(X,Y,epsilon)
    #plt.contour(X,Y,Z, [0], colors="blue")

    #plt.contour(calculate_eps_suppression(epsilon,X,Y)-epsilon, [0], colors="blue")

    ##Add labels for points
    for i in range(len(Points)):
        #plt.text(Points[i][0],Points[i][1],np.format_float_scientific(float(Points[i][2]), precision=1, exp_digits=1),fontsize=7,ha="middle",va="center")
        plt.text(Points[i][0],Points[i][1],np.format_float_positional(float(Points[i][2]), precision=4),fontsize=7,ha="center",va="center")

    plt.savefig(plot_path_start+file_name_middle+file_name_end)



###To Change


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

    sumtotal_laplace=sumtotal+Laplace_noise(epsilon1=epsilon/2, sensitivity=sensitivitySummation(upper, lower))
    totalelement_laplacian=totalelement +Laplace_noise(epsilon1=epsilon/2, sensitivity=1)
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

def generate3DmetricAverages(path="File_graphic\\CombiningAge.csv" , metric="Laplace_noise"):
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

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(x, y, z, cmap="turbo", linewidth=0.2, antialiased=True)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M', zlabel="accuracy_difference " + metric + " metric")
    ax.set_title("m vs M vs  " + metric+  " metric" )
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    # ax.set_box_aspect((3,3,4))
    plt.show()





# generate3DoriginalvsSuprresionAverages(column_average="average")
# generate3DoriginalvsSuprresionAverages(column_average="average_laplace")
# generate3DoriginalvsSuprresionAverages(column_average="average_gaussian")


#prueba
# generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="gaussian")
# generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="Laplace_noise")