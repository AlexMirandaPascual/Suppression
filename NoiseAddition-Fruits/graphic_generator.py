import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  suppression_algorithm import *

def generate3DmetricAverages2D(plot_path_start=os.path.join("Plots","Age_eps=1_delta=SQR"), path=os.path.join("File_graphic","Age_eps=1_delta=SQR_combined.csv"), plot_values="difference_laplace_M_minus_MoS", epsilon=1, smallsuppression=False, realmean=False, relative_error=True):
    df_all = pd.read_csv(path)
    Points=[]

    if realmean==False:
        file_name_realmean = ""
        title_realmean = ""
        if relative_error==True:
            average = df_all["L2_original"].iloc[0]
        else:
            average = 1 ##Absolute value: corresponds to dividing by 1
    else:
        file_name_realmean = "_realmean"
        title_realmean = " (real mean)"
        if relative_error==True:
            average = df_all["L2_realmean"].iloc[0]
        else:
            average = 1 ##Absolute value: corresponds to dividing by 1

    if smallsuppression==False:
        df = df_all[df_all["m"]>=0.1]
        file_name_end = "_10--90.pdf"
    else:
        df = df_all[df_all["m"]<0.1]
        file_name_end = "_1--9.pdf"

    plot_df = df.filter(["m","M",plot_values+file_name_realmean],axis=1)

    ##Delete all rows with an NaN value
    plot_df.dropna(inplace=True)

    x = plot_df["m"]
    y = plot_df["M"]
    z = plot_df[plot_values+file_name_realmean]/average*100
    for i in range(plot_df.shape[0]):
        Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])

    file_name_middle = "_"+plot_values+file_name_realmean
    
    if(plot_values=="difference_laplace_M_minus_MoS"):
        title = "PE difference of the Laplace mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"
    elif(plot_values=="difference_gaussian_M_minus_MoS"):
        title = "PE difference of the Gaussian mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
    elif(plot_values=="difference_laplace_M_minus_MoSChangeEpsDelta"):
        title = "PE difference of the Laplace mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $"+str(epsilon)+"$-DP"  
    elif(plot_values=="difference_gaussian_M_minus_MoSChangeEpsDelta"):
        title = "PE difference of the Gaussian mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $("+str(epsilon)+",\\delta)$-DP"  
    elif(plot_values=="difference_laplace_MChangeEpsDelta_minus_MoS"):
        title = "PE difference of the Laplace mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"   
    elif(plot_values=="difference_gaussian_MChangeEpsDelta_minus_MoS"):
        title = "PE difference of the Gaussian mechanism"+title_realmean
        epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP" 

    if len(x)<3:
        print("Plot "+ title + " for epsilon=" + str(epsilon) +" cannot be generated as there are less than three support values")
        return 0

    fig, ax = plt.subplots()
    graph=ax.tricontourf(x, y, z, levels=30, cmap="turbo", antialiased=True)
    fig.colorbar(graph)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M')
    ax.set_title(title)
    
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

    if smallsuppression==False:
        plt.text(0.65,0.3,epsilon_text_M,ha="center",va="center")
        plt.text(0.65,0.2,epsilon_text_MoS,ha="center",va="center")
    else:
        plt.text(0.065,0.03,epsilon_text_M,ha="center",va="center")
        plt.text(0.065,0.02,epsilon_text_MoS,ha="center",va="center")

    plt.savefig(plot_path_start+file_name_middle+file_name_end)

    plt.close()


def generate3DoriginalvsSuprresion(original_path="irishn_train.csv", path_with_suppression="File_graphic\\Age_base 1repeat.csv", column_name="Age"):
    df = pd.read_csv(path_with_suppression)
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
    ax.set_title("m vs M vs (average algorithm_suppression " +str(column_name) + "- average " + str(column_name))
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    # ax.set_box_aspect((3,3,4))
    plt.show()

def generate3DoriginalvsSuprresionAverages(original_path="irishn_train.csv", path_with_suppression="File_graphic\\AverageAge.csv", column_name="Age", column_average="average_laplace", epsilon=1, upper=100, lower=0):
    df = pd.read_csv(path_with_suppression)
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
    ax.set_title("m vs M vs (" + column_average + " algorithm_suppression " +str(column_name) + "- " + str(column_average)+ " "+ str(column_name))
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

