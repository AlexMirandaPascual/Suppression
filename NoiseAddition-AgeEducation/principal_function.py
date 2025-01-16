from  graphic_generator import *

def generateFileandGraph(column_name="Age", upper=100, lower=0):
    
    path_m_M = os.path.join("Files m and M",column_name)
    # If it does not exist, create the folder
    if not os.path.exists(path_m_M):
        os.makedirs(path_m_M)

    folder_name = "File_graphic"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_plots_name = "Plots"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_plots_name):
        os.makedirs(folder_plots_name)

    #generate_average_distance_list(filename="irishn_train.csv", column=column_name)
    #generate_files_m_M(path_average_distances=column_name+"distances.csv",path_of_file=path_m_M)

    df=pd.read_csv("irishn_train.csv")
    total_element=df[column_name].size
    delta=np.power((1/total_element), 2)

    output_file_name_base = os.path.join(folder_name, column_name + "_base.csv")
    #generate_iterations_suppressed_database(output_file_name=output_file_name_base, path_m_M=path_m_M, column_name=column_name, numberofrepeat=500)

    for eps in [0.25,0.5,0.75,1,2]:
        file_name_start = os.path.join(folder_name, column_name + "_eps=" + str(eps) + "_delta=" + str(delta))
        #file_name_MoS = file_name_start + "_MoS.csv"
        #file_name_MChangeEpsDelta = file_name_start + "_MChangeEpsDelta.csv"
        #file_name_M = file_name_start + "_M.csv"
        MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS.csv", file=output_file_name_base, upper=upper, lower=lower, epsilon=eps, delta=delta, EpsDeltaChange=False)
        MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS_ChangeEpsDelta.csv", file=output_file_name_base, upper=upper, lower=lower, epsilon=eps, delta=delta, EpsDeltaChange=True)
        M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M.csv", column_name=column_name, upper=upper, lower=lower, epsilon=eps, delta=delta, EpsDeltaChange=False, numberofrepeat=500)
        M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M_ChangeEpsDelta.csv", column_name=column_name, upper=upper, lower=lower, epsilon=eps, delta=delta, EpsDeltaChange=True, numberofrepeat=500)

        #Graphs
        file_name_combined = file_name_start + "_combined.csv"
        DifferenceBetweenMetrics(path_MoS_Average=file_name_start + "_MoS_Average.csv", 
                        path_MoS_ChangeEpsDelta_Average=file_name_start + "_MoS_ChangeEpsDelta_Average.csv",
                        path_M_Average=file_name_start + "_M_Average.csv",
                        path_M_ChangeEpsDelta_Average=file_name_start + "_M_ChangeEpsDelta_Average.csv",
                        output_file_name=file_name_combined)

        plot_name_start = os.path.join(folder_plots_name, column_name + "_eps=" + str(eps) + "_delta=" + str(delta))
        for string in ["difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta", "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS"]:
            generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, plot_values=string, epsilon=eps, smallsuppression=False)
            generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, plot_values=string, epsilon=eps, smallsuppression=True)

