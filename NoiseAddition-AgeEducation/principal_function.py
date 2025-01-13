from  graphic_generator import *

def generateFileandGraph(column_name="Age", upper=100, lower=0):
    #generate_average_distance_list(filename="irishn_train.csv", column=column_name)
    #generate_files_m_M(path_average_distances=column_name+"distances.csv",
    #    path_of_file="Files m and M\\"+column_name+"\\")

    df=pd.read_csv("irishn_train.csv")
    total_element=df[column_name].size
    delta=np.power((1/total_element), 2)

    folder_name = "File_graphic\\"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_plots_name = "Plots\\"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_plots_name):
        os.makedirs(folder_plots_name)

    output_file_name_base = folder_name + column_name + "_base.csv"
    #generate_iterations_suppressed_database(output_file_name=output_file_name_base, path_m_M="Files m and M\\"+column_name+"\\", column_name=column_name, numberofrepeat=500)

    for eps in [0.25,0.5,1,2]:
        file_name_start = folder_name + column_name + "_eps=" + str(eps) + "_delta=" + str(delta)
        #file_name_MoS = file_name_start + "_MoS.csv"
        #file_name_MChangeEpsDelta = file_name_start + "_MChangeEpsDelta.csv"
        #file_name_M = file_name_start + "_M.csv"
        file_name_combined = file_name_start + "_combined.csv"
        #MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS.csv", file=output_file_name_base, upper=upper, lower=lower, epsilon=eps, delta=delta)
        #M_Laplace_and_Gaussian_change_of_parameters(output_file_name = file_name_start + "_MChangeEpsDelta.csv", column_name=column_name, upper=upper, lower=lower, epsilon=eps, delta=delta)
        #M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M.csv", column_name=column_name, upper=upper, lower=lower, epsilon=eps, delta=delta)

        #Graphs
        DifferenceBetweenMetrics(path_MoS_Average=file_name_start + "_MoS_Average.csv", 
                        path_MChangeEpsDelta_Average=file_name_start + "_MChangeEpsDelta_Average.csv",
                        path_M_Average=file_name_start + "_M_Average.csv",
                       output_file_name=file_name_combined)

        plot_name_start = folder_plots_name + column_name + "_eps=" + str(eps) + "_delta=" + str(delta)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=True)



    #DifferenceBetweenMetrics()
    #generate3DmetricAverages(metric="Laplace_noise")
    #generate3DmetricAverages(metric="gaussian")

def generateFileandGraphHighestEducationCompleted():
    #generate_average_distance_list(filename="irishn_train.csv", column="HighestEducationCompleted")
    generate_files_m_M(path_average_distances = "HighestEducationCompleteddistances.csv",
        path_of_file = "Files m and M\\HighestEducationCompleted\\")

    df=pd.read_csv("irishn_train.csv")
    total_element=df["HighestEducationCompleted"].size
    delta=np.power((1/total_element), 2)

    generate_iterations_suppressed_database(path_m_M="Files m and M\\HighestEducationCompleted\\", column_name="HighestEducationCompleted", numberofrepeat=100)
    for eps in [0.25,0.5,1,2]:
        MoS_Laplace_and_Gaussian(file="HighestEducationCompleted_base.csv", upper=10, lower=1, epsilon=eps,delta=delta)
        M_Laplace_and_Gaussian_change_of_parameters(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=eps,delta=delta)
        M_Laplace_and_Gaussian(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=eps, delta=delta)
    #DifferenceBetweenMetrics(path_average_supression = "File_graphic\\AverageHighestEducationCompleted.csv",
    #    path_average_suppression = "File_graphic\\originalHighestEducationCompleted suppression.csv",
    #    file= "File_graphic\\CombiningHighestEducationCompleted.csv")
    #generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="gaussian")
    #generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="Laplace_noise")

