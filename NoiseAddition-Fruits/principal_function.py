from graphic_generator import *


def generateFileandGraph(fruit_name="Apple", upper_weight=140, lower_weight=60, upper_volume=105, lower_volume=45, real_weight=100, real_volume=80):
    #generate_average_distance_list(filename="fruits.csv", name_fruit=fruit_name)
    generate_files_m_M(path_average_distances=fruit_name+"distances.csv",
        path_of_file="Files m and M\\"+fruit_name+"\\")

    delta = np.power((1/1003), 2)

    folder_name = "File_graphic\\"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_plots_name = "Plots\\"
    # If folder does not exist, create the folder
    if not os.path.exists(folder_plots_name):
        os.makedirs(folder_plots_name)

    output_file_name_base = folder_name + fruit_name + "_base.csv"
    generate_iterations_suppressed_database(output_file_name=output_file_name_base, path_m_M="Files m and M\\"+fruit_name+"\\", fruit_name=fruit_name, numberofrepeat=500)

    for eps in [0.25,0.5,1,2]:
        file_name_start = folder_name + fruit_name + "_eps=" + str(eps) + "_delta=" + str(delta)
        #file_name_MoS = file_name_start + "_MoS.csv"
        #file_name_MChangeEpsDelta = file_name_start + "_MChangeEpsDelta.csv"
        #file_name_M = file_name_start + "_M.csv"
        file_name_combined = file_name_start + "_combined.csv"
        MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS.csv", file=output_file_name_base, upper_weight=upper_weight, lower_weight=lower_weight, upper_volume=upper_volume, lower_volume=lower_volume, real_weight=real_weight, real_volume=real_volume, epsilon=eps, delta=delta)
        M_Laplace_and_Gaussian_change_of_parameters(output_file_name = file_name_start + "_MChangeEpsDelta.csv", fruit_name=fruit_name, upper_weight=upper_weight, lower_weight=lower_weight, upper_volume=upper_volume, lower_volume=lower_volume, real_weight=real_weight, epsilon=eps, delta=delta)
        M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M.csv", fruit_name=fruit_name, upper_weight=upper_weight, lower_weight=lower_weight, upper_volume=upper_volume, lower_volume=lower_volume, real_weight=real_weight, epsilon=eps, delta=delta)

        #Graphs
        DifferenceBetweenMetrics(path_MoS_Average=file_name_start + "_MoS_Average.csv", 
                        path_MChangeEpsDelta_Average=file_name_start + "_MChangeEpsDelta_Average.csv",
                        path_M_Average=file_name_start + "_M_Average.csv",
                       output_file_name=file_name_combined)

        plot_name_start = folder_plots_name + fruit_name + "_eps=" + str(eps) + "_delta=" + str(delta)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=False, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=False, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=False, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=False, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=True, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=True, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=True, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=True, realmean=False)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=False, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=False, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=False, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=False, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=False, smallsuppression=True, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=False, gaussian=True, smallsuppression=True, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=False, smallsuppression=True, realmean=True)
        generate3DmetricAverages2D(plot_path_start=plot_name_start, path=file_name_combined, epsilon=eps, EpsDeltaChange=True, gaussian=True, smallsuppression=True, realmean=True)



def generateFileandGraphApple():
    delta = np.power((1/1000), 2)
    generate_average_distance_list()
    generate_files_m_M()
    generate_iterations_suppressed_database()
    MoS_Laplace_and_Gaussian(epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(epsilon=0.25,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(epsilon=1,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=2,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=2,delta=delta)
    M_Laplace_and_Gaussian(epsilon=2,delta=delta)
    #calculateAverageofelement()
    #DifferenceBetweenMetrics()
    #generate3DmetricAverages(metric="Laplace_noise")
    #generate3DmetricAverages(metric="gaussian")

def generateFileandGraphMelon():
    delta = np.power((1/1000), 2)
    generate_average_distance_list(name_fruit="Melon")
    generate_files_m_M(path_average_distances="Melondistances.csv", path_of_file = "Files m and M\\Melon\\")
    generate_iterations_suppressed_database(path_m_M = "Files m and M\\Melon\\", fruit_name="Melon")
    MoS_Laplace_and_Gaussian(file="Melon_base.csv", upper_weight= 2000, lower_weight= 1000, upper_volume= 300, lower_volume= 100, real_weight=1500, real_volume=200,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=0.25,delta=delta)
    MoS_Laplace_and_Gaussian(file="Melon_base.csv", upper_weight= 2000, lower_weight= 1000, upper_volume= 300, lower_volume= 100, real_weight=1500, real_volume=200,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(file="Melon_base.csv", upper_weight= 2000, lower_weight= 1000, upper_volume= 300, lower_volume= 100, real_weight=1500, real_volume=200,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=1,delta=delta)
    MoS_Laplace_and_Gaussian(file="Melon_base.csv", upper_weight= 2000, lower_weight= 1000, upper_volume= 300, lower_volume= 100, real_weight=1500, real_volume=200,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100, real_weight=1500, real_volume=200,epsilon=2,delta=delta)
    #calculateAverageofelement(file="File_graphic\\Melon_base.csv", File_name="File_graphic\\AverageMelon.csv")
    #DifferenceBetweenMetrics(path_average_supression="File_graphic\\AverageMelon.csv", path_average_suppression="File_graphic\\originalMelon suppression.csv",
    #                   file="File_graphic\\CombiningMelon.csv", real_weight=1500, real_volume=200)
    #generate3DmetricAverages(path = "File_graphic\\CombiningMelon.csv", metric="Laplace_noise")
    #generate3DmetricAverages(path = "File_graphic\\CombiningMelon.csv", metric="gaussian")

def generateFileandGraphStrawberry():
    delta = np.power((1/1000), 2)
    generate_average_distance_list(name_fruit="Strawberry")
    generate_files_m_M(path_average_distances="Strawberrydistances.csv", path_of_file = "Files m and M\\Strawberry\\")
    generate_iterations_suppressed_database(path_m_M = "Files m and M\\Strawberry\\", fruit_name="Strawberry")
    MoS_Laplace_and_Gaussian(file="Strawberry_base.csv", upper_weight= 45, lower_weight= 20, upper_volume= 30, lower_volume= 5, real_weight=30, real_volume=20,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=0.25,delta=delta)
    MoS_Laplace_and_Gaussian(file="Strawberry_base.csv", upper_weight= 45, lower_weight= 20, upper_volume= 30, lower_volume= 5, real_weight=30, real_volume=20,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(file="Strawberry_base.csv", upper_weight= 45, lower_weight= 20, upper_volume= 30, lower_volume= 5, real_weight=30, real_volume=20,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=1,delta=delta)
    MoS_Laplace_and_Gaussian(file="Strawberry_base.csv", upper_weight= 45, lower_weight= 20, upper_volume= 30, lower_volume= 5, real_weight=30, real_volume=20,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5, real_weight=30, real_volume=20,epsilon=2,delta=delta)
    #calculateAverageofelement(file="File_graphic\\Strawberry_base.csv", File_name="File_graphic\\AverageStrawberry.csv")
    #DifferenceBetweenMetrics(path_average_supression="File_graphic\\AverageStrawberry.csv", path_average_suppression="File_graphic\\originalStrawberry suppression.csv",
    #                   file="File_graphic\\CombiningStrawberry.csv", real_weight=30, real_volume=20)
    #generate3DmetricAverages(path = "File_graphic\\CombiningStrawberry.csv", metric="Laplace_noise")
    #generate3DmetricAverages(path = "File_graphic\\CombiningStrawberry.csv", metric="gaussian")

def generateFileandGraphPear():
    delta = np.power((1/1000), 2)
    generate_average_distance_list(name_fruit="Pear")
    generate_files_m_M(path_average_distances="Peardistances.csv", path_of_file = "Files m and M\\Pear\\")
    generate_iterations_suppressed_database(path_m_M = "Files m and M\\Pear\\", fruit_name="Pear")
    MoS_Laplace_and_Gaussian(file="Pear_base.csv", upper_weight= 200, lower_weight= 90, upper_volume= 160, lower_volume= 80, real_weight=140, real_volume=120,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=0.25,delta=delta)
    MoS_Laplace_and_Gaussian(file="Pear_base.csv", upper_weight= 200, lower_weight= 90, upper_volume= 160, lower_volume= 80, real_weight=140, real_volume=120,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(file="Pear_base.csv", upper_weight= 200, lower_weight= 90, upper_volume= 160, lower_volume= 80, real_weight=140, real_volume=120,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=1,delta=delta)
    MoS_Laplace_and_Gaussian(file="Pear_base.csv", upper_weight= 200, lower_weight= 90, upper_volume= 160, lower_volume= 80, real_weight=140, real_volume=120,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=2,delta=delta)
    M_Laplace_and_Gaussian(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80, real_weight=140, real_volume=120,epsilon=2,delta=delta)
    #calculateAverageofelement(file="File_graphic\\Pear_base.csv", File_name="File_graphic\\AveragePear.csv")
    #DifferenceBetweenMetrics(path_average_supression="File_graphic\\AveragePear.csv", path_average_suppression="File_graphic\\originalPear suppression.csv",
    #                   file="File_graphic\\CombiningPear.csv", real_weight=140, real_volume=120)
    #generate3DmetricAverages(path = "File_graphic\\CombiningPear.csv", metric="Laplace_noise")
    #generate3DmetricAverages(path = "File_graphic\\CombiningPear.csv", metric="gaussian")




