from  graphic_generator import *
def generateFileandGraphAge():
    #generate_average_distance_list()
    generate_files_m_M()

    df=pd.read_csv("irishn_train.csv")
    total_element=df["Age"].size
    delta=np.power((1/total_element), 2)


    generate_iterations_suppressed_database(numberofrepeat=500)
    MoS_Laplace_and_Gaussian(epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(epsilon=1,delta=delta)
    MoS_Laplace_and_Gaussian(epsilon=2,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(epsilon=2,delta=delta)
    M_Laplace_and_Gaussian(epsilon=2,delta=delta)
    #combining_averages()
    #generate3DmetricAverages(metric="Laplace_noise")
    #generate3DmetricAverages(metric="gaussian")

def generateFileandGraphHighestEducationCompleted():
    #generate_average_distance_list(filename="irishn_train.csv", column="HighestEducationCompleted")
    generate_files_m_M(m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        path_average_distances = "HighestEducationCompleteddistances.csv",
        path_of_file = "Files m and M\\HighestEducationCompleted\\")

    df=pd.read_csv("irishn_train.csv")
    total_element=df["HighestEducationCompleted"].size
    delta=np.power((1/total_element), 2)

    generate_iterations_suppressed_database(path_m_M="Files m and M\\HighestEducationCompleted\\", column_name="HighestEducationCompleted", numberofrepeat=100)
    MoS_Laplace_and_Gaussian(file="HighestEducationCompleted_base.csv", upper=10, lower=1, epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.25,delta=delta)
    M_Laplace_and_Gaussian(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.25,delta=delta)
    MoS_Laplace_and_Gaussian(file="HighestEducationCompleted_base.csv", upper=10, lower=1, epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.5,delta=delta)
    M_Laplace_and_Gaussian(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.5,delta=delta)
    MoS_Laplace_and_Gaussian(file="HighestEducationCompleted_base.csv", upper=10, lower=1, epsilon=0.75,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.75,delta=delta)
    M_Laplace_and_Gaussian(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=0.75,delta=delta)
    MoS_Laplace_and_Gaussian(file="HighestEducationCompleted_base.csv", upper=10, lower=1, epsilon=1,delta=delta)
    M_Laplace_and_Gaussian_change_of_parameters(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=1,delta=delta)
    M_Laplace_and_Gaussian(column_name="HighestEducationCompleted", upper = 10, lower = 1, epsilon=1,delta=delta)
    #combining_averages(path_average_supression = "File_graphic\\AverageHighestEducationCompleted.csv",
    #    path_average_suppression = "File_graphic\\originalHighestEducationCompleted suppression.csv",
    #    file= "File_graphic\\CombiningHighestEducationCompleted.csv")
    #generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="gaussian")
    #generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="Laplace_noise")

