from  graphic_generator import *
def generateFileandGraphAge():
    filedistanceL2()
    generate_files_m_M()
    generate_supression_file(numberofrepeat=100)
    agregateLaplaceandGaussian()
    agregatefileoriginalPrima()
    calculateAverageofelement()
    combining_averages()
    generate3DmetricAverages(metric="laplacian")
    generate3DmetricAverages(metric="gaussian")

def generateFileandGraphHighestEducationCompleted():
    filedistanceL2(filename="irishn_train.csv", column="HighestEducationCompleted")
    generate_files_m_M(m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        path_average_distances = "HighestEducationCompleteddistances.csv",
        path_of_file = "Files m and M\\HighestEducationCompleted\\")
    generate_supression_file(path_m_M="Files m and M\\HighestEducationCompleted\\", column_name="HighestEducationCompleted", numberofrepeat=100)
    agregateLaplaceandGaussian(file="HighestEducationCompleted_onlysupression.csv", upper=10, lower=1)
    agregatefileoriginalPrima(column_name="HighestEducationCompleted", upper = 10, lower = 1)
    calculateAverageofelement(file="File_graphic\\HighestEducationCompleted_onlysupression.csv", File_name= "File_graphic\\AverageHighestEducationCompleted.csv")
    combining_averages(path_average_supression = "File_graphic\\AverageHighestEducationCompleted.csv",
        path_average_prima = "File_graphic\\originalHighestEducationCompleted prima.csv",
        file= "File_graphic\\CombiningHighestEducationCompleted.csv")
    generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="gaussian")
    generate3DmetricAverages(path="File_graphic\\CombiningHighestEducationCompleted.csv", metric="laplacian")

