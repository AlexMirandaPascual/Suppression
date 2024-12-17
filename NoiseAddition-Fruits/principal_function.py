from graphic_generator import *

def generateFileandGraphApple():
    filedistanceL2()
    generate_files_m_M()
    generate_supression_file(numberofrepeat=100)
    agregateLaplaceandGaussian()
    agregatefileoriginalPrima()
    calculateAverageofelement()
    combining_averages()
    generate3DmetricAverages(metric="laplacian")
    generate3DmetricAverages(metric="gaussian")

def generateFileandGraphMelon():
    filedistanceL2(name_fruit="Melon")
    generate_files_m_M(path_average_distances="Melondistances.csv", path_of_file = "Files m and M\\Melon\\")
    generate_supression_file(path_m_M = "Files m and M\\Melon\\", fruit_name="Melon", numberofrepeat=100)
    agregateLaplaceandGaussian(file="Melon_onlysupression.csv", upper_weight= 2000, lower_weight= 1000, upper_volume= 300, lower_volume= 100)
    agregatefileoriginalPrima(fruit_name = "Melon", upper_weight=2000, lower_weight= 1000, upper_volume=300, lower_volume=100)
    calculateAverageofelement(file="File_graphic\\Melon_onlysupression.csv", File_name="File_graphic\\AverageMelon.csv")
    combining_averages(path_average_supression="File_graphic\\AverageMelon.csv", path_average_prima="File_graphic\\originalMelon prima.csv",
                       file="File_graphic\\CombiningMelon.csv", real_weight=1500, real_volume=200)
    generate3DmetricAverages(path = "File_graphic\\CombiningMelon.csv", metric="laplacian")
    generate3DmetricAverages(path = "File_graphic\\CombiningMelon.csv", metric="gaussian")
def generateFileandGraphStrawberry():
    filedistanceL2(name_fruit="Strawberry")
    generate_files_m_M(path_average_distances="Strawberrydistances.csv", path_of_file = "Files m and M\\Strawberry\\")
    generate_supression_file(path_m_M = "Files m and M\\Strawberry\\", fruit_name="Strawberry", numberofrepeat=100)
    agregateLaplaceandGaussian(file="Strawberry_onlysupression.csv", upper_weight= 45, lower_weight= 20, upper_volume= 30, lower_volume= 5)
    agregatefileoriginalPrima(fruit_name = "Strawberry", upper_weight=45, lower_weight= 20, upper_volume=30, lower_volume=5)
    calculateAverageofelement(file="File_graphic\\Strawberry_onlysupression.csv", File_name="File_graphic\\AverageStrawberry.csv")
    combining_averages(path_average_supression="File_graphic\\AverageStrawberry.csv", path_average_prima="File_graphic\\originalStrawberry prima.csv",
                       file="File_graphic\\CombiningStrawberry.csv", real_weight=30, real_volume=20)
    generate3DmetricAverages(path = "File_graphic\\CombiningStrawberry.csv", metric="laplacian")
    generate3DmetricAverages(path = "File_graphic\\CombiningStrawberry.csv", metric="gaussian")
def generateFileandGraphPear():
    filedistanceL2(name_fruit="Pear")
    generate_files_m_M(path_average_distances="Peardistances.csv", path_of_file = "Files m and M\\Pear\\")
    generate_supression_file(path_m_M = "Files m and M\\Pear\\", fruit_name="Pear", numberofrepeat=100)
    agregateLaplaceandGaussian(file="Pear_onlysupression.csv", upper_weight= 200, lower_weight= 90, upper_volume= 160, lower_volume= 80)
    agregatefileoriginalPrima(fruit_name = "Pear", upper_weight=200, lower_weight= 90, upper_volume=160, lower_volume=80)
    calculateAverageofelement(file="File_graphic\\Pear_onlysupression.csv", File_name="File_graphic\\AveragePear.csv")
    combining_averages(path_average_supression="File_graphic\\AveragePear.csv", path_average_prima="File_graphic\\originalPear prima.csv",
                       file="File_graphic\\CombiningPear.csv", real_weight=140, real_volume=120)
    generate3DmetricAverages(path = "File_graphic\\CombiningPear.csv", metric="laplacian")
    generate3DmetricAverages(path = "File_graphic\\CombiningPear.csv", metric="gaussian")
# generateFileandGraphMelon()
# generateFileandGraphPear()
# generateFileandGraphStrawberry()



