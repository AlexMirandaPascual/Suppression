from function_algoritmia import *
from graphic_generator import *
from generate_distance import *
import os

# generate_average_distance_list(filename="irishn_train.csv", column_name_1="Age", column_name_2="HighestEducationCompleted")
# generate_files_m_M(m=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                         path_average_distances="irishn_train distances.csv",
#                         path_of_file=os.path.join("Files m and M","Irish Age-HEC"))
generate_df_with_m_and_M(probabilities_path=os.path.join("Files m and M"), file_name="m_M error.csv",  column_name_1="Age", column_name_2="HighestEducationCompleted", k=5, number_of_repeat=100)
generate_error_original_file(path_file_original="irishn_train.csv", file_to_data="original_error", column_name_1="Age", column_name_2="HighestEducationCompleted", k=5)
generate3Dmetric2D()