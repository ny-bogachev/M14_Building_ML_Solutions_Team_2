from prefect import flow
import os
from prefect_tasks import load_params, check_file_exists, download_data, preprocess_data, train_model

@flow
def data_pipeline(params_path: str):
    params = load_params(params_path)
    
    data_file_path = os.path.join(params['data_dir'], 'trips_2024_07_02.csv')
    
    file_exists = check_file_exists(data_file_path)
    
    if not file_exists:
        download_data(params)
    
    preprocessed_file_path = preprocess_data(params)
    train_model(preprocessed_file_path)

if __name__ == "__main__":
    data_pipeline(params_path='/home/abuzar/Harbour.space/M14_Building_ML_Solutions_Team_2/params.yaml')
