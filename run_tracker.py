import json
from tracking_by_detection import tracking_by_detection

if __name__ == '__main__':
    
    # Open the 'config_file_path.json' file that contain
    # path of config file for object detector, tracker, etc.
    with open('config_file_path.json', 'r') as file:
        path_config = json.load(file)
    
    # Open the config file
    with open(path_config['config_file_path'], 'r') as file:
        dic_config = json.load(file)

    # Tracking by detection
    tracking_by_detection(config=dic_config)
