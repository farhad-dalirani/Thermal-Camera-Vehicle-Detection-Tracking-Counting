import json
import numpy as np 
import os
import shutil
import yaml
import cv2 as cv
import matplotlib.pyplot as plt

def create_folder(folder_path):
    """
    Create a new folder at the specified path or replace an existing one.
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If it exists, delete it and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted.")
    # Create it
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")

def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    """
    Converts a bounding box from COCO format to YOLO format.

    Parameters:
    - img_width (int): Width of the image.
    - img_height (int): Height of the image.
    - bbox (list[int]): Bounding box annotation in COCO format: [top-left x position, top-left y position, width, height].

    Returns:
    - list[float]: Bounding box annotation in YOLO format: [x_center_rel, y_center_rel, width_rel, height_rel].
    """
    # coco bounding box
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

if __name__ == '__main__':

    # Train, validation and test sequences
    splits = {
                'train': ['Egensevej','Hadsundvej','Hasserisvej','Hjorringvej','Hobrovej'],
                'valid': ['Ostre'], 
                'test': ['Ringvej']
            }

    # path of orginal aauRainSnow-dataset
    ds_org_path = './dataset/original-aauRainSnow-dataset'

    # path of converted dataset to Yolov7 PyTorch format
    ds_yolo_path = './dataset/yolov7-aauRainSnow-dataset'

    # Create a folder for the dataset in Yolov7 PyTroch format
    create_folder(folder_path=ds_yolo_path)
    create_folder(folder_path=os.path.join(ds_yolo_path, 'train'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'train/images'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'train/labels'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'valid'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'valid/images'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'valid/labels'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'test'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'test/images'))
    create_folder(folder_path=os.path.join(ds_yolo_path, 'test/labels'))

    # open thermal coco format json file for thermal camera
    with open(os.path.join(ds_org_path, 'aauRainSnow-thermal.json'), 'r') as json_file:
        # Load the JSON data
        data_coco = json.load(json_file)

    print(data_coco.keys())
    print('\n>  ', data_coco['info'])
    print('> Number of classes:', len(data_coco['categories']))
    print('> Number of images:', len(data_coco['images']))
    print('> Number of annotated bounding boxes:', len(data_coco['annotations']))

    # unique classes in the dataset
    unique_classes = []
    for ann_i in data_coco['annotations']:
        unique_classes.append(ann_i['category_id'])
    unique_classes = list(set(unique_classes)) 
    unique_classes.sort()
    unique_classes_with_name = [(cat_id, data_coco['categories'][cat_id-1]['name']) for cat_id in unique_classes]
    print('> Number of unique classes in orginal dataset:', len(unique_classes))
    print('> Unique classes in orginal dataset:', unique_classes)
    print('> Unique classes in orginal dataset:', unique_classes_with_name)

    # data.yaml file for yolo version of dataset
    yaml_file = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': len(unique_classes),
        'names': [cat_i[1] for cat_i in unique_classes_with_name]
    }
    with open(os.path.join(ds_yolo_path, 'data.yaml'), 'w') as file:
        yaml.dump(yaml_file, file, default_flow_style=None)

    # Dictionary to convert classes from orginal dataset to yolov7 format dataset
    dic_org_new_class = {}
    for idx, cat_i in enumerate(unique_classes_with_name):
        dic_org_new_class[cat_i[0]] = idx + 1
    print('> Class mapping from coco to yolo:', dic_org_new_class)    
    
    # Iterate bounding boxes in coco version one by one
    # and for each image group all annotation
    dic_img_annotations = {} 
    for ann_i in data_coco['annotations']:
        
        # Bounding box detail
        img_id = ann_i['image_id']
        cat_i = ann_i['category_id']
        bbox_i = ann_i['bbox']
        
        # Filter bounding boxes with negative width and height
        if bbox_i[2] <= 0 or bbox_i[3] <= 0:
            continue
        
        # Image detail
        img_width = data_coco['images'][img_id]['width']
        img_height = data_coco['images'][img_id]['height']
        img_file_name = data_coco['images'][img_id]['file_name']

        # Class label in the new dataset
        cat_i_new = dic_org_new_class[cat_i]        
        # Convert COCO bounding box to YOLO bounding box
        bbox_i_new = convert_bbox_coco_to_yolo(img_width=img_width, img_height=img_height, bbox=bbox_i)

        if img_id not in dic_img_annotations:
            dic_img_annotations[img_id] = (img_file_name, [])
        dic_img_annotations[img_id][1].append([cat_i_new, bbox_i_new]) 

    # Write annotations and images in yolo format
    for img_id_i in dic_img_annotations:
        print(img_id_i)
        # Image path
        img_i_path = dic_img_annotations[img_id_i][0]
        img_i_bboxes = dic_img_annotations[img_id_i][1]
        # Read the image
        img_i = cv.imread(os.path.join('dataset/original-aauRainSnow-dataset', img_i_path)) 
        # Read image mask
        seq_name = img_i_path.split('/')[0]
        img_name = img_i_path.split('/')[-1].split('.')[0]
        
        img_mask_i = cv.imread(os.path.join('dataset/original-aauRainSnow-dataset', seq_name, '{}-1-mask-thermal.png'.format(seq_name)))     
        # Mask image to remove not annotated area inside thermal camera images
        img_i[img_mask_i == 0] = 0

        # Determine write image and its annotation in train/validation or test
        for split_i in splits:
            if seq_name in splits[split_i]:                
                # Write image
                cv.imwrite(os.path.join('dataset/yolov7-aauRainSnow-dataset', split_i, 'images', '{}.png'.format(img_id_i)), img_i)
                # Write bounding boxes
                with open(os.path.join('dataset/yolov7-aauRainSnow-dataset', split_i, 'labels', '{}.txt'.format(img_id_i)), 'w') as file:
                    for idx_i, box_j in enumerate(img_i_bboxes):
                        # Write bounding box to the file
                        if idx_i != len(img_i_bboxes) - 1:
                            file.write('{} {} {} {} {}\n'.format(int(box_j[0]), box_j[1][0], box_j[1][1], box_j[1][2], box_j[1][3]))
                        else:
                            file.write('{} {} {} {} {}'.format(int(box_j[0]), box_j[1][0], box_j[1][1], box_j[1][2], box_j[1][3]))
                break

    print('Converting Dataset was finished.')


