
import os
import json
from zipfile import ZipFile
def Generate_DirctoryStructure():
    root_dir = './data/CarSeg_Data'

    # Create the root directory
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    # Create the images directory
    images_dir = os.path.join(root_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    # Create the train and val subdirectories in the images directory
    train_images_dir = os.path.join(images_dir, 'train')
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    val_images_dir = os.path.join(images_dir, 'val')
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)
    val_images_dir = os.path.join(images_dir, 'test')
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)

    # Create the labels directory
    labels_dir = os.path.join(root_dir, 'labels')
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Create the train and val subdirectories in the labels directory
    train_labels_dir = os.path.join(labels_dir, 'train')
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    val_labels_dir = os.path.join(labels_dir, 'val')
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)
    val_images_dir = os.path.join(labels_dir, 'test')
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)


def data_rename(dirs_path,yolo_directory):
    for path in dirs_path:
        files = os.listdir(path)
        for file in files :

            old_path = os.path.join(path, file)
            if 'train' in old_path:
                new_path = os.path.join(yolo_directory,'train',file.strip('0') )
                os.rename(old_path, new_path)
            elif 'val' in old_path:
                new_path = os.path.join(yolo_directory,'val',file.strip('0') )               
                os.rename(old_path, new_path)
            elif 'test' in old_path:
                new_path = os.path.join(yolo_directory,'test',file.strip('0') )                
                os.rename(old_path, new_path)


def read_json(json_path,data_type,label_path):
    file_path = os.path.join(label_path,data_type)
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Create a set of image IDs
    image_ids = set(json_file['image_id'] for json_file in data['annotations'])

    for img_id in image_ids:
        info = {}
        for json_file in data['annotations']:
            if json_file['image_id'] == img_id:
                file = {}
                for image_name in data['images']:
                    if image_name['id']== img_id:
                        file['width'] =image_name['width']
                        file['height'] = image_name['height']
                file['category_id'] = json_file['category_id']
                file['segmentation'] = json_file['segmentation'][0] 
                info[json_file['id']] = file
                file_path = os.path.join(label_path,data_type,'{}.txt'.format(img_id))
                with open(file_path, 'a') as f:
                        for i in info.values():
                            new_list = [i['category_id']]
                            new_list.extend(i['segmentation'])
                            result = []
                            for j in range(1,len(new_list[1:])+1):
                                if j%2 ==0:
                                    result.append(new_list[j]/file['height'])
                                else:
                                    result.append(new_list[j]/file['width'])
                            result.insert(0,i['category_id'])
                        f.write(' '.join(map(str, result)) + '\n')
def Generate_YOLOv5dataset():
    IMAGE_ARCHIVE_NAME = 'images'
    YOLOv5_Dataset = 'CarSeg_data'
    data_dir = './data'
    data_zip = os.path.join(data_dir, IMAGE_ARCHIVE_NAME + '.zip')
    image_dir = os.path.join(data_dir, IMAGE_ARCHIVE_NAME)
    if not os.path.exists(image_dir):
        with ZipFile(data_zip, 'r') as z_object:
           z_object.extractall(data_dir)
    
    YoloImage_dir = os.path.join(data_dir,YOLOv5_Dataset,IMAGE_ARCHIVE_NAME)
    YoloIabels_dir = os.path.join(data_dir,YOLOv5_Dataset,'labels')
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')
    val_dir = os.path.join(image_dir, 'val')
    annot_dir = os.path.join(image_dir, 'annotations')
    data_dir = [train_dir,val_dir,test_dir]
    yolo_dit = os.path.join(YoloImage_dir)
    data_rename(data_dir,yolo_dit)
    json_files = os.listdir(annot_dir)
    for json_file in json_files :
        data_type =json_file.split('.')[0] 
        json_path = os.path.join(annot_dir,json_file)
        read_json(json_path,data_type,YoloIabels_dir)


def main():
    Generate_DirctoryStructure()
    Generate_YOLOv5dataset()
    print('finish')
if __name__ == '__main__':
    main()