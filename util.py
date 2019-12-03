import sys
import cv2


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)


def get_data(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            #line_split = line.strip().split(',')

            #(filename, y1, x1, y2, x2, class_name) = line_split
            print(line)
            filename = line.split()[0]
            y1, x1, y2, x2, class_name = line.split()[1].split(',')

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
            # if np.random.randint(0,6) > 0:
            # 	all_imgs[filename]['imageset'] = 'trainval'
            # else:
            # 	all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

import xmltodict
import os
from os.path import join
def get_data_voc(VOCdevkit_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1
    
    for dir in os.listdir(VOCdevkit_path):
        print("Loading "+dir)
        curr_folder = join(VOCdevkit_path, dir)
        annotation_folder = join(curr_folder, "Annotations")
        image_folder = join(curr_folder, "JPEGImages")
        for xml in os.listdir(annotation_folder):
            with open(join(annotation_folder, xml)) as fp:
                raw_xml = fp.read()
                anno_dict = xmltodict.parse(raw_xml)["annotation"]
                filename = anno_dict["filename"]
                all_imgs[filename] = {}
                all_imgs[filename]['filepath'] = join(image_folder, filename)
                all_imgs[filename]['width'] = anno_dict["size"]["width"]
                all_imgs[filename]['height'] = anno_dict["size"]["height"]     
                objects = anno_dict["object"]
                if not isinstance(objects, list):
                    objects = [objects]
                all_imgs[filename]['bboxes'] = [{'class': bbox["name"], 
                                                 'x1': int(float(bbox["bndbox"]["xmin"])), 
                                                 'x2': int(float(bbox["bndbox"]["xmax"])), 
                                                 'y1': int(float(bbox["bndbox"]["ymin"])), 
                                                 'y2': int(float(bbox["bndbox"]["ymax"]))} for bbox in objects]
                for bbox in all_imgs[filename]['bboxes']:
                    class_name = bbox["class"]
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1
                        
                    if class_name not in class_mapping:
                        if class_name == 'bg' and found_bg == False:
                            print('Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                            found_bg = True
                        class_mapping[class_name] = len(class_mapping)
                
        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        
        return all_data, classes_count, class_mapping

def get_data_sign(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            #line_split = line.strip().split(',')

            #(filename, y1, x1, y2, x2, class_name) = line_split
            print(line)
            line = line.split()
            filename = line[0]

            for bbox in line[1:]:
                y1, x1, y2, x2, class_name = bbox.split(',')

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and found_bg == False:
                        print(
                            'Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    all_imgs[filename] = {}

                    #img = cv2.imread(filename)
                    #(rows, cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    #all_imgs[filename]['width'] = cols
                    #all_imgs[filename]['height'] = rows
                    all_imgs[filename]['width'] = 1080
                    all_imgs[filename]['height'] = 1920
                    all_imgs[filename]['bboxes'] = []
                # if np.random.randint(0,6) > 0:
                # 	all_imgs[filename]['imageset'] = 'trainval'
                # else:
                # 	all_imgs[filename]['imageset'] = 'test'

                all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

if __name__ == "__main__":
    data_set = get_data_sign("MonoImagesTrainBB.txt")
    
    