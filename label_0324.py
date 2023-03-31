import cv2
import numpy as np
import os
import json
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

if __name__ == "__main__":
    # root = './0324-1/day/BJTP'

    json_out = '/mnt/data/luoyan/BSUV/dataset/day/json'
    txt_path = '/mnt/data/luoyan/BSUV/dataset/day.txt'
    xml_path = '/mnt/data/luoyan/BSUV/dataset/day/voc'

    # json_out = '/mnt/data/luoyan/BSUV/dataset/night/json'
    # txt_path = '/mnt/data/luoyan/BSUV/dataset/night.txt'
    # xml_path = '/mnt/data/luoyan/BSUV/dataset/night/voc'
    lnum = 0

    with open(txt_path, 'r') as fd:
        for line in fd:
            if 'BJSJ' in line:
                fq_json = {
                    "version": "0.0.1",
                    "flags": {},
                    "shapes": [
                        {
                        "label": "1",
                        "points": [],
                        "group_id": '1',
                        "shape_type": "Polygon",
                        "flags": {}
                        }
                    ],
                    "imagePath": "",
                    "imageData":"",
                    "imageHeight": 1088,
                    "imageWidth": 1920
                }
                info = line.split('message ')[1]
                re_list = ['{', '}', ' ']
                for s in info:
                    if s in re_list:
                        info = info.replace(s, '')
                info = info.split("JKFQ':'")[1]
                fq_points = info.split(",-1")[0].split(',')
                for i in range(len(fq_points)):
                    if i+1 >= 2 and (i+1) % 2 == 0:
                        fq_json['shapes'][0]['points'].append([float(fq_points[i-1]), float(fq_points[i])])
                info = info.split('BJBJ')[1]
                img_name = info.split("','")[0].split('/')[-1].replace('first', 'old') 
                # JKFQ
                json_name = img_name.split('.jpg')[0] + '.json'
                fq_json['imagePath'] = img_name            
                json_out_path = os.path.join(json_out, json_name)
                with open(json_out_path, 'w') as f:
                    json.dump(fq_json, f, indent=2)
                
                width = 1920
                height = 1088
                channel = 3
        
                # # BJXX
                node_root = Element('annotation')
                node_folder = SubElement(node_root, 'folder')
                node_folder.text = 'JPEGImages'
                node_filename = SubElement(node_root, 'filename')
                node_filename.text = img_name

                node_size = SubElement(node_root, 'size')
                node_width = SubElement(node_size, 'width')
                node_width.text = '%s' % width

                node_height = SubElement(node_size, 'height')
                node_height.text = '%s' % height

                node_depth = SubElement(node_size, 'depth')
                node_depth.text = '%s' % channel

                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = 'green_box'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                
                # print(info)
                info = info.split("callEvent")[0].split(",'")
                for box_info in info:
                    if "box" in box_info:
                        box_info = box_info.split("'")[2].split(',')
                        # print(box_info)
                        left, top, right, bottom = box_info[0], box_info[1], box_info[-2], box_info[-1]
                        
                        node_bndbox = SubElement(node_object, 'bndbox')
                        node_xmin = SubElement(node_bndbox, 'xmin')
                        node_xmin.text = '%s' % left
                        node_ymin = SubElement(node_bndbox, 'ymin')
                        node_ymin.text = '%s' % top
                        node_xmax = SubElement(node_bndbox, 'xmax')
                        node_xmax.text = '%s' % right
                        node_ymax = SubElement(node_bndbox, 'ymax')
                        node_ymax.text = '%s' % bottom

                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)

                save_xml = os.path.join(xml_path, img_name.replace('jpg', 'xml'))
                with open(save_xml, 'wb') as f:
                    f.write(xml)


    
    fd.close()
