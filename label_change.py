import os
import cv2
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
from lxml import etree, objectify
import json
from xml.dom.minidom import Document
from pycocotools.coco import COCO
from PIL import Image
import shutil
from PIL import Image
from itertools import chain

def voc2yolo(root):
    trainval_percent = 0.9
    train_percent = 0.9
    xmlfilepath = root + '/Annotations'
    txtsavepath = root + '/ImageSets'
    os.makedirs(txtsavepath, exist_ok = True)

    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    for i in list:
        name = total_xml[i].split('.xml')[0] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    

    txtsavepath = root + '/labels'
    os.makedirs(txtsavepath, exist_ok = True)

    sets = ['train', 'test', 'val']

    for image_set in sets:
        image_ids = open(root + '/ImageSets/%s.txt' % (image_set)).read().strip().split()
        list_file = open(root + '/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(root + '/images/%s.jpg\n' %(image_id))
            convert_annotation(root, image_id)

def voc2coco(root):
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    category_set = dict()
    image_set = set()

    # (1) by txt
    json_save_path = root + '/COCO/voc2coco_txt.json'
    parseXmlFiles_by_txt(coco, image_set, category_set, root, json_save_path)
                          
    # # (2) by dir
    # ann_path = root + '/Annotations'
    # json_save_path = root + '/COCO/voc2coco.json'
    # parseXmlFiles(coco, image_set, category_set, ann_path, json_save_path)

def yolo2coco(root):
    jsonsavepath = root + '/COCO'
    os.makedirs(jsonsavepath, exist_ok = True)
    originImagesDir = root + '/images'
    originLabelsDir = root + '/labels'
    txtFileList = os.listdir(originLabelsDir)
    print(f"image number is {len(txtFileList)}")

    saveTempTxt = root + '/temp.txt'
    with open(saveTempTxt, 'w') as fw:
        for txtFile in tqdm(txtFileList, desc="generating COCO format"):
            imagePath = os.path.join(originImagesDir, txtFile.replace('txt', 'jpg'))
            assert os.path.exists(imagePath), f"can\'t find this image {imagePath}"
            image = cv2.imread(imagePath)
            H, W, _ = image.shape

            with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
                labelList = fr.readlines()
                for label in labelList:
                    label = label.strip().split()
                    x = float(label[1])
                    y = float(label[2])
                    w = float(label[3])
                    h = float(label[4])

                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * W
                    y1 = (y - h / 2) * H
                    x2 = (x + w / 2) * W
                    y2 = (y + h / 2) * H
                    fw.write(txtFile.replace('txt', 'jpg') + ' {} {} {} {} {}\n'.format(int(label[0]) + 1, x1, y1, x2, y2))
    fw.close()
    
    dataset = {'categories': [], 'annotations': [], 'images': []}

    classtxt = root + '/class.txt'
    with open(classtxt, 'r') as f:
        classes_ori = f.readlines()
        classes = [i.strip().split(' ') for i in classes_ori if i.strip() != '']

    for i, cls in classes:
        dataset['categories'].append({'id': i, 'name': cls})

    indexes = os.listdir(originImagesDir)

    anno_id = -1
    with open(saveTempTxt) as tr:
        annos = tr.readlines()
        with tqdm(total=len(indexes)) as pbar:
            for k, index in enumerate(indexes):
                im = cv2.imread(os.path.join(originImagesDir, index))
                assert im.all() != None, f"can\'t find this image {os.path.join(originImagesDir, index)}"
                height, width, _ = im.shape
                dataset['images'].append({'file_name': index,  'id': k, 'width': width, 'height': height})
                del_annos = []
                for anno in annos:
                    parts = anno.strip().split()
                    if parts[0] == index:
                        del_annos.append(anno)
                        anno_id += 1
                        cls_id = parts[1]
                        # x_min
                        x1 = float(parts[2])
                        # y_min
                        y1 = float(parts[3])
                        # x_max
                        x2 = float(parts[4])
                        # y_max
                        y2 = float(parts[5])
                        width = x2 - x1
                        height = y2 - y1
                        assert width > 0 and height > 0, f"width or height of {index}\'s box is not positive"
                        dataset['annotations'].append({
                            'area': width * height,
                            'bbox': [x1, y1, width, height],
                            'category_id': cls_id, 
                            'id': anno_id,
                            'image_id': k,
                            'iscrowd': 0,
                            'segmentation': [[x1, y1, x2, y2]]
                        })

                for da in del_annos:
                    annos.remove(da)
                pbar.update(1)

    if len(annos) != 0:
        print(f"\033[31m can\'t match image for these annotations:\n{annos}\033[0m")

    savefile = jsonsavepath + '/annotations.json'
    with open(savefile, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def yolo2voc(root):
    picPath = root + '/images/'
    txtPath = root + '/labels/'
    xmlPath = root + '/Annotations_yolo2voc/'
    os.makedirs(xmlPath, exist_ok = True)
    dic = {}
    with open(root + '/class.txt') as f:
        class_lines = f.readlines()
        for line in class_lines:
            class_id = line.split(' ')[0]
            class_name = line.split(' ')[1].replace('\n', '')
            dic[class_id] = class_name

    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)

        filename = xmlBuilder.createElement("filename")
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)

        size = xmlBuilder.createElement("size") 
        width = xmlBuilder.createElement("width")
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)

        height = xmlBuilder.createElement("height")
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)

        depth = xmlBuilder.createElement("depth")
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)

        annotation.appendChild(size)

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)

            pose = xmlBuilder.createElement("pose")
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)

            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)

            difficult = xmlBuilder.createElement("difficult")
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)

            bndbox = xmlBuilder.createElement("bndbox")
            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)

            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)

            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)

            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)

            object.appendChild(bndbox)

            annotation.appendChild(object)

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()

def coco2voc(root):
    # img.jpg
    origin_image_dir = root + '/images/'
    # annotations.json
    origin_anno_dir = root + '/COCO/'
    verbose = True
    get_CK5(root, origin_anno_dir, origin_image_dir, verbose)

def coco2yolo(root):
    json_path = root + '/COCO/annotations.json'
    txt_save_path = root + '/labels_coco2yolo/'
    os.makedirs(txt_save_path, exist_ok=True)
    parseJsonFile(json_path, txt_save_path)

def xyxy2xywhn(object, width, height):
    cat_id = object[0]
    xn = object[1] / width
    yn = object[2] / height
    wn = object[3] / width
    hn = object[4] / height
    out = "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(cat_id, xn, yn, wn, hn)
    return out

def save_anno_to_txt(images_info, save_path):
    filename = images_info['filename']
    txt_name = filename[:-3] + "txt"
    with open(os.path.join(save_path, txt_name), "w") as f:
        for obj in images_info['objects']:
            line = xyxy2xywhn(obj, images_info['width'], images_info['height'])
            f.write("{}\n".format(line))

def load_coco(anno_file, txt_save_path):
    if os.path.exists(txt_save_path):
        shutil.rmtree(txt_save_path)
    os.makedirs(txt_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()

    with open(os.path.join(txt_save_path, "classes.txt"), 'w') as f:
        for id in classesIds:
            f.write("{}\n".format(classes[id]))

    for imgId in tqdm(imgIds):
        info = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        width = img['width']
        height = img['height']
        info['filename'] = filename
        info['width'] = width
        info['height'] = height
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            bbox = list(map(float, ann['bbox']))
            xc = bbox[0] + bbox[2] / 2.
            yc = bbox[1] + bbox[3] / 2.
            w = bbox[2]
            h = bbox[3]
            obj = [ann['category_id'], xc, yc, w, h]
            objs.append(obj)
        info['objects'] = objs
        save_anno_to_txt(info, txt_save_path)

def parseJsonFile(json_path, txt_save_path):
    assert os.path.exists(json_path), "json path:{} does not exists".format(json_path)
    if os.path.exists(txt_save_path):
        shutil.rmtree(txt_save_path)
    os.makedirs(txt_save_path)
    assert json_path.endswith('json'), "json file:{} It is not json file!".format(json_path)
    load_coco(json_path, txt_save_path)

def save_annotations(root, filename, objs, filepath):
    CKanno_dir = root + '/Annotations_coco2voc'
    os.makedirs(CKanno_dir, exist_ok = True)
    annopath = CKanno_dir + "/" + filename[:-3] + "xml"
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
    print(f"save name {obj[0]}")

def showbycv(root, coco, dataType, img, classes, origin_image_dir, verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(root, filename, objs, filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

def catid2name(coco): 
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes

def get_CK5(root, origin_anno_dir, origin_image_dir, verbose=False):
    dataTypes = ['annotations']
    for dataType in dataTypes:
        # annFile = 'instances_{}.json'.format(dataType)
        annFile = '{}.json'.format(dataType)
        annpath = os.path.join(origin_anno_dir, annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(root, coco, dataType, img, classes, origin_image_dir, verbose=False)

def addCatItem(coco, category_set, category_item_id, name):
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id
 
def addImgItem(coco, image_set, image_id, file_name, size):
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    print(image_id)
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id
 
def addAnnoItem(coco, annotation_id, image_id, category_id, bbox):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
 
    annotation_item['segmentation'].append(seg)
 
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)
 
def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids
 
def parseXmlFiles_by_txt(coco, image_set, category_set, data_dir, json_save_path):
    image_id = 0
    category_id = 0
    sets = ['train', 'test', 'val']
    for image_set_name in sets:
        image_ids = open(data_dir + '/ImageSets/%s.txt' % (image_set_name)).read().strip().split()
        # labelfile=split + ".txt"
        # image_sets_file = data_dir + "/ImageSets/"+labelfile
        # ids=_read_image_ids(image_sets_file)
 
        for _id in image_ids:
            xml_file = data_dir + f"/Annotations/{_id}.xml"
    
            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None
    
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
    
            # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None
    
                if elem.tag == 'folder':
                    continue
    
                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in category_set:
                        raise Exception('file_name duplicated')
    
                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in image_set:
                        current_image_id = addImgItem(coco, image_set, image_id, file_name, size)
                        image_id = current_image_id
                        print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception('duplicated image: {}'.format(file_name))
                        # subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None
    
                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in category_set:
                            current_category_id = addCatItem(coco, category_set, category_id, object_name)
                            category_id = current_category_id
                        else:
                            current_category_id = category_set[object_name]
    
                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)
    
                    # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)
    
                    # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                        addAnnoItem(coco, current_category_id, current_image_id, current_category_id, bbox)
        json.dump(coco, open(json_save_path, 'w'), indent=2)
 
def parseXmlFiles(coco, image_set, category_set, xml_path, json_save_path):
    image_id = 0
    category_id = 0
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
 
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
 
        xml_file = os.path.join(xml_path, f) 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
 
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
 
            if elem.tag == 'folder':
                continue
 
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
 
            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(coco, image_set, image_id, file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                    image_id = current_image_id
                else:
                    # print(file_name)
                    continue
                    # raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None
 
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(coco, category_set, category_id, object_name)
                        category_id = current_category_id
                    else:
                        current_category_id = category_set[object_name]
 
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
 
                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)
 
                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(coco, current_category_id, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'), indent=2)

def convert(size, box):
    # size:(w,h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(root, image_id):
    classes = ['green_box']
    in_file = open(root + '/Annotations/%s.xml' %
                   (image_id), encoding='utf-8')
    # <object-class> <x> <y> <width> <height>
    out_file = open(root + '/labels/%s.txt' %
                    (image_id), 'w', encoding='utf-8')
    # load xml
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    if size != None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            if obj.find('difficult'):
                difficult = int(obj.find('difficult').text)
            else:
                difficult = 0
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert(root):
    jsonsFile = root + "/json"
    imgPath = root + "/src"
    destJson = root + "/json_COCO.json"

    imgs = os.listdir(imgPath)
    json_dict = {"images":[], "annotations": [], "categories": []}
    categories = {"green_box": 0}
    for cate, cid in categories.items():
        cat = {'id': cid , 'name': cate} # no + 1
        json_dict['categories'].append(cat)
 
    bnd_id = 1
    jsonnamelist = os.listdir(jsonsFile)
    jsonnamelist = [item for item in jsonnamelist if item[-4:] == 'json']
    for idx, jsonname in enumerate(tqdm(jsonnamelist)):
        image_id = idx + 1
        image_name = jsonname.replace(".json", ".jpg")
        if image_name not in imgs:
            with open('./error.txt', 'a') as target:
                info = f'No image file in image path:\n{jsonname} ==> {image_name}\n\n'
                target.write(info)
            continue
        img = Image.open(os.path.join(imgPath, image_name))
        width, height = img.size
 
        image = {'file_name': image_name, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)
 
        json_path = os.path.join(jsonsFile, jsonname)
        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)
            label = load_dict['shapes'][0]['label']
            if label not in categories.keys():
                new_id = len(categories)
                categories[label] = new_id+1
            category_id = categories[label]
 
            points = load_dict['shapes'][0]['points']
            pointsList = list(chain.from_iterable(points))
            pointsList = [float(p) for p in pointsList]
 
            seg = [pointsList]
 
            row = pointsList[0::2]
            clu = pointsList[1::2]
            left_top_x = min(row)
            left_top_y = min(clu)
            right_bottom_x = max(row)
            right_bottom_y = max(clu)
            wd = right_bottom_x - left_top_x
            hg = right_bottom_y - left_top_y
 
            ann = {'segmentation': seg, 'area': wd*hg, 'iscrowd': 0, 'image_id': image_id, 'bbox': [left_top_x, left_top_y, wd, hg], 'category_id': category_id, 'id': bnd_id}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    print(image_id, bnd_id)
 
    with open(destJson, 'w') as json_fp:
        json.dump(json_dict, json_fp, indent=2)


if __name__ == "__mai/n__":
    # root = './mask/label_mask_green'
    root = '/mnt/data/luoyan/road/BSUV'
    # root = './mask/mask_green'
    # voc2yolo(root)
    # voc2coco(root)
    # yolo2voc(root)
    # yolo2coco(root)
    # coco2voc(root)
    # coco2yolo(root)
    convert(root)