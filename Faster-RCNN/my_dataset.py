from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

class VOCDataset(Dataset):
    """
        VOC数据集的结构：
        VOCdevkit
        --VOC2012
        ----Annotations: 放入的所有的xml文件
        ----ImageSets
        ------Main     : 放入train.txx, val.txt文件
        ----JPEGImages : 放入所有的图片文件
    """
    def __init__(self, voc_root, transforms, voc_year='VOC2012', txt_name="train.txt"):
        self.root = os.path.join(voc_root, "VOCdevkit", voc_year)
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotation_root = os.path.join(self.root, "Annotations")

        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file".format(txt_name)

        with open(txt_path) as f:
            self.xml_list = [os.path.join(self.annotation_root, line.strip() + '.xml')
                             for line in f.readlines()]

        assert len(self.xml_list) > 0, f"in {txt_path} file dose not find any information."
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), f"not found '{xml_path}' file"

        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), f"{json_file} file not found"
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        xml_path = self.xml_list[index]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError(f"Image '{img_path} format not JPEG'")

        boxes = []
        labels = []
        is_crowd = []
        assert 'object' in data, f"{xml_path} not object information"
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print(f"bndbox w/h <= 0 in {xml_path} xml")
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if 'difficult' in obj:
                is_crowd.append(int(obj['difficult']))
            else:
                is_crowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        is_crowd = torch.as_tensor(is_crowd, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['is_crowd'] = is_crowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for sub in xml:
            sub_result = self.parse_xml_to_dict(sub)
            if sub.tag != 'object':
                result[sub.tag] = sub_result[sub.tag]
            else:
                if sub.tag not in result:
                    result[sub.tag] = []
                result[sub.tag].append(sub_result[sub.tag])
        return {xml.tag: result}

    def get_height_and_width(self, index):
        xml_path = self.xml_list[index]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])
        data_width = int(data['size']['width'])
        return data_height, data_width

    def coco_index(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        is_crowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            is_crowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        is_crowd = torch.as_tensor(is_crowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["is_crowd"] = is_crowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    # [(A, 1), (B, 2), (C, 3)] -> [(A, B, C), (1, 2, 3)]


# 测试
if __name__ == "__main__":
    import transforms
    from draw_bbox import draw_bbox
    from PIL import Image
    import json
    import matplotlib.pyplot as plt
    import torchvision.transforms as ts
    import numpy as np
    import random
    category_index = {}
    try:
        json_file = open('pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        print(class_dict)
        category_index = {v: k for k, v in class_dict.items()}
        print(category_index)
    except Exception as e:
        print(e)
        exit(-1)

    data_transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        'val': transforms.Compose([transforms.ToTensor()])
    }

    train_data_set = VOCDataset(os.getcwd(), data_transform['train'], voc_year='VOC2007', txt_name='train.txt')
    print(len(train_data_set))
    for index in random.sample(range(0, len(train_data_set)), k=5):
        print(index)
        img, target = train_data_set[index]
        print(target)
        img = ts.ToPILImage()(img)
        draw_bbox(img, target['boxes'].numpy(),
                  target['labels'].numpy(),
                  [1 for i in range(len(target['labels'].numpy()))],
                  category_index,
                  thresh=0.5,
                  line_thickness=1)
        plt.imshow(img)
        plt.show()








