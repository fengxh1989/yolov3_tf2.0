import cv2
import numpy as np
import os
import random
import json

classes_id = {'holothurian': 0, 'echinus': 1, 'scallop': 2, 'starfish': 3}

YOLO_ANCHORS = np.array(
    ((20, 20),
     (14, 28),
     (28, 14),
     (40, 40),
     (28, 56),
     (56, 28),
     (80, 80),
     (56, 113),
     (113, 56)), dtype=np.float32)


class DataReader(object):
    def __init__(self, batch_size, image_h, image_w, image_path, labels_path, anchors):
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_path = image_path
        self.labels_path = labels_path
        self.labels_sets = self.collect_data()
        self.data_length = len(self.labels_sets)
        self.flag = 0
        self.anchors = anchors
        self.branch_num = 3

    def collect_data(self):
        labels_list = os.listdir(self.labels_path)
        random.seed(1000)
        random.shuffle(labels_list)
        return labels_list

    def get_image(self, image_name):
        img = cv2.imread(os.path.join(self.image_path, image_name))
        # cv2.namedWindow('src', cv2.WINDOW_NORMAL)
        # cv2.imshow('src', img)
        img_resize = cv2.resize(img, (self.image_w, self.image_h))
        # cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
        # cv2.imshow('dst', img_resize)
        # cv2.waitKey()
        img_resize = img_resize.astype(np.float32)/255
        return img_resize

    def keep_regulation(self, list_var):
        max_num = 0
        for each_layer in list_var:
            num = len(each_layer)
            if num > max_num:
                max_num = num
        for index, each_layer in enumerate(list_var):
            num = len(each_layer)
            for i in range(max_num - num):
                list_var[index].append([0, 0, 0, 0, 0])
        return list_var

    def get_format(self):
        images = np.zeros(shape=(self.batch_size, self.image_h, self.image_w, 3), dtype=np.float32)

        grid_shapes = [np.array([52, 52]),
                       np.array([26, 26]),
                       np.array([13, 13])]

        obj_mask = [np.zeros(shape=(self.batch_size, 52, 52, 3, 1), dtype=np.float32),
                    np.zeros(shape=(self.batch_size, 26, 26, 3, 1), dtype=np.float32),
                    np.zeros(shape=(self.batch_size, 13, 13, 3, 1), dtype=np.float32)]

        obj_info = [np.zeros(shape=(self.batch_size, 52, 52, 3, 5), dtype=np.float32),
                    np.zeros(shape=(self.batch_size, 26, 26, 3, 5), dtype=np.float32),
                    np.zeros(shape=(self.batch_size, 13, 13, 3, 5), dtype=np.float32)]

        true_boxes = []

        for num in range(self.batch_size):
            if self.flag >= self.data_length:
                self.flag = 0
            label_name = self.labels_sets[self.flag]
            label_path = os.path.join(self.labels_path, label_name)
            with open(label_path) as fp:
                data = json.load(fp)
            img = self.get_image(data['img_path'])
            images[num, ...] = img
            objects = data['object']
            src_w = data['width']
            src_h = data['height']
            true_box = [[], [], []]
            for obj in objects:
                x1 = obj['x1']
                y1 = obj['y1']
                x2 = obj['x2']
                y2 = obj['y2']
                obj_classes = obj['class']
                class_index = classes_id[obj_classes]
                obj_w = x2-x1+1
                obj_h = y2-y1+1
                obj_cx = x1 + obj_w / 2
                obj_cy = y1 + obj_h / 2

                obj_w = obj_w/src_w
                obj_h = obj_h/src_h
                obj_cx = obj_cx/src_w
                obj_cy = obj_cy/src_h
                bbox_norm = np.array([obj_cx, obj_cy, obj_w, obj_h, class_index])
                bbox = np.array([obj_cx*self.image_w, obj_cy*self.image_h, obj_w*self.image_w, obj_h*self.image_h,
                                 class_index])

                best_iou = 0
                best_anchor = 0
                for k, each_anchor in enumerate(self.anchors):
                    box_max = bbox[2:4] / 2
                    box_min = -box_max
                    anchor_max = each_anchor / 2
                    anchor_min = -anchor_max
                    intersect_min = np.maximum(box_min, anchor_min)
                    intersect_max = np.minimum(box_max, anchor_max)
                    intersect_wh = np.maximum(intersect_max - intersect_min, 0)
                    intersect_area = intersect_wh[0] * intersect_wh[1]
                    box_area = bbox[2] * bbox[3]
                    anchor_area = each_anchor[0] * each_anchor[1]
                    iou = intersect_area / (box_area + anchor_area - intersect_area)
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = k
                if best_iou > 0:
                    branch_index = best_anchor // self.branch_num
                    true_box[branch_index].append([obj_cx, obj_cy, obj_w, obj_h, class_index])
                    output_locat_x = np.floor(bbox_norm[0] * grid_shapes[branch_index][0]).astype(
                        np.int)  # grid cell columns
                    output_locat_y = np.floor(bbox_norm[1] * grid_shapes[branch_index][1]).astype(
                        np.int)  # grid cell rows
                    obj_mask[branch_index][num, output_locat_y, output_locat_x,
                                                best_anchor - branch_index * self.branch_num, 0] = 1
                    obj_info[branch_index][
                        num, output_locat_y, output_locat_x, best_anchor - branch_index * self.branch_num] = np.array([
                        bbox_norm[0] * grid_shapes[branch_index][0] - output_locat_x,
                        bbox_norm[1] * grid_shapes[branch_index][1] - output_locat_y,
                        np.log(bbox_norm[2] / self.anchors[best_anchor][0] * self.image_w),
                        np.log(bbox_norm[3] / self.anchors[best_anchor][1] * self.image_h),
                        class_index])

            true_box = self.keep_regulation(true_box)
            true_boxes.append(true_box)
            self.flag += 1

        max_box_num = 0
        for each_box in true_boxes:
            box_num = len(each_box[0])
            if box_num > max_box_num:
                max_box_num = box_num
        for i, each_box in enumerate(true_boxes):
            box_num = len(each_box[0])
            for k in range(len(each_box)):
                for j in range(max_box_num - box_num):
                    true_boxes[i][k].append([0, 0, 0, 0, 0])
        true_boxes = np.array(true_boxes, dtype=np.float32)
        # print("Extract batch complete")
        return images, obj_mask, obj_info, true_boxes


if __name__ == '__main__':
    reader = DataReader(10, 416, 416, '/home/fengxh/kesci/a/data/train/image', '/home/fengxh/kesci/a/data/train/json',
                        YOLO_ANCHORS)
    reader.get_format()
    print('tttttttttt')

