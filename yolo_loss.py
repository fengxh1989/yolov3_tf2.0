import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Lambda
import numpy as np
import data_read
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import losses_utils


def yolo_parse_output(yolo_output=None, anchors=None, num_classes=7, input_shape=None):
    anchor_reshape = tf.reshape(anchors, shape=(1, 1, 1, tf.shape(anchors)[0], 2))
    anchor_reshape = tf.cast(anchor_reshape, dtype=K.dtype(yolo_output))
    output_shape = tf.shape(yolo_output)
    height_index = K.arange(0, stop=output_shape[1])
    width_index = K.arange(0, stop=output_shape[2])
    tmp1, tmp2 = tf.meshgrid(height_index, width_index)
    conv_index = tf.reshape(tf.concat([tmp1, tmp2], axis=0), (2, output_shape[1], output_shape[2]))
    conv_index = tf.transpose(conv_index, (1, 2, 0))
    conv_index = K.expand_dims(K.expand_dims(conv_index, 0), -2)  # shape will be (1, 13, 13, 1, 2)
    conv_index = K.cast(conv_index, K.dtype(yolo_output))

    yolo_output = tf.reshape(yolo_output, shape=(-1, output_shape[1], output_shape[2], tf.shape(anchors)[0], 5 + num_classes))
    box_xy = yolo_output[..., :2]
    box_wh = yolo_output[..., 2:4]
    box_confidence = yolo_output[..., 4:5]
    box_classes = yolo_output[..., 5:]
    box_xy_sig = tf.sigmoid(box_xy)
    box_wh_coord = box_wh
    box_xy = (box_xy_sig + conv_index) / tf.cast(output_shape[1:3], dtype=K.dtype(yolo_output))
    box_wh = tf.exp(box_wh) * anchor_reshape / tf.cast(input_shape, dtype=K.dtype(yolo_output))
    box_confidence = tf.sigmoid(box_confidence)
    box_classes = tf.sigmoid(box_classes)

    box_coord = K.concatenate((box_xy_sig, box_wh_coord), axis=-1)
    return box_xy, box_wh, box_confidence, box_classes, box_coord


def loss_layer(x, anchors, num_classes=4, input_shape=(416, 416)):
    in_shape = tf.constant(input_shape, dtype=tf.float32)
    # anchors = K.constant(anchors)
    detectors_mask = x[:3]
    matching_true_boxes = x[3:6]
    true_box_all = x[6]
    ys = x[7:10]
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    m = K.shape(ys[0])[0]  # batch size, tensor
    loss = 0
    mf = K.cast(m, K.dtype(ys[0]))
    for index, y in enumerate(ys):
        box_xy_predict, box_wh_predict, box_confidence_predict, box_classes_predict, box_coord_predict = \
            yolo_parse_output(y, anchors=anchors[(index*3):(index*3+3), :], input_shape=in_shape, num_classes=num_classes)
        box_xy_predict = K.expand_dims(box_xy_predict, 4)
        box_wh_predict = K.expand_dims(box_wh_predict, 4)
        pred_wh_half = box_wh_predict / 2.
        pred_mins = box_xy_predict - pred_wh_half
        pred_maxes = box_xy_predict + pred_wh_half
        true_box = true_box_all[:, index, :, :]
        true_boxes_shape = K.shape(true_box)

        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
        true_boxes = K.reshape(true_box, [
            true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
        ])
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        # Find IOU of each predicted box with each ground truth box.
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_areas = box_wh_predict[..., 0] * box_wh_predict[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        # Best IOUs for each location.
        best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
        best_ious = K.expand_dims(best_ious)
        object_detections = K.cast(best_ious > 0.4, dtype=K.dtype(best_ious))
        no_object_weights = (no_object_scale * (1 - object_detections) * (1 - detectors_mask[index]))
        no_objects_loss = no_object_weights*K.binary_crossentropy(detectors_mask[index],  box_confidence_predict, from_logits=False)
        object_loss = object_scale * detectors_mask[index] * K.binary_crossentropy(detectors_mask[index],  box_confidence_predict, from_logits=False)
        xy_loss = (coordinates_scale * detectors_mask[index] * K.binary_crossentropy(matching_true_boxes[index][..., 0:2], box_coord_predict[..., 0:2], from_logits=False))
        wh_loss = (coordinates_scale * detectors_mask[index] * K.square(matching_true_boxes[index][..., 2:4] - box_coord_predict[..., 2:4]))
        matching_classes = K.cast(matching_true_boxes[index][..., 4], 'int32')
        matching_classes = K.one_hot(matching_classes, num_classes)
        classification_loss = (class_scale * detectors_mask[index] * K.binary_crossentropy(matching_classes,  box_classes_predict, from_logits=False))
        no_objects_loss = K.sum(no_objects_loss)
        object_loss = K.sum(object_loss)
        xy_loss = K.sum(xy_loss)
        wh_loss = K.sum(wh_loss)
        classification_loss = K.sum(classification_loss)
        total_loss = no_objects_loss+object_loss+xy_loss+wh_loss+classification_loss
        loss += total_loss
    losses = loss/mf
    return losses


class yolo_loss(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='yolo_loss'):
    super(yolo_loss, self).__init__(
        loss_layer,
        name=name,
        reduction=reduction)


if __name__ == '__main__':
    y1 = tf.random.normal((10, 13, 13, 27))
    y2 = tf.random.normal((10, 26, 26, 27))
    y3 = tf.random.normal((10, 52, 52, 27))
    ys = [y3, y2, y1]
    true_boxes = tf.random.normal((10, 3, 2, 5))
    obj_mask = [tf.random.normal((10, 52, 52, 3, 1)),
                tf.random.normal((10, 26, 26, 3, 1)),
                tf.random.normal((10, 13, 13, 3, 1))]
    obj_info = [tf.random.normal((10, 52, 52, 3, 5)),
                tf.random.normal((10, 26, 26, 3, 5)),
                tf.random.normal((10, 13, 13, 3, 5))]
    loss = yolo_loss()
    loss_value = loss([*obj_mask, *obj_info, true_boxes, *ys], data_read.YOLO_ANCHORS)
    print(loss_value)
