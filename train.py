import tensorflow as tf
import yolo_net
import time
import datetime
import yolo_loss
import data_read
import os

@tf.function
def train_step(model, loss_func, images, obj_mask, obj_info, true_boxes, anchors):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_func([*obj_mask, *obj_info, true_boxes, *predictions], anchors)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


if __name__ == '__main__':
    model_train = yolo_net.yolo_v3()  # TODO define model
    loss_func = yolo_loss.yolo_loss()  # TODO define loss function
    if os.path.exists('./model.h5'):
        model_train.load_weights('./model.h5')
        print('load model weights')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # TODO define optimizer
    reader = data_read.DataReader(1, 416, 416,
                                  '/home/fengxh/kesci/a/data/train/image',
                                  '/home/fengxh/kesci/a/data/train/json',
                                  data_read.YOLO_ANCHORS)

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for i in range(100):
        images, obj_mask, obj_info, true_boxes = reader.get_format()
        loss_value, predictions_value = train_step(model_train,
                                                   loss_func,
                                                   images, obj_mask, obj_info, true_boxes,
                                                   data_read.YOLO_ANCHORS)
        print('training finish {}'.format(i), loss_value.numpy())
    model_train.save_weights('./model.h5')
