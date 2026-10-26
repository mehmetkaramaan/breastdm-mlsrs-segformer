import tensorflow as tf

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)


def total_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    return cce + dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred):
    yt = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    yp = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    intersection = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp)
    return (2.0 * intersection + 1e-6) / (union + 1e-6)


def iou(y_true, y_pred):
    yt = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    yp = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    intersection = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - intersection
    return intersection / (union + 1e-6)
