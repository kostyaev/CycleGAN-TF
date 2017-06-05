import tensorflow as tf

def mae(pred_y, true_y):
    return tf.reduce_mean((true_y - pred_y) ** 2)

def abs_criterion(pred_y, true_y):
    return tf.reduce_mean(tf.abs(pred_y - true_y))