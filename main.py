import os
import numpy as np
from models import *
import tensorflow as tf

class configuration(object):
    def __init__(self):
        return None

def parser_helper(example_proto):
    dics = {'voice_embed': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'voice_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'anchor_embed': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'anchor_shape': tf.FixedLenFeature(shape=(2), dtype=tf.int64),
            'label': tf.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    voice_embed = tf.decode_raw(parsed_example['voice_embed'], tf.float32)
    voice_embed = tf.reshape(voice_embed, parsed_example['voice_shape'])
    anchor_embed = tf.decode_raw(parsed_example['anchor_embed'], tf.float32)
    anchor_embed = tf.reshape(anchor_embed, parsed_example['anchor_shape'])
    label = tf.cast(tf.sparse_tensor_to_dense(parsed_example['label']), tf.int32)
    voice_num = parsed_example['voice_shape'][0]
    anchor_num = parsed_example['anchor_shape'][0]
    return anchor_embed, voice_embed, label, anchor_num, voice_num

def dataset_prepare(tf_list, batch_size = 32):
    dataset = tf.data.TFRecordDataset(tf_list)  
    dataset = dataset.map(parser_helper)
    dataset = dataset.shuffle(1000)
    dataset = dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=batch_size,padded_shapes=([None,None],[None,None],[None],[],[])
        )
    )
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

def train(config):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7, allow_growth = True)
    sess_config = tf.ConfigProto(gpu_options = gpu_options)
    tf.reset_default_graph()
    """Train CAD model"""
    input_q = tf.placeholder(shape = [None, None, 64], dtype = tf.float32) # window-level acoustic feature from teacher
    input_d = tf.placeholder(shape = [None, None, 64], dtype = tf.float32) # window-level acoustic feature from classroom
    input_y = tf.placeholder(shape = [None, None], dtype = tf.float32) # label for the windows
    len_q = tf.placeholder(shape = [None], dtype = tf.int32) # valid window size of each sequence in batch
    len_d = tf.placeholder(shape = [None], dtype = tf.int32)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if config.model_type == 'average':
        embed_q, embed_d = input_q, input_d
    elif config.model_type == 'dnn':
        embed_q, embed_d = fc128(input_q, input_d)
    elif config.model_type == 'gru':
        embed_q, embed_d = bigru64(input_q, input_d, len_q, len_d)
    elif config.model_type == 'lstm':
        embed_q, embed_d = bilstm64(input_q, input_d, len_q, len_d)
    elif config.model_type == 'transformer':
        embed_q, embed_d = transformer(input_q, input_d, len_q, len_d)
    else:
        raise Exception("model type not allowed")

    embed_q = tf.reduce_mean(embed_q, axis = 1)[:,None,:] # average pooling for the embedings of teacher's windows
    
    att_output = tf.reshape(tf.matmul(embed_q, embed_d, transpose_b=True), shape=tf.shape(input_d)[:2])
    w = tf.get_variable(name = "w", initializer= tf.truncated_normal_initializer(), shape =[], dtype=tf.float32)
    b = tf.get_variable(name = "b", initializer= tf.constant_initializer(0.1), shape = [], dtype=tf.float32)
    logits = tf.abs(w) * att_output + b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = input_y)
    mask = tf.cast(tf.sequence_mask(len_d), tf.float32)
    accuracy = tf.cast(tf.equal(tf.cast(tf.sigmoid(logits) > 0.5, tf.float32),input_y), tf.float32)
    accuracy = tf.reduce_sum(accuracy * mask)/tf.cast(tf.reduce_sum(mask), tf.float32)

    trainable_vars= tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(lr)
    grads, vars = zip(*optimizer.compute_gradients(loss))
    train_op = optimizer.apply_gradients(zip(grads, vars), global_step= global_step)

    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables: %d" %variable_count)
    loss_summary = tf.summary.scalar("Loss", loss)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    merged_summary = tf.summary.merge([loss_summary, accuracy_summary])

    train_next = dataset_prepare(['data/train/train_1.tfrecord'])
    test_exist_next = dataset_prepare(['data/test_exist/exist_1.tfrecord'])
    test_new_next = dataset_prepare(['data/test_new/new_1.tfrecord'])
    
    with tf.Session(config = sess_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=999)
        os.makedirs(os.path.join(config.model_path, "Check_Point"), exist_ok=True)
        os.makedirs(os.path.join(config.model_path, "logs"), exist_ok=True)
        train_writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs/train"), sess.graph)
        test_exist_writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs/test_exist"), sess.graph)
        test_new_writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs/test_new"), sess.graph)
        
        train_loss_acc, test_exist_loss_acc, test_new_loss_acc = 0, 0, 0
        train_accuracy_acc, test_exist_accuracy_acc, test_new_accuracy_acc = 0, 0, 0
        lr_factor = 1
        for iter in range(config.iteration):
            query, doc, label, query_len, doc_len = sess.run(train_next)            
            train_loss, _, train_summary, train_accuracy = sess.run([loss, train_op, merged_summary, accuracy], feed_dict ={
                input_q:query, input_d:doc, input_y:label, len_q: query_len, 
                len_d: doc_len, lr: config.lr*lr_factor
            })
            train_loss_acc += train_loss
            train_accuracy_acc += train_accuracy
            
            query, doc, label, query_len, doc_len = sess.run(test_exist_next)
            test_exist_loss, test_exist_summary, test_exist_accuracy = sess.run([loss, merged_summary, accuracy], feed_dict={
                input_q:query, input_d:doc, input_y:label, len_q: query_len,
                len_d: doc_len
            })
            test_exist_loss_acc += test_exist_loss
            test_exist_accuracy_acc += test_exist_accuracy

            query, doc, label, query_len, doc_len = sess.run(test_new_next)
            test_new_loss, test_new_summary, test_new_accuracy = sess.run([loss, merged_summary, accuracy], feed_dict={
                input_q:query, input_d:doc, input_y:label, len_q: query_len,
                len_d: doc_len
            })

            test_new_loss_acc += test_new_loss
            test_new_accuracy_acc += test_new_accuracy

            if iter%10 == 0:
                train_writer.add_summary(train_summary, iter)
                test_exist_writer.add_summary(test_exist_summary, iter)
                test_new_writer.add_summary(test_new_summary, iter)
            
            if (iter+1)%100 == 0:
                info_content = """
                iter: {}
                train_loss: {} - test_exist_loss: {} - test_new_loss: {}
                train_accuracy: {} - test_exist_accuracy: {} - test_new_accuracy: {}
                """.format(
                    iter+1, round(train_loss_acc/100,4), round(test_exist_loss_acc/100,4), round(test_new_loss_acc/100,4),
                    round(train_accuracy_acc/100, 4), round(test_exist_accuracy_acc/100,4), round(test_new_accuracy_acc/100,4)
                )
                print(info_content)
                train_loss_acc, test_exist_loss_acc, test_new_loss_acc = 0, 0, 0
                train_accuracy_acc, test_exist_accuracy_acc, test_new_accuracy_acc = 0, 0, 0
            
            if (iter+1)%config.lr_decay_step == 0:
                lr_factor = lr_factor * 0.5
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)
            
            if (iter+1) % config.model_save_step == 0:
                saver.save(sess, os.path.join(config.model_path, "./Check_Point/model.ckpt"), global_step=iter//config.model_save_step)
                print("model is saved!")

if __name__ == "__main__":
    config = configuration()
    config.model_type = "average"
    config.batch_size = 128
    config.iteration = 20000
    config.lr = 0.001
    config.lr_decay_step = 10000
    config.model_save_step = 2000
    config.model_path = 'model/average'
    train(config)