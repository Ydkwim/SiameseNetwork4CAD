import tensorflow as tf
from modeling import gelu
from modeling import transformer_model
from modeling import create_attention_mask_from_input_mask

def bigru(inputs, units, seq_len, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_reversed = tf.reverse_sequence(inputs, seq_len, name='reverse_data', seq_dim=1, batch_dim=0)
        gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=units)
        outputs_fw, _ = tf.nn.dynamic_rnn(cell = gru_fw_cell, inputs = inputs, sequence_length=seq_len, time_major=False, dtype=tf.float32)

        gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=units)
        outputs_bw, _ = tf.nn.dynamic_rnn(cell = gru_bw_cell, inputs = inputs_reversed, sequence_length=seq_len, time_major=False, dtype=tf.float32)
        
        outputs_bw_reverse = tf.reverse_sequence(outputs_bw, seq_len, name='reverse_data', seq_dim=1, batch_dim=0)
        outputs = tf.concat([outputs_fw, outputs_bw_reverse], -1)
    return outputs

def bilstm(inputs, units, seq_len, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_reversed = tf.reverse_sequence(inputs, seq_len, name='reverse_data', seq_dim=1, batch_dim=0)
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_units=units, use_peepholes=True)
        outputs_fw, _ = tf.nn.dynamic_rnn(cell = lstm_fw_cell, inputs = inputs, sequence_length=seq_len, time_major=False, dtype=tf.float32)

        lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_units=units, use_peepholes=True)
        outputs_bw, _ = tf.nn.dynamic_rnn(cell = lstm_bw_cell, inputs = inputs_reversed, sequence_length=seq_len, time_major=False, dtype=tf.float32)
        
        outputs_bw_reverse = tf.reverse_sequence(outputs_bw, seq_len, name='reverse_data', seq_dim=1, batch_dim=0)
        outputs = tf.concat([outputs_fw, outputs_bw_reverse], -1)
    return outputs

def fc128(input_q, input_d):
    """
    input_q: window-level acoustic features of the teahcer
    input_d: window-level acoustic features of classroom recording
    """
    embed_q = tf.layers.dense(
        inputs=input_q,units=128,activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(),
        name='q_map')
    embed_d = tf.layers.dense(
        inputs=input_d,units=128,activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(),
        name='d_map')
    return embed_q, embed_d

def bigru64(input_q, input_d, len_q, len_d):
    """
    input_q: window-level acoustic features of the teahcer
    input_d: window-level acoustic features of classroom recording
    len_q: number of valid windows of input_q for each sequence in batch
    len_d: number of valid windows of input_d for each sequence in batch
    """
    embed_q = bigru(
        inputs=input_q,units=64,seq_len=len_q,
        scope="q_bigru")
    embed_d = bigru(
        inputs=input_d,units=64,seq_len=len_d, 
        scope="d_bigru")
    return embed_q, embed_d

def bilstm64(input_q, input_d, len_q, len_d):
    embed_q = bilstm(
        inputs=input_q,units=64,seq_len=len_q, 
        scope="q_bilstm")
    embed_d = bilstm(
        inputs=input_d,units=64,seq_len=len_d, 
        scope="d_bilstm")
    return embed_q, embed_d

def transformer(input_q, input_d, len_q, len_d):
    """Use the transformer code from google BERT
    """
    with tf.variable_scope("embed_q"):
        raw_mask_q = tf.cast(tf.sequence_mask(len_q), tf.float32)
        attention_mask_q = create_attention_mask_from_input_mask(
            from_tensor=input_q,
            to_mask=raw_mask_q
        )
        embed_q_all = transformer_model(
            input_tensor=input_q,
            attention_mask=attention_mask_q,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=128,
            intermediate_act_fn=gelu,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            do_return_all_layers=True
        )
        embed_q = embed_q_all[-1]

    with tf.variable_scope("embed_d"):
        raw_mask_d = tf.cast(tf.sequence_mask(len_d), tf.float32)
        attention_mask_d = create_attention_mask_from_input_mask(
            from_tensor=input_d,
            to_mask=raw_mask_d
        )
        embed_d_all = transformer_model(
            input_tensor=input_d,
            attention_mask=attention_mask_d,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=128,
            intermediate_act_fn=gelu,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            do_return_all_layers=True
        )
        embed_d = embed_d_all[-1]
    return embed_q, embed_d