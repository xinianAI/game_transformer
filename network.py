import tensorflow as tf
import numpy as np
import pandas
import os
import matplotlib as plt


class weijia(object):
    def __init__(self, father):
        self.father = father

    def prt(self):
        print(self.father, 'is my fater.')


class parent(weijia):
    def __init__(self, father, name, act):
        super(parent, self).__init__(father)
        self.name = name
        self.act = act

    def prt1(self):
        print(self.name + " " + self.act + " to him.")


class Embedding(Layer):
    def __init__(self, vocab_size, model_dim, **kwargs):
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings=self.add_weight(
            shape=(self.vocab_size, self.model_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="embeddings"
        )
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        inputs=tf.cast(inputs, tf.int32)
        embeddings = tf.gather(self.embeddings, inputs)
        embeddings*=self.model_dim**0.5
        return embeddings

    def get_config(self):
        config=super(Embedding, self).get_config()
        config.update({
            "vocab_size":self.vocab_size,
            "model_dim":self.model_dim
        })
        return config


class PositionEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        def get_position_encoding(seq_len, model_dim):
            position_encoding=np.zero(shape=(seq_len,model_dim))
            for pos in range(seq_len):
                for i in range(model_dim):
                    position_encoding[pos, i]=pos/(np.power(10000,2*i/model_dim))
            position_encoding[::,::2]=np.sin(position_encoding[::,::2])
            position_encoding[::,1::2]=np.cos(position_encoding[::,1::2])
            return np.expand_dims(position_encoding,axis=0)
        seq_len, model_dim=input_shape.as_list()[1:3]
        self.position_encoding=self.add_weight(
            shape=(1,seq_len,model_dim),
            initializer=Constant(get_position_encoding(seq_len,model_dim)),
            trainable=False,
            name="position_encoding"
        )
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs):
        return self.position_encoding


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class scaled_dot_product_attention(Layer):
    


class transformer(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    par = parent('gaox', 'weijia', 'ketou')
    par.prt()
    par.prt1()
    tr = transformer()
    print(tr.solve(par.prt(), par.prt1()))


