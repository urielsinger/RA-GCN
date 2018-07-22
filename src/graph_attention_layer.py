from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow import keras as K
# from keras import activations, constraints, initializers, regularizers
# from keras import backend as K
# from keras.layers import Layer, Dropout, LeakyReLU

from src.utils import variable_summaries

class GraphAttention(K.layers.Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 attn_kernel_constraint=None,
                 gcn_layer_name=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.gcn_layer_name = gcn_layer_name
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = K.activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = K.initializers.get(kernel_initializer)
        self.attn_kernel_initializer = K.initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = K.regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = K.regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = K.regularizers.get(activity_regularizer)
        self.kernel_constraint = K.constraints.get(kernel_constraint)
        self.attn_kernel_constraint = K.constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1] if type(input_shape[0][-1]) != TensorShape else input_shape[0][-1]._dims[0]

        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F.value, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # Attention kernel
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               name='att_kernel_{}'.format(head),
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 name='att_kernel_{}'.format(head),
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint)
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        # Parameters
        # N = K.shape(X)[0]  # Number of nodes in the graph

        with tf.name_scope(self.gcn_layer_name):
            outputs = []
            for head in range(self.attn_heads):
                with tf.name_scope(f'kernel_{head}'):
                    kernel = self.kernels[head]  # W in the paper (F x F')
                    variable_summaries(kernel[head])

                with tf.name_scope(f'attention_{head}'):
                    # Compute inputs to attention network
                    attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
                    # variable_summaries(attention_kernel[head])

                linear_transf_X = K.backend.dot(X, kernel)  # (N x F')

                # Compute feature combinations
                # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
                attn_for_self = K.backend.dot(linear_transf_X, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
                attn_for_neighs = K.backend.dot(linear_transf_X, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

                # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                dense = attn_for_self + K.backend.transpose(attn_for_neighs)  # (N x N) via broadcasting

                # Add nonlinearty
                dense = K.layers.LeakyReLU(alpha=0.2)(dense)

                # Mask values before activation (Vaswani et al., 2017)
                mask = K.backend.exp(A * -10e9) * -10e9
                masked = dense + mask

                # Feed masked values to softmax
                softmax = K.activations.softmax(masked)  # (N x N), attention coefficients
                dropout = K.layers.Dropout(self.attn_dropout)(softmax)  # (N x N)

                # Linear combination with neighbors' features
                node_features = K.backend.dot(dropout, linear_transf_X)  # (N x F')

                if self.attn_heads_reduction == 'concat' and self.activation is not None:
                    # In case of 'concat', we compute the activation here (Eq 5)
                    node_features = self.activation(node_features)

                # Add output of attention head to final output
                outputs.append(node_features)

            with tf.name_scope("activations"):
                # Reduce the attention heads output according to the reduction method
                if self.attn_heads_reduction == 'concat':
                    output = K.backend.concatenate(outputs)  # (N x KF')
                else:
                    output = K.backend.mean(K.backend.stack(outputs), axis=0)  # N x F')
                    if self.activation is not None:
                        # In case of 'average', we compute the activation here (Eq 6)
                        output = self.activation(output)
                tf.summary.histogram('activations',output)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

class GraphResolutionAttention(K.layers.Layer):

    def __init__(self,
                 F_,
                 num_hops=1, # K
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='ones',
                 resolution_attn_kernel_initializer='ones',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 resolution_attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 attn_kernel_constraint=None,
                 resolution_attn_kernel_constraint=None,
                 gcn_layer_name = None,
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.attn_mode = kwargs.pop("attention_mode", "gat")
        self.weight_mask = kwargs.pop("weight_mask", False)
        self.l_bias = kwargs.pop("l_bias", None)
        self.gcn_layer_name= gcn_layer_name
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = K.activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = K.initializers.get(kernel_initializer)
        self.attn_kernel_initializer = K.initializers.get(attn_kernel_initializer)
        self.resolution_attn_kernel_initializer = K.initializers.get(resolution_attn_kernel_initializer)
        self.kernel_regularizer = K.regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = K.regularizers.get(attn_kernel_regularizer)
        self.resolution_attn_kernel_regularizer = K.regularizers.get(resolution_attn_kernel_regularizer)
        self.activity_regularizer = K.regularizers.get(activity_regularizer)
        self.kernel_constraint = K.constraints.get(kernel_constraint)
        self.attn_kernel_constraint = K.constraints.get(attn_kernel_constraint)
        self.resolution_attn_kernel_constraint = K.constraints.get(resolution_attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        self.num_hops = num_hops
        assert num_hops >= 1

        super(GraphResolutionAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F.value, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # regular GAT attention and layerwise attention decompose to
            # different attention kernels (1 X 2F_ and 1 X K (num_hops) )
            if (self.attn_mode == 'layerwise') or (self.attn_mode == 'gat'):
                # Attention kernel for layerwise attention
                attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                                   initializer=self.attn_kernel_initializer,
                                                   name='att_kernel_{}'.format(head),
                                                   regularizer=self.attn_kernel_regularizer,
                                                   constraint=self.attn_kernel_constraint)
                attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                     initializer=self.attn_kernel_initializer,
                                                     name='att_kernel_{}'.format(head),
                                                     regularizer=self.attn_kernel_regularizer,
                                                     constraint=self.attn_kernel_constraint)
                if self.attn_mode == 'layerwise':
                    warnings.warn("todo :: implement multihead layerwise attetion and deprecate", DeprecationWarning)
                    pass
                    # self.resolution_kernel = self.add_weight(shape=(self.num_hops,),
                    #                                          initializer=self.resolution_attn_kernel_initializer,
                    #                                          name='resolution_att_kernel',
                    #                                          regularizer=self.resolution_attn_kernel_regularizer,
                    #                                          constraint=self.resolution_attn_kernel_constraint)

            # Full attention kernel inflate the regular attention in K (num_hops) dimensions
            elif self.attn_mode == 'full':
                attn_kernel_self = self.add_weight(shape=(self.F_, self.num_hops),
                                                   initializer=self.attn_kernel_initializer,
                                                   name='att_kernel_{}'.format(head),
                                                   regularizer=self.attn_kernel_regularizer,
                                                   constraint=self.attn_kernel_constraint)
                attn_kernel_neighs = self.add_weight(shape=(self.F_, self.num_hops),
                                                     initializer=self.attn_kernel_initializer,
                                                     name='att_kernel_{}'.format(head),
                                                     regularizer=self.attn_kernel_regularizer,
                                                     constraint=self.attn_kernel_constraint)
            else:
                raise ValueError("unsuitable attention mode")

            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        # FIXME: add resolution kernel for each attention head
        if self.attn_mode == 'layerwise' and self.num_hops > 1:
            self.resolution_kernel = self.add_weight(shape=(self.num_hops,),
                                                     initializer=self.resolution_attn_kernel_initializer,
                                                     name='resolution_att_kernel',
                                                     regularizer=self.resolution_attn_kernel_regularizer,
                                                     constraint=self.resolution_attn_kernel_constraint)

        elif self.attn_mode in ['full', 'layerwise']:
            pass
        else:
            raise ValueError("insuitable attention model for GRAT mode")

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # k-Adjacency matrices (N x N x k)

        k_repeat = lambda x: K.backend.repeat_elements(K.backend.expand_dims(x, axis=-1), rep = self.num_hops, axis = 2)
        with tf.name_scope(self.gcn_layer_name):
            outputs = []
            for head in range(self.attn_heads):
                with tf.name_scope(f'kernel_{head}'):
                    kernel = self.kernels[head]  # W in the paper (F x F')
                    variable_summaries(kernel[head])
                with tf.name_scope(f'attention_{head}'):
                    attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
                    variable_summaries(attention_kernel[head])

                # Compute inputs to attention network
                linear_transf_X = K.backend.dot(X, kernel)  # (N x F')

                # Compute feature combinations
                # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1k]^T [Wh_i] + [a_2k]^T [Wh_j]
                # TODO: normalize attention with softmax in K dim
                attn_for_self = K.backend.dot(linear_transf_X, attention_kernel[0])  # (N x 1 x K), [a_1k]^T [Wh_i]
                attn_for_neighs = K.backend.dot(linear_transf_X, attention_kernel[1])  # (N x 1 x K), [a_2k]^T [Wh_j]

                if self.attn_mode == "full":
                    # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                    # comment: Neat way to create Topliz matrix (diag-const matrix) from two 1-D vectors
                    # (v1,v2) with the following sum structure [[v1:+v21],[v1:+v22],...,[v1:+v2N]]
                    dense = K.backend.transpose(K.backend.transpose(K.backend.expand_dims(attn_for_self, 1))
                                        + K.backend.expand_dims(K.backend.transpose(attn_for_neighs), 2)) # (N x N x K) via broadcasting

                    if self.weight_mask == True:
                        # masking with weights of path (giving structure additional meaning)
                        dense = dense * A

                    # Mask values before activation (Vaswani et al., 2017)
                    # Add nonlinearty
                    dense = K.layers.LeakyReLU(alpha=0.2)(dense)

                    # TODO: try different comparison like zeroing nonlikely paths.
                    # Using mask values will probably dump values very low. Some variations can be tested
                    #   1. comparison = K.less_equal(A, K.const(1e-15))
                    #   2. K.max(dense * mask, axis=2) # take highest value of dense * mask
                    #   3. dense[K.max(mask, axis=2)] # take value of most informative scale
                    #   4. Try max instead of mean in the 'softmax=..' piece
                    mask = K.backend.exp(A * -10e9) * -10e9
                    masked = K.activations.softmax(dense + mask,axis=1) # what is the right axis to softmax? probably both
                    softmax = K.backend.mean(masked, axis=2)  # a_{i,j} importance is decided by the 2nd axis mean
                    softmax = softmax / K.backend.sum(softmax, axis=-1, keepdims=True)
                elif self.attn_mode in ["layerwise", "gat"]:
                    # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                    dense = attn_for_self + K.backend.transpose(attn_for_neighs)  # (N x N) via broadcasting

                    # Add nonlinearty
                    dense = K.layers.LeakyReLU(alpha=0.2)(dense)
                    if self.num_hops > 1:
                        dense = k_repeat(dense) # inflate dense dimension repeat

                    if self.weight_mask == True:
                        # masking with weights of path (giving structure additional meaning)
                        dense = dense * A

                    # Mask values before activation (Vaswani et al., 2017)
                    mask = K.backend.exp(A * -10e9) * -10e9
                    if self.attn_mode == "layerwise" and self.num_hops > 1:
                        masked = tf.tensordot(dense + mask, self.resolution_kernel, axes=[2, 0]) # (N x N), attention coefficients
                    elif self.num_hops ==  1:
                        masked = dense + mask

                    # Feed masked values to softmax
                    softmax = K.backend.softmax(masked)  # (N x N), attention coefficients

                dropout = K.layers.Dropout(self.attn_dropout)(softmax)  # (N x N)

                # Linear combination with neighbors' features
                node_features = K.backend.dot(dropout, linear_transf_X)  # (N x F')

                if self.attn_heads_reduction == 'concat' and self.activation is not None:
                    # In case of 'concat', we compute the activation here (Eq 5)
                    node_features = self.activation(node_features)

                # Add output of attention head to final output
                outputs.append(node_features)

            with tf.name_scope("activations"):
                # Reduce the attention heads output according to the reduction method
                if self.attn_heads_reduction == 'concat':
                    output = K.backend.concatenate(outputs)  # (N x KF')
                else:
                    output = K.backend.mean(K.backend.stack(outputs), axis=0)  # N x F')
                    if self.activation is not None:
                        # In case of 'average', we compute the activation here (Eq 6)
                        output = self.activation(output)
                tf.summary.histogram('activations',output)

            return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
