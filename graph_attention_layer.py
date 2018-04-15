from __future__ import absolute_import
import warnings
from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf
import numpy as np

class GraphAttention(Layer):

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
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
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
        F = input_shape[0][-1]

        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
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

        X = tf.Print(X, [tf.reduce_max(X, None), tf.reduce_min(X, None),X, inputs[1]],
                                   'input X=', summarize=20, first_n=3)

        # Parameters
        N = K.shape(X)[0]  # Number of nodes in the graph

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(linear_transf_X, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(linear_transf_X, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            comparison = K.equal(A, K.constant(0.))
            mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
            masked = dense + mask

            # Feed masked values to softmax
            softmax = K.softmax(masked)  # (N x N), attention coefficients
            dropout = Dropout(self.attn_dropout)(softmax)  # (N x N)

            # Linear combination with neighbors' features
            node_features = K.dot(dropout, linear_transf_X)  # (N x F')

            node_features = tf.Print(node_features, [tf.reduce_max(node_features, None), tf.reduce_min(node_features, None), node_features],
                               'node_features  minmax=', summarize=20, first_n=3)

            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')

        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

class GraphResolutionAttention(Layer):

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
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
        if self.attn_mode == 'layerwise':
            self.resolution_kernel = self.add_weight(shape=(self.num_hops,),
                                                     initializer=self.resolution_attn_kernel_initializer,
                                                     name='resolution_att_kernel',
                                                     regularizer=self.resolution_attn_kernel_regularizer,
                                                     constraint=self.resolution_attn_kernel_constraint)

        elif self.attn_mode == 'full':
            pass
        else:
            raise ValueError("insuitable attention mode")

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # k-Adjacency matrices (N x N x k)

        # Parameters
        N = K.shape(X)[0]  # Number of nodes in the graph

        outputs = []
        k_repeat = lambda x: K.repeat_elements(K.expand_dims(x, axis = -1), rep = self.num_hops, axis = 2)
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1k]^T [Wh_i] + [a_2k]^T [Wh_j]
            attn_for_self = K.dot(linear_transf_X, attention_kernel[0])  # (N x 1 x K), [a_1k]^T [Wh_i]
            attn_for_neighs = K.dot(linear_transf_X, attention_kernel[1])  # (N x 1 x K), [a_2k]^T [Wh_j]

            if self.attn_mode == "full":
                # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                # comment: Neat way to create Topliz matrix (diag-const matrix) from two 1-D vectors
                # (v1,v2) with the following sum structure [[v1:+v21],[v1:+v22],...,[v1:+v2N]]
                dense = K.transpose(K.transpose(K.expand_dims(attn_for_self, 1))
                                    + K.expand_dims(K.transpose(attn_for_neighs), 2)) # (N x N x K) via broadcasting

                # Add nonlinearty
                dense = LeakyReLU(alpha=0.2)(dense)

                if self.weight_mask == True:
                    # masking with weights of path (giving structure additional meaning)
                    dense = dense * A

                # Mask values before activation (Vaswani et al., 2017)
                # TODO: try different comparison like zeroing nonlikely paths.
                # Using mask values will probably dump values very low. Some variations can be tested
                #   1. comparison = K.less_equal(A, K.const(1e-15))
                #   2. K.max(dense * mask, axis=2) # take highest value of dense * mask
                #   3. dense[K.max(mask, axis=2)] # take value of most informative scale
                comparison = K.equal(A, K.constant(0.))
                mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
                mask = activations.softmax(mask,axis=-1)
                masked = K.sum(dense * mask, axis=2) # 3rd dim element-wise dot product
            else:
                # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

                # Add nonlinearty
                dense = LeakyReLU(alpha=0.2)(dense)
                dense = k_repeat(dense) # inflate dense dimension repeat
                # dense = K.expand_dims(dense, axis=2)  # we add the extra dimension:
                # dense = K.repeat_elements(dense, rep=self.num_hops, axis=2)  # we replicate the elements

                if self.weight_mask == True:
                    # masking with weights of path (giving structure additional meaning)
                    dense = dense * A

                # Mask values before activation (Vaswani et al., 2017)
                comparison = K.equal(A, K.constant(0.))
                mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
                masked = tf.tensordot(dense + mask, self.resolution_kernel, axes=[2, 0]) # (N x N), attention coefficients

            # Feed masked values to softmax
            softmax = K.softmax(masked)  # (N x N), attention coefficients
            dropout = Dropout(self.attn_dropout)(softmax)  # (N x N)

            # Linear combination with neighbors' features
            node_features = K.dot(dropout, linear_transf_X)  # (N x F')

            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)

        return output

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
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.attn_mode = kwargs.pop("attention_mode", "gat")
        self.weight_mask = kwargs.pop("weight_mask", False)
        self.l_bias = kwargs.pop("l_bias", None)
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.resolution_attn_kernel_initializer = initializers.get(resolution_attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.resolution_attn_kernel_regularizer = regularizers.get(resolution_attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.resolution_attn_kernel_constraint = constraints.get(resolution_attn_kernel_constraint)
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

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
