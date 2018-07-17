from __future__ import print_function

import datetime
import time
import pandas as pd
from tensorflow import keras as K
import json
from utils import *

from src.graph_attention_layer import GraphAttention, GraphResolutionAttention
RESULTS_PATH= "./notebook/results.csv"

'''
Possible parameters for config
MODEL = GAT, GAT (goes with 'affinity' FILTER)  or GRAT
FILTER = 'affinity', 'localpool','chebyshev' ,'noamuriel' , 'affinity', 'affinity_k'
ATTN_MODE = 'full', 'layerwise' (1 x K) :: 'full' (2F' x K) :: 'gat' (2F' x 1)
'''

# Define parameters
confing_name = "baseline_config" # free_experiment_config
config_path= fr"experiments_config/cora_exp/{confing_name}.json"
general_config_path= fr"experiments_config/general_config.json"
benchmark = None # None avoids logging
# benchmark = {"repeats":1, "log_notes":"grat_nu_lw_weight"} # None
debug_mode = True # uses eager_execution

log_to_tensorboard = False if benchmark == None else False
benchmark = dict(repeats=1) if benchmark == None else benchmark
with open(config_path,'r') as f:
    config_param = json.load(f)
with open(general_config_path,'r') as f:
    general_config = json.load(f)
if config_param["NOTES"] is None:
    config_param["NOTES"] = f"{config_param['MODEL']}_{config_param['FILTER']}_" \
                            f"maxdeg{config_param['MAX_DEGREE']}_{config_param['ATTN_MODE']}attn_" \
                            f"{str(config_param['L_BIAS'])}bias_weight{str(config_param['WEIGHT_MASK'])[0]}" + config_param['FREENOTES']
if debug_mode:
    tf.enable_eager_execution()
    # tf.executing_eagerly()
else:
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

# specifies the type of the Affinity kernel
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
PATIENCE = np.inf  if general_config["patience"] == "inf" else general_config["patience"]   # early stopping patience
print(f"{'#'*150}\nExperiment description -\n"
      f"\t dataset='{config_param['DATASET']}'\t model='{config_param['MODEL']}'\t moments='{config_param['FILTER']}'\t attn_mode='{config_param['ATTN_MODE']}'"
      f"\t weigthed_mask={config_param['WEIGHT_MASK']}\t bias_mat={config_param['L_BIAS']}\n{'#'*150}")

# Get data
A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_test, idx_val, idx_train = load_data(config_param['DATASET'])
number_classes = y_train.shape[1]

# thresh = 2000
# idx_test = [k for k,v in zip(idx_test,idx_test) if v < thresh ]
# idx_val = [k for k,v in zip(idx_val,idx_test) if v < thresh ]
# idx_train = [k for k,v in zip(idx_train,idx_test) if v < thresh ]
# A = A[:2500, :2500]
# X = X[:2500,:]
# y_train = y_train[:2500,:]
# train_mask = train_mask[:2500]

# Normalize X
X /= X.sum(1).reshape(-1, 1)

# ___________________ - Define input for model - ___________________
def _build_model_input(A_matrix):

    if config_param['FILTER'] == 'affinity':
        """ As in the original paper, A is the affinity matrix """
        print('Using plain affinity matrix filters...')
        support = 1
        A_matrix = A_matrix + np.eye(A_matrix.shape[0])  # Add self-loops
        graph = [X, A_matrix]
        G = [K.Input(shape=(None, None), sparse=False)]

    elif config_param['FILTER'] == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        A_matrix_ = preprocess_adj(A_matrix, SYM_NORM)
        support = 1
        graph = [X, A_matrix_]
        G = [K.Input(shape=(None, None), sparse=True)]

    elif config_param['FILTER'] == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A_matrix, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, config_param['MAX_DEGREE'])
        if config_param['L_BIAS'] is not None:
            Bias = get_adjointed_l_bias(X, A_matrix, l=config_param['L_BIAS'], radius=config_param['R_BIAS']
                                        , n_jobs=config_param['N_JOBS'], cache=config_param['DATASET'])
            T_k = [np.concatenate((T_k[0] ,np.expand_dims(Bias, axis=-1)), axis=-1)]
        graph = [X] + T_k
        support = config_param['MAX_DEGREE'] + 1 if config_param['L_BIAS'] is None else config_param['MAX_DEGREE'] + 2
        G = [K.Input(shape=(None, None), sparse=True) for _ in range(support)]

    elif config_param['FILTER'] in ['noamuriel','affinity_k']:
        """ noamuriel polynomial basis filters (Defferard et al., NIPS 2016)  """
        print(f"Using {config_param['FILTER']} polynomial basis filters...")
        A_norm = preprocess_adj(A_matrix, SYM_NORM) if config_param['FILTER'] == 'noamuriel' else A_matrix+scipy.sparse.eye(A_matrix.shape[0])
        A_k = noamuriel_polynomial(A_norm, config_param['MAX_DEGREE'], to_tensor=True)
        if config_param['L_BIAS'] is not None:
            A_k = [np.transpose(np.array([A_matrix.todense() for A_matrix in A_k]), [1,2,0])] if config_param['MAX_DEGREE'] == 1 else A_k
            Bias = get_adjointed_l_bias(X, A_matrix, l=config_param['L_BIAS'], radius=config_param['R_BIAS'], n_jobs=config_param['N_JOBS'], cache=config_param['DATASET'])
            A_k = [np.concatenate((A_k[0], np.expand_dims(Bias, axis=-1)), axis=-1)]
        graph = [X] + A_k
        support = config_param['MAX_DEGREE'] if config_param['L_BIAS'] is None else config_param['MAX_DEGREE'] + 1
        if support ==1:
            G = [K.Input(shape=(None, None), sparse=False)]
        else:
            G = [K.Input(shape=(None, None, support), sparse=False)]

    else:
        raise Exception('Invalid filter type.')

    return G, graph, support

def compile_model():

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    if config_param['MODEL'] == "GRAT":
        param = dict(attention_mode=config_param['ATTN_MODE'],weight_mask=config_param['WEIGHT_MASK'], l_bias=config_param['L_BIAS'])
        H = K.layers.Dropout(general_config["dropout_rate"])(X_in)
        H = GraphResolutionAttention(general_config["F_"], num_hops=general_config["num_hops"], activation=general_config["activation"]
                     , kernel_regularizer=K.regularizers.l2(general_config["l2_reg"]), attn_heads=general_config["n_attn_heads_1"]
                     , attn_heads_reduction=general_config["attn_heads_reduction_1"], gcn_layer_name='GRCN_L1'
                     , **param)([H]+G)
        H = K.layers.Dropout(general_config["dropout_rate"])(H)
        Y = GraphResolutionAttention(number_classes , num_hops=general_config["num_hops"], activation='softmax'
                    , attn_heads=general_config["n_attn_heads_2"], attn_heads_reduction=general_config["attn_heads_reduction_2"]
                    ,gcn_layer_name='GRCN_L2', **param)([H]+G)
    elif config_param['MODEL'] == "GAT":
        H = K.layers.Dropout(general_config["dropout_rate"])(X_in)
        H = GraphAttention(general_config["F_"], kernel_regularizer=K.regularizers.l2(general_config["l2_reg"])
                   , attn_heads=general_config["n_attn_heads_1"], attn_heads_reduction=general_config["attn_heads_reduction_1"]
                   , activation = general_config["activation"], gcn_layer_name='GRCN_L1'
                   )([H]+G)
        H = K.layers.Dropout(general_config["dropout_rate"])(H)
        Y = GraphAttention(number_classes , activation='softmax'
                   ,attn_heads=general_config["n_attn_heads_2"], attn_heads_reduction=general_config["attn_heads_reduction_2"]
                   , gcn_layer_name='GRCN_L2')([H]+G)

    # Compile model
    model = K.Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(lr=0.01))
    return model

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit


G, graph, support = _build_model_input(A)
general_config["num_hops"] = support
X_in = K.Input(shape=(X.shape[1],))

if log_to_tensorboard:
    metric_writer = tf.summary.FileWriter(f"./tensorboard_logdir/{datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')}"
                                          f"_{config_param['DATASET']}_{config_param['NOTES']}",)
    test_writer = tf.summary.FileWriter(f"./tensorboard_logdir/test",)
    graph_dense = graph[1].todense() if scipy.sparse.issparse(graph[1]) else graph[1]

results_list = []
model = compile_model()

for r in range(benchmark["repeats"]):
    model.reset_states()
    print(f"\n----------- running {r+1} experimet -----------\n")
    for epoch in range(1, general_config["epochs"]+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        if epoch%10 == 0:
            model.fit(graph, y_train, sample_weight=train_mask,batch_size=A.shape[0]
                      , epochs=1, shuffle=False, verbose=0)
        else:
            history = model.fit(graph, y_train, sample_weight=train_mask,batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        with tf.name_scope(f"metrics"):
            summary_list = []
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],[idx_train, idx_val])
            with tf.name_scope(f"train"):
                train_loss = train_val_loss[0]
                summary_list.append(tf.Summary.Value(tag="train_loss", simple_value=train_loss))
                train_acc = train_val_acc[0]
                summary_list.append(tf.Summary.Value(tag="train_acc", simple_value=train_acc))
            with tf.name_scope(f"val"):
                val_loss = train_val_loss[1]
                summary_list.append(tf.Summary.Value(tag="val_loss", simple_value=val_loss))
                val_acc = train_val_acc[1]
                summary_list.append(tf.Summary.Value(tag="val_acc", simple_value=val_acc))
        # merged = tf.summary.merge_all()
        summary_train = tf.Summary(value=summary_list)
        if log_to_tensorboard:
            metric_writer.add_summary(summary_train, epoch)
            merged = tf.summary.merge_all()
            trainable_summary = merged.eval(session=sess, feed_dict={G[0]: graph_dense , X_in: graph[0]})
            metric_writer.add_summary(trainable_summary, epoch)

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_loss),
              "train_acc= {:.4f}".format(train_acc),
              "val_loss= {:.4f}".format(val_loss),
              "val_acc= {:.4f}".format(val_acc),
              "time= {:.4f}".format(time.time() - t))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1
    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    if log_to_tensorboard:
        test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=test_acc[0]), \
                                              tf.Summary.Value(tag="test_loss", simple_value=test_loss[0])]))
        metric_writer.close()
        test_writer.close()

    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    results_list.append((test_loss[0], test_acc[0]))
    model = compile_model()
    wait = 0

if benchmark["repeats"] > 1:
    results = pd.read_csv(RESULTS_PATH) if os.path.isfile(RESULTS_PATH) else pd.DataFrame()
    results[benchmark["log_notes"]] = np.array(results_list)[:, 1]
    col2save = list(filter(lambda s: 'Unnamed' not in s, results.columns))
    results[col2save].to_csv(RESULTS_PATH)
