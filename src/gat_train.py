from __future__ import print_function

import datetime
import time
import pandas as pd
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.backend import squeeze
from utils import *

from src.graph_attention_layer import GraphAttention, GraphResolutionAttention
RESULTS_PATH= "./notebook/results.csv"

# Define parameters
benchmark = {"repeats":10, "log_notes":"noweight_nobias_fullattn_"} # None
log_to_tensorboard = False if benchmark == None else False
FREENOTES = "reproduce_with_GRAT"
DATASET = 'cora' # citeseer, cora
MODEL = "GRAT"          # GAT (goes with 'affinity' FILTER)  or GRAT
FILTER = 'affinity_k'    # 'localpool','chebyshev' ,'noamuriel' , 'affinity', 'affinity_k'
ATTN_MODE = "full"      # 'layerwise' (1 x K) :: 'full' (2F' x K) :: 'gat' (2F' x 1)
WEIGHT_MASK = False
L_BIAS = None
R_BIAS = 3
N_JOBS = 1
MAX_DEGREE = 3  # maximum polynomial degree
NOTES = f"{MODEL}_{FILTER}_maxdeg{MAX_DEGREE}_{ATTN_MODE}attn_{str(L_BIAS)}bias_weightMask{str(WEIGHT_MASK)[0]}" + FREENOTES
benchmark = dict(repeats=1) if benchmark == None else benchmark
NOTES = "BASELINE_GRAT"

# MODEL, FILTER, ATTN_MODE, WEIGHT_MASK, L_BIAS, R_BIAS, N_JOBS = \
#     ("GAT",'affinity', None, False, None, None, None) # base implementation
# NOTES = "BASELINE_GAT"

# MODEL = "GAT" # GAT (goes with 'affinity' FILTER)  or GRAT
# specifies the type of attention
# ATTN_MODE = 'full' # 'layerwise' (1 x K) :: 'full' (2F' x K) :: 'gat' (2F' x 1)
# FILTER = 'affinity' # 'localpool','chebyshev' ,'noamuriel' , 'affinity', 'affinity_k'


# specifies the type of the Affinity kernel
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 300
PATIENCE = np.inf  # early stopping patience
print(f"{'#'*150}\nExperiment description -\n"
      f"\t dataset='{DATASET}'\t model='{MODEL}'\t moments='{FILTER}'\t attn_mode='{ATTN_MODE}'"
      f"\t weigthed_mask={WEIGHT_MASK}\t bias_mat={L_BIAS}\n{'#'*150}")

# Get data
A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_test, idx_val, idx_train = load_data(DATASET)
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

if FILTER == 'affinity':
    """ As in the original paper, A is the affinity matrix """
    print('Using plain affinity matrix filters...')
    support = 1
    A = A + np.eye(A.shape[0])  # Add self-loops
    graph = [X, A]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]

elif FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    if L_BIAS is not None:
        Bias = get_adjointed_l_bias(X, A, l=L_BIAS, radius=R_BIAS, n_jobs=N_JOBS, cache=DATASET)
        T_k = [np.concatenate((T_k[0] ,np.expand_dims(Bias, axis=-1)), axis=-1)]
    graph = [X] + T_k
    support = MAX_DEGREE + 1 if L_BIAS is None else MAX_DEGREE + 2
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

elif FILTER in ['noamuriel','affinity_k']:
    """ noamuriel polynomial basis filters (Defferard et al., NIPS 2016)  """
    print(f'Using {FILTER} polynomial basis filters...')
    A_norm = normalize_adj(A, SYM_NORM) if FILTER == 'noamuriel' else A+scipy.sparse.eye(A.shape[0])
    A_k = noamuriel_polynomial(A_norm, MAX_DEGREE, to_tensor=True)
    if L_BIAS is not None:
        A_k = [np.transpose(np.array([A.todense() for A in A_k]), [1,2,0])] if MAX_DEGREE == 1 else A_k
        Bias = get_adjointed_l_bias(X, A, l=L_BIAS, radius=R_BIAS, n_jobs=N_JOBS, cache=DATASET)
        A_k = [np.concatenate((A_k[0], np.expand_dims(Bias, axis=-1)), axis=-1)]
    graph = [X] + A_k
    support = MAX_DEGREE if L_BIAS is None else MAX_DEGREE + 1
    if support ==1:
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]
    else:
        G = [Input(shape=(None, None, support), batch_shape=(None, None, support), sparse=False)]

else:
    raise Exception('Invalid filter type.')


def compile_model():
    X_in = Input(shape=(X.shape[1],))

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    if MODEL == "GRAT":
        param = dict(attention_mode=ATTN_MODE,weight_mask=WEIGHT_MASK, l_bias=L_BIAS)
        H = Dropout(0.5)(X_in)
        H = GraphResolutionAttention(16, support, activation='relu', kernel_regularizer=l2(5e-4), gcn_layer_name='GRCN_L1', **param)([H]+G)
        H = Dropout(0.5)(H)
        Y = GraphResolutionAttention(number_classes , support, activation='softmax',gcn_layer_name='GRCN_L2', **param)([H]+G)
    elif MODEL == "GAT":
        H = Dropout(0.5)(X_in)
        H = GraphAttention(16, support, activation='relu', kernel_regularizer=l2(5e-4),gcn_layer_name='GRCN_L1')([H]+G)
        H = Dropout(0.5)(H)
        Y = GraphAttention(number_classes , support, activation='softmax', gcn_layer_name='GRCN_L2')([H]+G)

    # Compile model
    model = Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
    return model

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999


# Fit
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
if log_to_tensorboard:
    metric_writer = tf.summary.FileWriter(f"./tensorboard_logdir/{datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')}"
                                          f"_{DATASET}_{NOTES}",)
    test_writer = tf.summary.FileWriter(f"./tensorboard_logdir/test",)
    graph_dense = graph[1].todense() if scipy.sparse.issparse(graph[1]) else graph[1]

results_list = []
model = compile_model()
for r in range(benchmark["repeats"]):
    model.reset_states()
    print(f"\n----------- running {r+1} experimet -----------\n")
    for epoch in range(1, NB_EPOCH+1):

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

if benchmark is not None:
    results = pd.read_csv(RESULTS_PATH) if os.path.isfile(RESULTS_PATH) else pd.DataFrame()
    results[benchmark["log_notes"]] = np.array(results_list)[:, 1]
    col2save = list(filter(lambda s: 'Unnamed' not in s, results.columns))
    results[col2save].to_csv(RESULTS_PATH)
