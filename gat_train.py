from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from graph_attention_layer import GraphAttention,GraphResolutionAttention
from utils import *

import time

# Define parameters
MODEL, FILTER, ATTN_MODE, WEIGHT_MASK, L_BIAS, R_BIAS = \
    ("GAT",'affinity', None, False, 20, 10) # base implementation

MODEL, FILTER, ATTN_MODE, WEIGHT_MASK, L_BIAS, R_BIAS = \
    ("GRAT",'affinity_k',"full", True, 20, 10) # our modifications

# MODEL = "GAT" # GAT (goes with 'affinity' FILTER)  or GRAT
# specifies the type of attention
# ATTN_MODE = 'full' # 'layerwise' (1 x K) :: 'full' (2F' x K) :: 'gat' (2F' x 1)
# FILTER = 'affinity' # 'localpool','chebyshev' ,'noamuriel' , 'affinity', 'affinity_k'

DATASET = 'cora'
# specifies the type of the Affinity kernel
MAX_DEGREE = 3  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 300
PATIENCE = 40  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'affinity':
    """ As in the original paper, A is the affinity matrix """
    print('Using plain affinity matrix filters...')
    support = 1
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
    support = MAX_DEGREE + 1
    graph = [X] + T_k + get_adjointed_l_bias(X, A, l = L_BIAS, radius = R_BIAS)
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

elif FILTER in ['noamuriel','affinity_k']:
    """ noamuriel polynomial basis filters (Defferard et al., NIPS 2016)  """
    print(f'Using {FILTER} polynomial basis filters...')
    A_norm = normalize_adj(A, SYM_NORM) if FILTER == 'noamuriel' else A
    A_k = noamuriel_polynomial(A_norm, MAX_DEGREE, to_tensor=True)
    support = MAX_DEGREE + 1
    graph = [X] + A_k + get_adjointed_l_bias(X, A, l = L_BIAS, radius = R_BIAS)
    G = [Input(shape=(None, None, support), batch_shape=(None, None, support), sparse=False)]

else:
    raise Exception('Invalid filter type.')



X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
if MODEL == "GRAT":
    param = dict(attention_mode=ATTN_MODE,weight_mask=WEIGHT_MASK, l_bias = L_BIAS)
    H = Dropout(0.5)(X_in)
    H = GraphResolutionAttention(16, support, activation='relu', kernel_regularizer=l2(5e-4), **param)([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphResolutionAttention(y.shape[1], support, activation='softmax', **param)([H]+G)
elif MODEL == "GAT":
    H = Dropout(0.5)(X_in)
    H = GraphAttention(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphAttention(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
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
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))