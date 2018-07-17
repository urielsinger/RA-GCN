from __future__ import print_function

import time

from graph import GraphConvolution, GraphResolutionConvolution
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import pandas as pd
from src.utils import *

# Define parameters
benchmark = {"repeats":10, "log_notes":"gcn_baseline"} # None
benchmark = dict(repeats=1) if benchmark == None else benchmark
RESULTS_PATH= "../notebook/results.csv"
MODEL = "gcn" # gcn or grcn
DATASET = 'cora_exp'
FILTER = 'localpool'
# FILTER = 'localpool'  # 'localpool','chebyshev'
MAX_DEGREE = 3  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 300
PATIENCE = np.inf  # early stopping patience

# Get data
# X, A, y = load_data_cora(dataset=DATASET)
# y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_test, idx_val, idx_train = load_data(DATASET)
number_classes = y_train.shape[1]

# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
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
    graph = [X] + T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

elif FILTER == 'noamuriel':
    """ noamuriel polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using noamuriel polynomial basis filters...')
    A_norm = preprocess_adj(A, SYM_NORM)
    A_k = noamuriel_polynomial(A_norm, MAX_DEGREE)
    support = MAX_DEGREE
    graph = [X] + A_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))

def compile_model():
    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    if MODEL == "gcn":
        H = Dropout(0.5)(X_in)
        H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
        H = Dropout(0.5)(H)
        Y = GraphConvolution(number_classes, support, activation='softmax')([H]+G)
    elif MODEL == "grcn":
        H = Dropout(0.5)(X_in)
        H = GraphResolutionConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
        H = Dropout(0.5)(H)
        Y = GraphResolutionConvolution(number_classes, support, activation='softmax')([H]+G)

    # Compile model
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
    return model

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
results_list = []
model = compile_model()
for r in range(benchmark["repeats"]):
    model.reset_states()
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
    results_list.append((test_loss[0], test_acc[0]))
    model = compile_model()
    wait = 0

if benchmark["repeats"] > 1:
    results = pd.read_csv(RESULTS_PATH) if os.path.isfile(RESULTS_PATH) else pd.DataFrame()
    results[benchmark["log_notes"]] = np.array(results_list)[:, 1]
    col2save = list(filter(lambda s: 'Unnamed' not in s, results.columns))
    results[col2save].to_csv(RESULTS_PATH)
