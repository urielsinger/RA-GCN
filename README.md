## 2DO
### model
- [ ] Modify attention similarity in $a^T(Wh_j||Wh_i)$
- [ ] Test normalized attention
- [ ] Get citeseer test to work
- [ ] Check matrix-wise softmax in the activation
- [ ] Try different comparison like zeroing nonlikely paths.
- [ ] Using mask values will probably dump values very low. Some variations can be tested
    1. comparison = K.less_equal(A, K.const(1e-15))
    2. Take highest value of dense * mask
    3. Take value of most informative scale

### refractoring and optimization
- [ ] Make the summaries and tfboard optional

### upcomming experiments:
+ GAT layerwise support=3 (w/wo weight) noamuriel
+ GCN layerwise support 3 noamuriel
+ sofmax change - softmax = softmax/K.sum(softmax,axis=-1,keepdims=True)
+ noamuriel + I -
print(f'Using {FILTER} polynomial basis filters...')
    A_norm = preprocess_adj(A, SYM_NORM) if FILTER == 'noamuriel' else A+scipy.sparse.eye(A.shape[0])
    A_k = noamuriel_polynomial(A_norm, MAX_DEGREE, to_tensor=True)