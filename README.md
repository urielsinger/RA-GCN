## 2DO
- [ ] Modify attention similarity in $a^T(Wh_j||Wh_i)$
- [ ] Test normalized attention
- [ ] Try different comparison like zeroing nonlikely paths.
- [ ] Using mask values will probably dump values very low. Some variations can be tested
    1. comparison = K.less_equal(A, K.const(1e-15))
    2. Take highest value of dense * mask
    3. Take value of most informative scale