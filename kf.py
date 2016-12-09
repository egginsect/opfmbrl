import pickle
from pykalman import KalmanFilter
import pdb
import sys
with open(sys.argv[1]) as f:
    X=pickle.load(f)
print X.shape
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=X.shape[1])
kf = kf.em(X,n_iter=10)
out = kf.smooth(X)
pdb.set_trace()
