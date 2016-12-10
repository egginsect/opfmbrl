import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
numGen=30
numLoad=60
numCapacitor=30
T = 2000
t = np.linspace(1, T, num=T).reshape(T,1)
p = np.exp(-np.cos((t-15)*2*np.pi/T)+0.01*np.random.randn(T,1))
u = 2*np.exp(-np.random.uniform(0,1)*np.cos((t+np.random.uniform(35,40))*np.pi/T) - \
    np.random.uniform(0,1)*np.cos(t*4*np.pi/T)+0.01*np.random.randn(T,1))
p = np.asmatrix(p)
u = np.asmatrix(u)
u=u.T

import cvxpy as cvx

Qbase = 35
Cbase, Dbase = 5, 5
Abase = 1
Bbase = 2
PmaxBase = 16
g = cvx.Variable(numGen, T)
q = cvx.Variable(numCapacitor, T)
c = cvx.Variable(numCapacitor, T)
con =list()
environmentVar = dict()
loss = 0
for i in xrange(numGen):
    A = Abase + np.random.uniform(-0.5, 1)
    B = Bbase + np.random.uniform(-1, 1)
    loss = loss + cvx.sum_entries(A * (g[i, :]) ** 2 + B * (g[i, :]))
    Pmax = PmaxBase + np.random.randn()
    con.append(g[i, :] <= Pmax)
    if 'g'+str(i) in environmentVar:
        environmentVar['g'+str(i)].append((A,B,Pmax))
    else:
        environmentVar['g'+str(i)]=(A,B,Pmax)
obj = cvx.Minimize(loss)
con = list()
con.append(cvx.sum_entries(g, axis=0) == numLoad * u + cvx.sum_entries(c, axis=0))
environmentVar['u'] = u.tolist()
#con.append(cvx.sum_entries(g,axis=0)==numLoad*u)
#con.append(q[:,0]==q[:,T-1]+c[:,T-1])
for i in xrange(0, T - 2):
    con.append(q[:, i + 1] == q[:, i] + c[:, i])
for i in xrange(q.size[0]):
    Q = Qbase + np.random.randn() * 2
    D = Dbase - np.random.randn() * 2
    C = Cbase + np.random.randn() * 2
    con.append(q[i, :] <= Q)
    con.append(c[i, :] >= -D)
    con.append(c[i, :] <= C)
    if 'q' + str(i) in environmentVar:
        environmentVar['q' + str(i)].append((Q, D, C))
    else:
        environmentVar['q' + str(i)] = (Q, D, C)
con.append(q >= 0)
con.append(g >= 0)

import pickle
powerData={'generator':g.value,'load':u,'capacitor':q.value}
with open('powerData.pkl','w') as f:
    pickle.dump(powerData, f)

prob = cvx.Problem(obj, con)
prob.solve()

