import numpy as np
from scipy import optimize
from sklearn import base, metrics
from sklearn.utils.validation import check_X_y

def kernel1(x1,x2,gamma):
    dis = np.linalg.norm(np.subtract(x1,x2))**2
    return np.exp(-dis/gamma)


def prepocess_kernel(kernel, X, gamma):
    if(kernel=='linear'):
        return X, np.identity(len(X[0]))

    if(kernel=='rbf'):

        K=np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(i, len(X)):
            
                K[i][j]=K[j][i]= kernel1(X[i],X[j],gamma)
               # print (K[i][j])

        return K, K

def grad_sigmoid(t):

    return 1.0/((1+np.exp(t))*(1+np.exp(-t)))


def sigmoid(t):

    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))

    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)

    return out


def log_loss(Z):

    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))

    return out

def hinge(x,a):
    return max(0,-a*x)


def obj_margin(x0, X, y, alpha, n_class, weights, L, sample_weight, kernel_constant, mode,h):

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]

    theta = L.dot(c)
    loss_fd = weights[y]

    if(mode == 'CL'):
        eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
        prob = np.pad(
            sigmoid(eta).T,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))
    
        prob = np.diff(prob)

        obj = 0.
        for i in range(len(prob)):
            obj -= sample_weight[i] * np.log(prob[i][y[i]])

        obj += alpha * 0.5 * np.dot(w, np.dot(kernel_constant,w.T))
        return obj

    elif mode=='TOR':
        obj=0.

        for i in range(len(y)):
            if(y[i]==0):
                obj += sample_weight[i] * hinge(theta[0] - np.log(sigmoid(w.dot(X[i]))), h[0] )
            elif(y[i]==n_class-1):
                obj += sample_weight[i] * hinge(np.log(sigmoid(w.dot(X[i]))) -theta[-1], h[-1])
            else:
                obj += sample_weight[i]*(hinge(np.log(sigmoid(w.dot(X[i])))-theta[y[i]-1], h[y[i]-1]) + hinge(theta[y[i]]-np.log(sigmoid(w.dot(X[i]))), h[y[i]]))

        obj += alpha * 0.5 * np.dot(w, np.dot(kernel_constant,w.T))

        return obj

    else:
        Xw = X.dot(w)
        Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
        S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
        err = loss_fd.T * log_loss(S * Alpha)
        if sample_weight is not None:
            err *= sample_weight
        obj = np.sum(err)

        
        obj += alpha * 0.5 * np.dot(w, np.dot(kernel_constant,w.T))

    return obj

def CL_grad_log(x, y, theta, w, n_class):

# -------------- log(1/ 1+exp(wx-theta))'x     = - exp(wx-theta)/(1+exp(wx-theta))*x  = - 1 / (1+exp(theta-wx))*x
#----------------log(1/ 1+exp(wx-theta))'theta =   exp(wx-theta)/(1+exp(wx-theta))    =  1 / (1+exp(theta-wx))
#----------------log(1- 1/ 1+exp(wx-theta))'x  = log( 1/ 1+exp(theta-wx))'x       =   1 / (1+exp(wx - theta))*x 
#----------------log(1- 1/ 1+exp(wx-theta))'theta  = log( 1/ 1+exp(theta-wx))'theta       =  - 1 / (1+exp(wx - theta))
        xw = x.dot(w)
        de = np.zeros_like(theta)


        if(y==0):  
            de[0]=1.
            return -sigmoid(-theta[y] + xw)*x,  sigmoid(-theta[y] + xw)*de

        if(y==n_class-1):
            de[y-1]=1.
            return sigmoid( theta[y-1] -  xw)*x, -sigmoid(theta[y-1] - xw)*de

        c1=grad_sigmoid(theta[y]-xw)
        c2=grad_sigmoid(theta[y-1]-xw)
        c3= sigmoid( theta[y] - xw) - sigmoid ( theta[y-1] - xw) 

        de[y], de[y-1]=c1/c3, -c2/c3

        return  -(c1-c2)/c3*x, de

def hinge_grad(wx, theta, b, ind, x, n_class,h):

    t=np.array([0. for i in range(n_class-1)])
    if(b==0):
        if(np.log(sigmoid(wx))-theta<0):
            t[ind]=h
            return -h*sigmoid(-wx)*x, t
        else:
            return 0.*x, t
    else:
        if(theta-np.log(sigmoid(wx))<0):
            t[ind]=-h
            return h*sigmoid(-wx)*x, t
        else:
            return 0.*x, t


def grad_margin(x0, X, y, alpha, n_class, weights, L, sample_weight, kernel_constant, mode,h):

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y] 
    if(mode == 'CL'):

        grad_w = alpha * np.matmul(w, kernel_constant)   # regulaziation terms
        grad_theta = np.zeros_like(theta)

        for i in range(len(y)):
            t1,t2 =  CL_grad_log(X[i], y[i], theta, w, n_class) 
            
            grad_w-= t1 * sample_weight[i]
            grad_theta -= t2 * sample_weight[i]

        
        
        grad_c = L.T.dot(grad_theta) 

        return np.concatenate((grad_w, grad_c), axis=0)

    elif mode == 'TOR':
        grad_w = alpha * np.matmul(w, kernel_constant)   # regulaziation terms
        grad_theta = np.zeros_like(theta)

        for i in range(len(y)):
            if(y[i]==0):
                t1,t2 = hinge_grad(w.dot(X[i]), theta[0], 1, 0, X[i], n_class,h[0])
            elif(y[i]==n_class-1):
                t1,t2 = hinge_grad(w.dot(X[i]), theta[-1], 0, -1, X[i], n_class,h[y[i]])
            else:
                t1,t2 = hinge_grad(w.dot(X[i]), theta[y[i]-1], 0, y[i]-1, X[i], n_class,h[y[i]-1])
                t3,t4 = hinge_grad(w.dot(X[i]), theta[y[i]], 1, y[i], X[i], n_class,h[y[i]])
                t1+=t3
                t2+=t4
         
            grad_w+= t1 * sample_weight[i]
            grad_theta += t2 * sample_weight[i]

        grad_c = L.T.dot(grad_theta)
        return np.concatenate((grad_w, grad_c), axis=0)

    else:
        Xw = X.dot(w)
        Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
        S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
        # Alpha[idx] *= -1
        # W[idx.T] *= -1

        Sigma = S * loss_fd.T * sigmoid(-S * Alpha)
        if sample_weight is not None:
            Sigma *= sample_weight

        grad_w = X.T.dot(Sigma.sum(0)) + alpha * np.matmul(w, kernel_constant)
        grad_theta = -Sigma.sum(1)


        grad_c = L.T.dot(grad_theta)


        return np.concatenate((grad_w, grad_c), axis=0)



def threshold_fit(X, y,  kernel_constant, alpha,n_class, mode='AE',
                  max_iter=1000, verbose=False, tol=1e-12,
                  sample_weight=None, hinge_const=[1.,1.,1.,1.,1.]):


    X, y = check_X_y(X, y, accept_sparse='csr')
    unique_y = np.sort(np.unique(y))
    if not np.all(unique_y == np.arange(unique_y.size)):
        raise ValueError(
            'Values in y must be %s, instead got %s'
            % (np.arange(unique_y.size), unique_y))

    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    if(mode=='TOR'):
        L[np.triu_indices(n_class-1)] = -1.
        for i in range(len(L)):
            L[i][-1]=1.
    else:
        L[np.tril_indices(n_class-1)] = 1.

    if mode == 'AE':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == '0-1':
        loss_fd = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = 1  # border case
      #  print(loss_fd.shape, loss_fd)
    elif mode == 'CL':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    elif mode == 'TOR':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    else:
        raise NotImplementedError

    if mode!='TOR':
        x0 = np.zeros(n_features + n_class - 1)
        x0[X.shape[1]:] = np.arange(n_class - 1)
        options = {'maxiter' : max_iter, 'disp': False}
        bounds = [(None, None)] * (n_features + 1) + \
                 [(0, None)] * (n_class - 2)

    else:   
        x0 = np.zeros(n_features + n_class - 1)
        x0[X.shape[1]:] = np.linspace(1, 1, 4)
        x0[-1]=-0.5
        options = {'maxiter' : max_iter, 'disp': True}
        bounds = [(None, None)] * (n_features) + \
                 [(0, None)] * (n_class - 2) + \
                 [(None, 0) * 1]
  #  print (n_samples, n_features, n_class)

    best_f=10E+9
    best_sol=[]
    for it in range(100):
        for i in range(n_features):
            x0[i] = np.random.uniform(-5,5)
        x0[X.shape[1]:] = np.arange(n_class - 1)
     
        sol = optimize.minimize(obj_margin, x0, method='L-BFGS-B',
            jac=grad_margin, bounds=bounds, options=options,
            args=(X, y, alpha, n_class, loss_fd, L, sample_weight, kernel_constant, mode, hinge_const),
            tol=tol)
        
        if(best_f>sol.fun):
            best_f=sol.fun
            best_sol = sol.x

    if verbose and not sol.success:
        print(sol.message)


    w, c = best_sol[:X.shape[1]], sol.x[X.shape[1]:]
    theta = L.dot(c)
    return w, theta


def threshold_predict(X, w, theta, mode):

    if(mode == 'CL'):
        eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
      
        prob = np.pad(
            sigmoid(eta).T,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))

        prob = np.diff(prob)
        pred =[]
        for i in range(len(prob)):
            pred.append(np.argmax(prob[i]))

    else:
        tmp = theta[:, None] - np.asarray(X.dot(w)) 
      
        pred = np.sum(tmp < 0, axis=0).astype(np.int) 
        #print (np.sum(tmp < 0, axis=0).shape, np.sum(tmp < 0, axis=0))
    
    return pred


def pre_proba(X, w, theta, mode):

  
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)

  
    prob = np.pad(
        sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=(0, 1))


    return X.dot(w)
        
    


class OrdinalRg(base.BaseEstimator):

    def __init__(self, alpha=1., verbose=0, hinge_const=[1.,1.,1.,1.,1.], max_iter=1000, kernel='linear', gamma=0.1, loss='IT'):
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma
        self.loss = loss
        self.hinge_const = hinge_const
        if(loss=='IT'):
            self.mode='0-1'
        elif(loss=='AT'):
            self.mode='AE'
        elif(loss=='CL'):
            self.mode='CL'
        elif(loss=='TOR'):
            self.mode='TOR'

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        self.X=X
        y_tmp = y - y.min() 

        X, kernel_constant = prepocess_kernel(self.kernel, X, self.gamma)


        self.coef_, self.theta_ = threshold_fit(
            X, y_tmp, kernel_constant, self.alpha, self.n_class_, mode=self.mode,
            verbose=self.verbose, max_iter=self.max_iter,
            sample_weight=sample_weight, hinge_const=self.hinge_const)

        return self

    def predict(self, X):

        X_new=[]
        if(self.kernel!='linear'):
            for i in X:
                row=[]
                for j in self.X:
                    row.append(kernel1(i,j,self.gamma))

                X_new.append(row)
        else:
            X_new=X

        return threshold_predict(np.array(X_new), self.coef_, self.theta_, self.mode) +\
         self.classes_.min()

    def predict_proba(self, X):
        X_new=[]
        if(self.kernel!='linear'):
            for i in X:
                row=[]
                for j in self.X:
                    row.append(kernel1(i,j,self.gamma))

                X_new.append(row)
        else:
            X_new=X
        return pre_proba(np.array(X_new), self.coef_, self.theta_, self.mode)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return -metrics.mean_absolute_error(
            pred,
            y,
            sample_weight=sample_weight)

