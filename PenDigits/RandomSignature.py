import numpy as np
from tqdm.auto import tqdm


def sigmoid(x):
    return 1/(1+np.exp(-x))

hyperparams_dict = {
'varA':0.3,
'mean':0,
'res_size':10,
'activation': sigmoid
}

def get_random_coeff(d,hparams = hyperparams_dict):
# hyperparam_dict: contains mean and var of normal 
# distribution that we want to sample as well as dimension
# of randomized signature.
# d: dimension of the control, and therefore number 
# of different matrix,vector pairs of coeff to be generated
# output: tuple(As,bs), with As np array (d,res_size,res_size)
# and bs np array (d,res_size). As[i,:,:] and bs[i,:] 
# are the i-th components of the vector field generating the 
# rand signature.
 
    random_projection = []
    random_bias = []
    for i in range(d):


        projection = np.random.normal(hparams['mean'],np.sqrt(hparams['varA']),
            (hparams['res_size'], hparams['res_size']))
        
        norm = np.linalg.norm(projection, 2)
        #why are we normalizing to 0.99?
        projection = projection# / norm * 0.99
        random_projection.append(projection)

        random_bias.append(np.random.normal(hparams['mean'], np.sqrt(hparams['varA']), size=hparams['res_size']))

    return np.array(random_projection), np.array(random_bias)



def compute_signature(As,bs,path,trajectory = False,hparams = hyperparams_dict):
# Solve the IVP given by r-Sig_0 = (1,...,1) (may need to change this) and 
# dr-Sig = sigma(As[i,:,:]r-Sig+b[i,:])dpath[i,:]. Working for a single path, that is
# if we have a batch of paths we have to call this function once for each path.
# As,bs,path: coeffeicient of the above ODE
# trajectory: bool, if False only return the value of r-Sig_T, where the T
# is the final time of path  (i.e. len(path[0])). if True, then the whole
# trajectory r-Sig_t t=0,...,T is returned.
# hparams: dictionary of parameters containing activation function (may be replaced
# with just passing the activation function) 
# output: if trajectory is False, return vector of length As.shape[1] corresponding to r-Sig_T.
# If trajectory is True, return matrix (r-Sig_0,...,r-Sig_T) of shape (len(dX)+1,As.shape[1])

    dX = np.diff(path,axis = 0)
    
    if trajectory:
        Sig = np.zeros((len(dX)+1,As.shape[1]))
        Sig[0,:] = np.ones(As.shape[1])
        for i in range(len(dX)):
            Sig[1+i,:] = Sig[i,:]+hparams['activation'](np.dot(As, Sig[i,:]) + bs).T.dot(dX[i])
    else:
        Sig = np.ones(As.shape[1])
        for i in range(len(dX)):
            dSig = hparams['activation'](np.dot(As, Sig) + bs).T.dot(dX[i])
            Sig += dSig
            
    return Sig

def compute_signature_vect(As,bs,paths,hparams = hyperparams_dict):
## same as compute_signature. Only difference being the use of the einsum function. Can be used only if all the
## paths have the same length and in that case it is faster than compute_signature. The other hyperparamers are as
## in compute_signature. 

    dX = np.diff(paths, axis = 1)
    Sig = np.ones((paths.shape[0],As.shape[1]))
    for i in tqdm(range(dX.shape[1])):
        #Einstein notation where b: batch_index, i: index of path coordinate, j: index of res_dim, k: row of the matrix A
        temp = np.einsum('ijk,bk->bij',As,Sig,optimize = True)
        temp = hparams['activation'](temp+bs)
        Sig += np.einsum('bij,bi -> bj',temp,dX[:,i,:],optimize = True)
    
    return Sig    


def get_signature(paths,trajectory = False,vect = False,**kwargs):
#Return the Signature for all the path in paths.
#paths: batch of paths for which we want to compute the Signature.
#trajectory: see above (function compute_signature)
#output: depending on trajectory, ndarray of shape [len(paths),dim(r-Sig)], where
# the dimension of r-Sig depends on the value of trajectory as described above.
# if vect, paths should be an array containing *all* paths and the
# signature is computed using np.einsum (faster)
    
    if vect:
        #paths dimension
        paths_dim = paths.shape[2]
        # generate vector fields
        [As,bs] = get_random_coeff(d = paths_dim,**kwargs)
        # compute signature
        Sigs = compute_signature_vect(As,bs,paths,**kwargs)
    
    else:
        # paths dimension
        paths_dim = paths[0].shape[1]
        # generate vector fields
        [As,bs] = get_random_coeff(d = paths_dim,**kwargs)
    
        #Compute Signature for all observations
        Sigs = np.array([compute_signature(As,bs,paths[i],trajectory,**kwargs) 
                        for i in tqdm(range(len(paths)))])
    return Sigs


    
