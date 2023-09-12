import numpy as np

__all__ = ['standardized_moment']
# high-order statistics (https://en.wikipedia.org/wiki/Standardized_moment)

def standardized_moment(x, k):
    '''
    returns the standardized moment of degree k for the input variable x
    https://en.wikipedia.org/wiki/Standardized_moment

    x must be a 1-d or 2-d numpy array. for 2-d array, the standardized_moment is calculated for the columns
    '''

    if k==1:
        return np.zeros_like(x[...,0])
    elif k==2:
        return np.zeros_like(x[...,0]) + 1
    else:
        n=x.shape[-1]

        mean = x.mean(axis=-1).astype(np.float64)
        diff = x - mean if len(x.shape)==1 else x - mean[...,None]

        bottom=(diff**2).sum(axis=-1)**(k/2)

        # this line return 0.0 rather than nan for indetermate inputs
        if hasattr(bottom, '__len__'):
            bottom[bottom==0]=1 # where bottom is 0, top is also 0
        else:
            bottom = 1.0 if bottom == 0.0 else bottom

        top=(diff**k).sum(axis=-1)
        coeff=n**(k/2-1)
    return coeff*top/bottom

# hos base functions, input values for the "method" of create_HOS_feature_extractor
get_skewness=lambda x:standardized_moment(x,3)[...,None] # degree 3 standardized_moment
get_kurtosis=lambda x:standardized_moment(x,4)[...,None] # degree 4 standardized_moment
sm5=lambda x:standardized_moment(x,5)[...,None]
sm6=lambda x:standardized_moment(x,6)[...,None]
sm7=lambda x:standardized_moment(x,7)[...,None]
sm8=lambda x:standardized_moment(x,8)[...,None]

get_std = lambda x:x.std(axis=-1)[...,None] # standard deviation
get_mean = lambda x:x.mean(axis=-1)[...,None] # mean
get_min = lambda x:x.min(axis=-1)[...,None]
get_max = lambda x:x.max(axis=-1)[...,None]

base_functions = [get_std, get_mean, get_min, get_max, get_skewness, get_kurtosis]#, sm5, sm6, sm7, sm8]

__all__ += ['feature_xyz']

def feature_xyz(forms):
    '''
    feature vector using node's (x, y, z) coordinates
    '''
    form_hos = [np.concatenate([f(np.asarray([coord[xyz] for coord in F['coords']])) for f in base_functions for xyz in range(3)],axis=-1) for T, F, Fc in forms]
    return np.asarray(form_hos)

__all__ += ['feature_edge_force']

def feature_edge_force(forms):
    '''
    feature vector using edge's force magnitude
    '''
    form_hos = [np.concatenate([f(np.asarray(F['edge_forces'])) for f in base_functions],axis=-1) for T, F, Fc in forms]
    return np.asarray(form_hos)

def _edge_lengths(F):
    '''
    compute edge lengths of a form
    '''
    coords = np.asarray(F['coords'])
    edges = np.asarray(F['edges'])

    starts = coords[edges[..., 0]]
    ends =coords[edges[..., 1]]

    dist = np.linalg.norm(ends - starts,axis=-1)
    return dist

__all__ += ['feature_edge_length']

def feature_edge_length(forms):
    '''
    feature vector using edge's length
    '''
    form_hos = [np.concatenate([f(_edge_lengths(F)) for f in base_functions],axis=-1) for T, F, Fc in forms]
    return np.asarray(form_hos)

__all__ += ['feature_load_path']

def feature_load_path(forms):
    '''
    feature vector using form's total load path
    '''
    form_hos = [np.concatenate([f(_edge_lengths(F) * np.asarray(F['edge_forces'])) for f in base_functions],axis=-1) for T, F, Fc in forms]
    return np.asarray(form_hos)

__all__ += ['get_feature_hos']

def get_feature_hos(forms):
    # coordniates + edge forces + edge lengths + load path
    features = [feature_xyz,
                feature_edge_force,
                feature_edge_length,
                feature_load_path]
    return np.concatenate([f(forms) for f in features],axis=-1)
