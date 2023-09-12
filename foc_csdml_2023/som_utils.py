import numpy as np
import matplotlib.pyplot as plt

__all__ = ['find_bmus_for_vectors']

def find_bmus_for_vectors(v1, v2, axis=0, dist=False, k=1):
    '''
    axis:
    0, find the bmu for each element of v2 in v1
    1, find the bmu for each element of v1 in v2, which is equivalant to swapping v1 and v2

    dist: return only the indices if False and otherwise both indices and distances

    k: return the nearest k neighbors, note that the first k neighbors are not sorted
    '''
    # the squared euclidean length of each input vector
    x2=np.einsum('ij,ij->i', v2, v2)
    # the squared euclidean length of each som cell vector
    y2=np.einsum('ij,ij->i', v1,v1)
    # the dot project of each input vector to each som cell vector
    d=np.dot(v1, v2.T)
    # the squared euclidean distance
    d = x2[None,...] - 2*d + y2[...,None]
    # partition the result, such that the 2nd item (indice=1) is in the correct place of a sorted list
    # and thus the 1st item is the minimum one

    # see https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition
    # argpartition returns the indice (bmus), while partition returns the partitioned list (dist)
    if axis==0:
        if dist:
            return np.argpartition(d,k-1,axis=0)[:k].T, np.partition(d,k-1,axis=0)[:k].T
        else:
            return np.argpartition(d,k-1,axis=0)[:k].T
    else:
        d=d.T
        if dist:
            return np.argpartition(d,k-1,axis=0)[:k].T, np.partition(d,k-1,axis=0)[:k].T
        else:
            return np.argpartition(d,k-1,axis=0)[:k].T

__all__ += ['normalize']

def normalize(x):
    mean = x.mean(axis=0)[None]
    std = x.std(axis=0)[None]
    std[std==0] = 1
    return (x - mean) / std

__all__ += ['get_forms_to_show']

def get_forms_to_show(features, lattice):
    '''
    returns a list of integers, where the i-th element is the index of the best-matching forms for i-th cell
    '''
    mapsize = lattice.shape[:2]
    dimension = lattice.shape[-1]

    mapsize_ = mapsize[0]*mapsize[1]
    lattice_flatten = lattice.reshape(mapsize_, dimension)

    bmu,dist=find_bmus_for_vectors(lattice_flatten, features, dist=True)
    form_to_show={}
    last_dist={}

    # get the bast-representing unit of each cell
    for i, b, d in zip(range(len(bmu)), bmu, dist):
        b = int(b) # make numpy integer hashable
        if b not in form_to_show.keys():
            form_to_show[b]=i
            last_dist[b]=d
        else:
            if last_dist[b] > d:
                form_to_show[b]=i
                last_dist[b]=d

    form_to_show=[form_to_show[i] if i in form_to_show.keys() else None for i in range(mapsize_)]
    return form_to_show

__all__ += ['render_som']

def render_som(forms, features, lattice, cell_size = 200, clip_size = 50):
    '''
    render the som
    '''
    form_to_show = get_forms_to_show(features, lattice)
    mapsize = lattice.shape[:2]

    # generate plots for the best-representing units
    def _plot_(F):
        fig = plt.figure(figsize=(8,8))
        ax=plt.axes([0,0,0.5,1], projection='3d')
        cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],view='3D-45',thickness_base=0.5,thickness_ratio=0.02,load_len_scale=10)
        plt.axis('off')
        # force matplotlib to draw the figure
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # close matplotlib canvas
        plt.close()
        # return image
        return image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    print('generate cell renderings ...')

    # list of cell images
    cell_images=[]
    blank_cell = np.ones((cell_size,cell_size,3),dtype=np.uint8)*255

    for i in range(len(form_to_show)):
        if form_to_show[i] is None:
            cell_images.append(blank_cell)
        else:
            T, F, Fc = forms[form_to_show[i]]
            cell_img = _plot_(F)
            cell_img = cell_img[clip_size:-clip_size, clip_size:-clip_size]
            cell_images.append(cv2.resize(cell_img,(cell_size, cell_size)))

    # put all cell images together
    rows = [np.hstack(cell_images[i*mapsize[1]:i*mapsize[1]+mapsize[1]]) for i in range(mapsize[0])] # mapsize[0, 1] are rows and columns
    som_rendering = np.vstack(rows)
    return som_rendering
