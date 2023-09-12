import cem_mini
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['bridge_topology', 'plot_bridge', 'basic_bridge_generator', 'basic_bridge_generator_sampler']

def bridge_topology(nr,
                    length = 10,
                    width = 10,
                    height = -4,
                    left_dev_mags = [3,-3],
                    center_dev_mags = [-50,10],
                    right_dev_mags = None,
                    slab_loads = [0,0,-2],
                    no_outer_dev=True):
    '''
    create a bridge from topology like this

    <=====1 --- 4 ===> support
         / \   / \
    <===/==2--/--3===> slab 2
    <==0-----5=======> slab 1

    where == are the trail edges, -- and / are deviation edges, < and > shows the force flow directions

    ---- parameters ----
    nr:              number of nodes per trail paths (number of rows), must >=2

    length:          a number or cem_mini.samplers.
                     generate the x coord of the inital nodes, also used to generate the constrained planes

    width:           a number or cem_mini.samplers.
                     generate the y coord of the inital nodes

    height:          a number or cem_mini.samplers.
                     generate the z coord of the inital nodes

    left_dev_mags:   list of 2 / 3 / 6 elements. The elements can be either
                     numbers or cem_mini.samplers.

                     when 6 elements [A1, B1, C1, A2, B2, C2] are given,
                     [A1, A2] will be used as the start and end dev_mag between slab 1 and slab 2,
                     [B1, B2] will be used as the start and end dev_mag between slab 1 and support,
                     [C1, C2] will be used as the start and end dev_mag between slab 2 and support,

                     when 3 elements [A, B, C] are given,
                     A will generate [A1, A2], B for [B1, B2], and C for [C1, C2]

                     when 2 elements [A, B] are given,
                     A will generate [A1, A2], B for [B1, B2], and C1 = B1, C2 = B2

    center_dev_mags: list of 2 / 3 elements.
                     The elements can be either numbers or cem_mini.samplers.

                     when 3 elements [A, B, C] are given,
                     A will be the dev_mag of edge [1-4]
                     B will be the dev_mag of edge [0-5]
                     C will be the dev_mag of edge [2-3]

                     when 2 elements [A, B] are given, C = B

    right_dev_mags:  same definition of left_dev_mags,
                     except that it controls the right side of the bridge.
                     If None is given, the left_dev_mags will be used (symmitrical bridge)

    slab_loads:      loads (list of 3 numbers) of all slab nodes

    '''

    nt=6 # 6 trail paths

    # axial graph with 6 trail edges
    # 1 - 4 will be the support (arch or cable)
    # 0 - 5 will be one side of the slab
    # 2 - 3 will be another side of the slab
    graph = cem_mini.axial_graph(nt,nr, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], no_outer_dev)

    # sample the l, w, h values
    if hasattr(length,'__call__'):
        length=length()

    if hasattr(width,'__call__'):
        width=width()

    if hasattr(height,'__call__'):
        height=height()

    # sampling left_dev_mags
    if len(left_dev_mags)==2 or len(left_dev_mags)==3 or len(left_dev_mags)==6:
        if len(left_dev_mags)==2:
            # A, B, B, A, B, B
            left_dev_mags=[left_dev_mags[0], left_dev_mags[1], left_dev_mags[1], left_dev_mags[0], left_dev_mags[1], left_dev_mags[1]]
        elif len(left_dev_mags)==3:
            # A, B, C, A, B, C
            left_dev_mags=left_dev_mags + left_dev_mags

        # convert to number if sampler is given
        left_dev_mags=[i() if hasattr(i,'__call__') else i for i in left_dev_mags]
    else:
        raise Exception ('left_dev_mags must have a length of 2, 3 or 6')

    if len(center_dev_mags)==2:
        center_dev_mags=[center_dev_mags[0],center_dev_mags[1],center_dev_mags[1]]

    if len(center_dev_mags)==3:
        center_dev_mags=[i() if hasattr(i,'__call__') else i for i in center_dev_mags]
    else:
        raise Exception ('center_dev_mags must have a length of 2 or 3')

    # sampling right_dev_mags
    if right_dev_mags is None:
        right_dev_mags = left_dev_mags
    else:
        if len(right_dev_mags)==2 or len(right_dev_mags)==3 or len(right_dev_mags)==6:
            if len(right_dev_mags)==2:
                # A, B, B, A, B, B
                right_dev_mags=[right_dev_mags[0], right_dev_mags[1], right_dev_mags[1], right_dev_mags[0], right_dev_mags[1], right_dev_mags[1]]
            elif len(right_dev_mags)==3:
                # A, B, C, A, B, C
                right_dev_mags=right_dev_mags + right_dev_mags

            # convert to number if sampler is given
            right_dev_mags=[i() if hasattr(i,'__call__') else i for i in right_dev_mags]
        else:
            raise Exception ('right_dev_mags must have a length of 2, 3 or 6')

    num_nodes=graph['num_nodes']
    trail_paths=graph['trail_paths']

    # trail lengths will be overrided by contrained planes, therefore the values do not matter
    trail_length = [[0.1] * (nr-1)] * 6

    # in addition, deviation edges [[0, 5], [1, 4], [2, 3]] will be added
    # to connect the trail paths and the deviation edges of the two sides of the bridge
    deviation_edges=graph['deviation_edges'] + [(0, 5), (1, 4), (2, 3)]
    dev_force_mag = np.zeros(len(deviation_edges), dtype=np.float32) # default dev force mag

    # retrieve deviation edge indexes and the generate the corresponding dev_mag values
    # a linear interpolation will be made for the dev_mag if start != end
    left_support_slab1 = [i * 6 for i in range((nr - 1) if no_outer_dev else nr)]
    left_support_slab1_dev = np.linspace(left_dev_mags[1], left_dev_mags[4], (nr - 1) if no_outer_dev else nr)

    left_support_slab2 = [i * 6 + 1 for i in range((nr - 1) if no_outer_dev else nr)]
    left_support_slab2_dev = np.linspace(left_dev_mags[2], left_dev_mags[5], (nr - 1) if no_outer_dev else nr)

    left_slab1_slab2 = [i * 6 + 2 for i in range((nr - 1) if no_outer_dev else nr)]
    left_slab1_slab2_dev = np.linspace(left_dev_mags[0], left_dev_mags[3], (nr - 1) if no_outer_dev else nr)

    right_support_slab1 = [i * 6 + 4 for i in range((nr - 1) if no_outer_dev else nr)]
    right_support_slab1_dev = np.linspace(right_dev_mags[1], right_dev_mags[4], (nr - 1) if no_outer_dev else nr)

    right_support_slab2 = [i * 6 + 3 for i in range((nr - 1) if no_outer_dev else nr)]
    right_support_slab2_dev = np.linspace(right_dev_mags[2], right_dev_mags[5], (nr - 1) if no_outer_dev else nr)

    right_slab1_slab2 = [i * 6 + 5 for i in range((nr - 1) if no_outer_dev else nr)]
    right_slab1_slab2_dev = np.linspace(right_dev_mags[0], right_dev_mags[3], (nr - 1) if no_outer_dev else nr)

    # assigning the values
    dev_force_mag[left_support_slab1] = left_support_slab1_dev
    dev_force_mag[left_support_slab2] = left_support_slab2_dev
    dev_force_mag[left_slab1_slab2] = left_slab1_slab2_dev

    dev_force_mag[right_support_slab1] = right_support_slab1_dev
    dev_force_mag[right_support_slab2] = right_support_slab2_dev
    dev_force_mag[right_slab1_slab2] = right_slab1_slab2_dev

    dev_force_mag[-3] = center_dev_mags[1] # edge[0,5], slab1
    dev_force_mag[-2] = center_dev_mags[0] # edge[1,4], support
    dev_force_mag[-1] = center_dev_mags[2] # edge[2,3], slab2

    T=cem_mini.create_topology(num_nodes)
    # trail_length will be completely overrided by constrained planes
    cem_mini.set_trail_paths(T, trail_paths, trail_length)
    cem_mini.set_deviation_edges(T,deviation_edges,dev_force_mag)

    # initial nodes
    x = length / 2
    y = width / 2
    z = height

    node_pos = {
                0:[-x, -y, 0], 1:[-x, 0, z], 2:[-x, y, 0],
                3:[x, y, 0], 4:[x, 0, z], 5:[x, -y, 0],
               }

    cem_mini.set_original_node_positions(T,node_pos)

    # constrained planes (-x or x direction based on the trail index)
    constrained_planes = {(i*nt+j):[(-x-length*i if j<3 else x+length*i), 0, 0, (-1 if j<3 else 1), 0, 0] for i in range(nr) for j in range(nt)}
    cem_mini.set_constrained_planes(T,constrained_planes)

    loads = {(i*nt+j):(slab_loads if j%3!=1 else [0,0,0]) for i in range(nr) for j in range(nt)}
    cem_mini.set_node_loads(T,loads)

    return T

def plot_bridge(F, figsize=(8,4)):
    '''
    plot the form diagram of the bridge
    '''
    plt.figure(figsize=figsize)
    ax=plt.axes([0,0,0.5,1], projection='3d')
    cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],F['loads'],view='3D-45',thickness_base=0.5,thickness_ratio=0.02,load_len_scale=10)
    plt.axis('on')

    ax=plt.axes([0.5,0,0.5,1], projection='3d')
    cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],F['loads'],view='2D-XZ',thickness_base=0.5,thickness_ratio=0.02,load_len_scale=10)
    plt.axis('on')

    plt.show()


# the wrapper function of bridge_topology which simplifies and merges some parameters

def basic_bridge_generator(nd_num, a_left, b_left, a_center, b_center, height):
    '''
    the wrapper function of bridge_topology which simplifies and merges some parameters

    the function takes 6 numerical inputs and produces the correspinding bridge geometry

    nd_num:       number of nodes

    a_left:       the bounding value A of the force magnitude of the deviation edges
                  on the left side of the bridge, see the "left_dev_mags" parameter in
                  "bridge_topology"

    b_left:       the bounding value B of the force magnitude of the deviation edges
                  on the left side of the bridge, see the "left_dev_mags" parameter in
                  "bridge_topology"

    a_center:     the bounding value A of the force magnitude of the deviation edges
                  at the bridge center, see the "center_dev_mags" parameter in
                  "bridge_topology"

    b_center:     the bounding value B of the force magnitude of the deviation edges
                  at the bridge center, see the "center_dev_mags" parameter in
                  "bridge_topology"

    height:       height og the bridge
    '''
    T = bridge_topology(int(np.round(nd_num)), left_dev_mags=[a_left,b_left], center_dev_mags=[a_center,b_center], height=height)

    # general information about the generator
    T['generator']={'name':'basic_bridge_generator','version':'0.1.0'}
    # numerical value of the parameters
    T['generator']['param']={'nd_num':nd_num, 'a_left':a_left, 'b_left':b_left, 'a_center':a_center,'b_center':b_center,'height':height}
    # position of the parameters
    T['generator']['param_pos']={'nd_num':0, 'a_left':1, 'b_left':2, 'a_center':3,'b_center':4,'height':5}
    # parameters as list
    T['generator']['param_list']=[nd_num, a_left, b_left, a_center, b_center, height]

    # semantic information that will be used by the machine learning steps
    a=a_left>0
    b=b_left>0
    c=a_center>0
    d=b_center>0

    if a and c:
        T['labels']={'typology':['underspanned suspension bridge','upper deck suspension bridge','suspension bridge with main cable below the deck']}
    elif a and d:
        T['labels']={'typology':['deck arch bridge', 'arch bridge with arch below the deck']}
    elif b and d:
        T['labels']={'typology':['tied arch bridge', 'arch bridge with arch above the deck']}
    else: # b and c
        T['labels']={'typology':['suspension bridge', 'suspension bridge with main cable above the deck']}
    F, Fc = cem_mini.CEM(T)
    return T, F, Fc

def basic_bridge_generator_sampler(num=1, # decision_function=lambda *x:True,
                                   seed=None,
                                   nd_num_=[3, 8],
                                   a_left_=[.5, 10], b_left_=[.5, 10],
                                   a_center_=[1,100], b_center_=[1,100],
                                   height_=[1,80]):
    '''
    sampler of the bridge generator

    num:                how many groups of parameters to be sampled

    seed:               random seed

    others:             pre-defined sampling range that are not intended to be changed

    returns:            the input parameter for basic_bridge_generator
    '''
    if seed is not None:
        np.random.seed(seed)

    sample = lambda x:(x[1]-x[0])*np.random.random()+x[0]

    params=[]

    while len(params)<num:
        nd_num = np.random.randint(nd_num_[1]-nd_num_[0])+nd_num_[0]
        a_left=sample(a_left_)
        b_left=sample(b_left_)
        a_center=sample(a_center_)
        b_center=sample(b_center_)
        height=sample(height_)

        bridge_type=np.random.randint(4)

        if bridge_type==0:
            # deck arch bridges / arch bridge with arch at the bottom
            b_left*=-1
            a_center*=-1
            height*=-1
        elif bridge_type==1:
            # suspension bridge
            a_left*=-1
            b_center*=-1
        elif bridge_type==2:
            # deck truss bridge?
            b_left*=-1
            b_center*=-1
            height*=-1
        else:
            # Tied-Arch Bridge / arch bridge with arch on top
            a_left*=-1
            a_center*=-1

#         if decision_function(nd_num, a_left, b_left, a_center, b_center, height):
#             params.append([nd_num, a_left, b_left, a_center, b_center, height])
        params.append([nd_num, a_left, b_left, a_center, b_center, height])

    return params
