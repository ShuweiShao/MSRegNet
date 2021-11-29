"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np
import cv2

def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def miccai2018_gen_s2s(gen, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        X = next(gen)[0]
        Y = next(gen)[0]
        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])


def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """
    zeros = None
    while True:
        idxes = np.random.randint(len(vol_names)-3, size=batch_size)
        """
        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        """
        X = []
        Y = []
        for idx in idxes:
            x = cv2.imread(vol_names[idx], 0)
            x = x[0:1024, :]
            #x = cv2.resize(x, (640, 1024), interpolation=cv2.INTER_CUBIC)
            x = x / 255.0
            x = x[np.newaxis, ..., np.newaxis]
            y = cv2.imread(vol_names[idx + 3], 0)
            #y = cv2.resize(y, (640, 1024), interpolation=cv2.INTER_CUBIC)
            y = y[0:1024, :]
            y = y / 255.0
            y = y[np.newaxis, ..., np.newaxis]
            X.append(x)
            Y.append(y)

        if batch_size > 1:
            return_vals1 = [np.concatenate(X, 0)]
            return_vals2 = [np.concatenate(Y, 0)]
            return_vals = [return_vals1[0], return_vals2[0]]
        else:
            return_vals = [X[0],Y[0]]

        if zeros is None:
            volshape = return_vals[0].shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))

        yield (return_vals, [return_vals[1], zeros])


def load_example_by_name(vol_name, seg_name, np_var='vol_data'):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name, np_var)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')
        import nibabel as nib
        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X
