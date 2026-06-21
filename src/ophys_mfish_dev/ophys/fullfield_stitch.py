import numpy as np
from tifffile import imread
from PIL import Image
from PIL.TiffTags import TAGS
import json


#############################################
# Full field
#############################################

def read_si_fullfield_metadata(fullfield_fn):
    '''Reads metadata from a ScanImage full-field tiff file.
    
    Args:
        fullfield_fn: str or Path, path to the full-field tiff file
        
    Returns:
        num_slices: int, number of slices in the z-stack
        num_volumes: int, number of volumes in the z-stack
        num_columns: int, number of columns in the full-field image
    '''
    
    with Image.open(fullfield_fn) as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    
    num_slices_ind = np.where(['SI.hStackManager.actualNumSlices = ' in x for x in meta_dict['Software'][0].split('\n')])[0][0]
    num_slices_txt = meta_dict['Software'][0].split('\n')[num_slices_ind]
    num_slices = int(num_slices_txt.split('= ')[1])
    
    num_volumes_ind = np.where(['SI.hStackManager.actualNumVolumes = ' in x for x in meta_dict['Software'][0].split('\n')])[0][0]
    num_volumes_txt = meta_dict['Software'][0].split('\n')[num_volumes_ind]
    num_volumes = int(num_volumes_txt.split('= ')[1])

    artist_json = json.loads(meta_dict['Artist'][0])
    num_columns = len(artist_json['RoiGroups']['imagingRoiGroup']['rois'])

    return num_slices, num_volumes, num_columns


def stitch_fullfield(fullfield_fn, channels=[0]):
    '''Stitches a full-field tiff file.

    Args:
        fullfield_fn: str or Path, path to the full-field tiff file
        channels: list of int, channels to stitch
            Only applicable if the full-field tiff file contains multiple channels

    Returns:
        fullfield_stitched: 2D or 3D array, stitched full-field image
    '''
    num_slices, num_volumes, num_columns = read_si_fullfield_metadata(fullfield_fn)
    fullfield_all = imread(fullfield_fn)
    assert fullfield_all.shape[0] == num_slices * num_volumes
    num_rows = int((fullfield_all.shape[1]+1) / num_columns)
    fullfield_stitched = []
    if len(fullfield_all.shape) == 4:
        for channel in channels:
            fullfield = fullfield_all[:,channel,:,:]
            fullfield_stitched.append(_stitch(fullfield, num_slices, num_columns, num_rows))
    else:
        fullfield_stitched.append(_stitch(fullfield_all, num_slices, num_columns, num_rows))
    if len(fullfield_stitched) == 1:
        fullfield_stitched = fullfield_stitched[0]
    else:
        fullfield_stitched = np.stack(fullfield_stitched, axis=0)
    return fullfield_stitched


def _stitch(fullfield, num_slices, num_columns, num_rows):
    ind = np.hstack([np.arange(i, fullfield.shape[0], num_slices) for i in range(num_slices)])
    fullfield_ = np.concatenate([fullfield[ind,:,:],np.zeros((fullfield.shape[0],1,fullfield.shape[2]))], axis=1) 
    fullfield_ = np.concatenate([fullfield_[:, i*num_rows : (i+1)*num_rows, :] for i in range(num_columns)],axis=2)
    im = fullfield_.mean(axis=0)
    return im


#####################################
# Medium-size full-field z-stack
# E.g., for checking local injection
#####################################
def fullfield_zstack(fn):
    ''' Make fullfield zstack
    fn: file path
    return
        ff_zstack: volume-averaged fullfield z-stack
    '''
    num_slices, num_volumes, num_columns = read_si_fullfield_metadata(fn)
    fullfield_all = imread(fn)

    num_rows = int((fullfield_all.shape[1]+1) / num_columns)
    ind = np.hstack([np.arange(i, fullfield_all.shape[0], num_slices) for i in range(num_slices)])
    fullfield_ = np.concatenate([fullfield_all[ind,:,:],np.zeros((fullfield_all.shape[0],1,fullfield_all.shape[2]))], axis=1) 
    fullfield_ = fullfield_all[ind,:,:]
    fullfield_stitched = np.concatenate([np.roll(fullfield_[:, i*num_rows : (i+1)*num_rows, :], -i*12, axis=1) for i in range(num_columns)],axis=2)

    ff_zstack = np.stack([fullfield_stitched[i*num_slices : (i+1)*num_slices, :, :].mean(axis=0) for i in range(num_volumes)])

    # leave the old code, just in case
    # repeat = fullfield_stitched.shape[0] / num_slices
    # assert repeat == int(repeat)
    # repeat = int(repeat)

    # ff_zstack = np.stack([fullfield_stitched[i*num_slices : (i+1)*num_slices, :, :].mean(axis=0) for i in range(repeat)])
    

    return ff_zstack