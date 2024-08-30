import numpy as np
from pathlib import Path
import skimage
from stackreg import StackReg

from lamf_analysis.ophys import zstack

######################################
# Session to session drift calculation

def calculate_session_to_session_diff(opid_1, opid_2, n_averaging_planes=10, sr_method='affine'):
    ''' Calculate the number of planes to shift to align two sessions
    From the second session (opid_2) to the first session (opid_1)

    Args:
        opid_1: int, ophys plane index of the first session
        opid_2: int, ophys plane index of the second session

    Returns:
    results_info: dict, contains the following keys
        'corrcoef_arr_single': np.array (1d), correlation coefficient between 
            the registered middle plane of the second session z-stack
            and the registered first session z-stack
        'fov_reg': np.array, registered middle plane of the second session z-stack
        'best_tmat': np.array, transformation matrix for the registration
        'tmat_list': list, list of transformation matrices to each plane of the first session z-stack
            best_tmat is the one that has the highest correlation coefficient
        'temp_cc': list, correlation coefficient between 
            the second session and 
            the first session z-stack
            using optimal transformation per plane (of the first z-stack)
        'corrcoef_mat': np.array (2d), correlation coefficient between
            each plane of the second session z-stack after transformation 
            and each plane of the first session z-stack
        'mean_diag': np.array, mean of the diagonal of
            the correlation coefficient matrix (corrcoef_mat)
        'stack_to_stack_diff_planes': int, number of planes to shift the second stack
            to align to the first stack
        'sr_method': str, method used for stack registration
    '''

    if opid_1 == opid_2:
        return None

    if sr_method == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif sr_method == 'rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)
    else:
        raise ValueError('"sr_method" should be either "affine" or "rigid_body"')
        
    stack_1 = get_decrosstalked_registered_local_zstack(opid_1)
    stack_1 = med_filt_z_stack(stack_1)
    stack_1 = zstack.rolling_average_stack(stack_1, n_averaging_planes=n_averaging_planes)
    stack_2 = get_decrosstalked_registered_local_zstack(opid_2)
    stack_2 = med_filt_z_stack(stack_2)
    stack_2 = zstack.rolling_average_stack(stack_2, n_averaging_planes=n_averaging_planes)

    fov_2 = stack_2[len(stack_2)//2]
    corrcoef_arr_single, fov_reg, best_tmat, tmat_list, temp_cc = fov_stack_register_stackreg(fov_2, stack_1)

    registered_stack = np.zeros_like(stack_2)
    for zi, zstack_plane in enumerate(stack_2):
        registered_stack[zi] = sr.transform(zstack_plane, tmat=best_tmat)

    corrcoef_mat = corrcoef_stack(stack_1, registered_stack)
    delay_range = range(-corrcoef_mat.shape[0]//2 + 1, corrcoef_mat.shape[0]//2 + 1)
    mean_diag = np.zeros(len(delay_range))
    for i in range(len(delay_range)):
        mean_diag[i] = np.mean(np.diag(corrcoef_mat, delay_range[i]))
    stack_to_stack_diff_planes = -delay_range[np.argmax(mean_diag)]  # be careful about the sign!

    results_info = {'corrcoef_arr_single': corrcoef_arr_single,
                    'fov_reg': fov_reg,
                    'best_tmat': best_tmat,
                    'tmat_list': tmat_list,
                    'temp_cc': temp_cc,
                    'corrcoef_mat': corrcoef_mat,
                    'mean_diag': mean_diag,
                    'stack_to_stack_diff_planes': stack_to_stack_diff_planes,  # be careful about the sign!
                    'sr_method': sr_method}  
    return results_info


def get_decrosstalked_registered_local_zstack(opid):
    ''' Get decrosstalked and registered local zstack for a given opid
    This should be specific for pipeline or data structure

    Currently only works in this capsule (id: )
    '''
    load_dir = Path('/root/capsule/scratch/decrosstalked_zstacks')
    stack_fn = load_dir / f'{opid}_decrosstalked_local_zstack_reg.npy'
    assert stack_fn.exists()
    stack = np.load(stack_fn)
    return stack


def fov_stack_register_stackreg(fov, stack, use_clahe=True, sr_method='affine', tmat=None, use_valid_pix=True):
    ''' Register a field of view (fov) to a stack of z-stack using StackReg

    Parameters:
        fov: np.array, field of view to register
        stack: np.array, reference z-stack
        use_clahe: bool, whether to use CLAHE for normalization
        sr_method: str, method for stack registration
        tmat: np.array, transformation matrix for the registration
        use_valid_pix: bool, whether to remove blank pixels after transformation
    
    Returns:
        corrcoef_arr: np.array (1d), correlation coefficient between 
            each plane of the stack and the registered fov
        fov_reg: np.array, registered fov
        best_tmat: np.array, transformation matrix for the registration
        tmat_list: list, list of transformation matrices to each plane of the stack
            best_tmat is the one that has the highest correlation coefficient
        temp_cc: list, correlation coefficient between 
            the stack and the registered fov
            using optimal transformation per plane (of the stack)
    '''

    if use_clahe:
        fov_for_reg = zstack.image_normalization(skimage.exposure.equalize_adapthist(
            fov.astype(np.uint16)))  # normalization to make it uint16
        stack_for_reg = np.zeros_like(stack)
        for pi in range(stack.shape[0]):
            stack_for_reg[pi, :, :] = zstack.image_normalization(
                skimage.exposure.equalize_adapthist(stack[pi, :, :].astype(np.uint16)))
    else:
        fov_for_reg = fov.copy()
        stack_for_reg = stack.copy()

    if sr_method == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif sr_method == 'rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)
    else:
        raise ValueError('"sr_method" should be either "affine" or "rigid_body"')

    assert fov.min() >= 0
    if use_valid_pix:
        valid_pix_threshold = fov.min()/10 # to remove blank pixels after transformation
        # valid_pix_threshold = 10 # to remove blank pixels after transformation
    else:
        valid_pix_threshold = -1 # to include all pixels
    num_pix_threshold = fov.shape[0] * fov.shape[1] / 2

    corrcoef_arr = np.zeros(stack.shape[0])

    temp_cc = []
    if tmat is None:        
        tmat_list = []
        for zi in range(stack_for_reg.shape[0]):
            zstack_plane_clahe = stack_for_reg[zi]
            zstack_plane = stack[zi]
            tmat = sr.register(zstack_plane_clahe, fov_for_reg)
            fov_reg = sr.transform(fov, tmat=tmat)            
            valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
            if len(valid_y) > num_pix_threshold:
                temp_cc.append(np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                           fov_reg[valid_y, valid_x].flatten())[0,1])
                tmat_list.append(tmat)
            else:
                temp_cc.append(0)
                tmat_list.append(np.eye(3))
        temp_ind = np.argmax(temp_cc)
        best_tmat = tmat_list[temp_ind]
    else:
        best_tmat = tmat
    fov_reg = sr.transform(fov, tmat=best_tmat)
    valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
    for zi, zstack_plane in enumerate(stack):
        corrcoef_arr[zi] = np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                   fov_reg[valid_y, valid_x].flatten())[0,1]
    return corrcoef_arr, fov_reg, best_tmat, tmat_list, temp_cc


######################################
# Within and across z-drift
#
# With sessions more than 2, session to session drift is first calculated 
# using the first 1 min episodic mean FOV between a pair using the above code.
# For the entire container, each session is aligned to the first session
# Difference between the first session and another is calculated by
# the mean of length < 3 differences.
# E.g., Session 1 to session 4 difference is calculated by
# Avg([(Session 1 to session 4), *([Session 1 to session n to session 4 for n not in (1,4)])])


def get_container_stack_matching_ordered(opids):
    # sort opids first
    # Assume opids are ordered in time
    # TODO: add validation of the order
    assert np.diff(opids).min() > 0

    matched_inds = np.zeros((len(opids), len(opids)))
    for i in range(len(opids)-1):
        for j in range(i+1, len(opids)):
            matched_inds[i, j] = get_matched_ind(opids[i], opids[j])  # be careful about the sign!
            matched_inds[j, i] = -matched_inds[i, j]
    # from the first opid to the rest
    relative_matched_inds = np.zeros(len(opids))
    for i in range(len(opids)-1):
        for j in range(i+1, len(opids)):
            diff_ac = [matched_inds[i, j]]
            for k in range(len(opids)):
                if k == i or k == j:
                    continue
                diff_ac.append(matched_inds[i, k] + matched_inds[k, j])
            relative_matched_inds[j] = np.mean(diff_ac)
    return relative_matched_inds


def get_matched_ind(opid_1, opid_2):
    ''' TODO: change this to a correct path in the (future) pipeline
    '''
    load_dir = Path('/root/capsule/scratch/session_to_session_diff')
    sts_fn = load_dir / f'{opid_1}_{opid_2}_session_to_session_diff.npy'
    sts = np.load(sts_fn, allow_pickle=True).item()
    return sts['stack_to_stack_diff_planes'] # be careful about the sign!


def calculate_within_across_zdrift(opids):
    ''' Calculate within and across session z-drift for a container
    TODO: change paths to the correct ones in the pipeline
    '''
    stack_matched_inds = get_container_stack_matching_ordered(opids)

    within_session_zdrift = []
    success_session_inds = []  # To deal with occasional failures
    for i, opid in enumerate(opids):
        zdrift_dir = Path('/root/capsule/scratch/zdrift')
        opid_dir = zdrift_dir/ str(opid)
        load_fn = opid_dir/f'{opid}_zdrift_pconly_dc_single_onemeanEMF.npy'
        if load_fn.exists():
            results = np.load(load_fn, allow_pickle=True).item()
            within_session_zdrift.append(results['matched_plane_indices'])
            success_session_inds.append(i)
        else:
            within_session_zdrift.append(np.nan)

    within_across_zdrift = [wz + az for wz, az in zip(within_session_zdrift, stack_matched_inds)]
    return within_across_zdrift, success_session_inds


def plot_within_across_zdrift(within_across_zdrift, success_session_inds,
                              zstack_interval=0.75, ax=None):
    ref_session_ind = success_session_inds[0]

    if ax is None:
        fig, ax = plt.subplots()
    
    ref_ind = within_across_zdrift[ref_session_ind][0]
    for i, zdrift in enumerate(within_across_zdrift):
        if i not in success_session_inds:
            continue
        ax.plot(np.arange(i, i+1, 1/len(zdrift)), (zdrift - ref_ind)* zstack_interval)
    ax.set_xlabel('Session #')
    ax.set_ylabel('Relative depth (um)')
    ax.set_title(f'Container depth order {container_depth_order}')

    return ax