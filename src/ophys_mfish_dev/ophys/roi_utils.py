import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brain_observatory_qc.data_access.from_lims as from_lims
import matplotlib
from skimage import measure
from typing import Union, Tuple
import cv2


###################################################################################################
# I/O
###################################################################################################


def napari_to_rois_list(path):
    """Load a napari labeled image into a numpy array
    """
    img = tif.imread(path)

    # take each unique value and make it a binary mask, collect all masks in a list
    roi_masks = []
    for i in np.unique(img):
        mask = np.zeros(img.shape)
        mask[img == i] = 1
        roi_masks.append(mask)

    return roi_masks


def load_napari_tif(path):
    """Load a napari labeled image into a numpy array"""
    img = tif.imread(path)

    return img


def s2p_rois_to_mask(stat, output_shape=None):
    """Take a stat file from suite2p and convert it to list of roi masks"""
    
    if output_shape is None:
        output_shape = (512, 512)
        print(f"output_shape for roi canvas not specified, using {output_shape}")
    roi_masks = []
    for i in range(len(stat)):
        mask = np.zeros(output_shape)
        soma_crop = stat[i]["soma_crop"]

        # use soma crop to create mask
        ypix = stat[i]["ypix"]
        xpix = stat[i]["xpix"]

        mask[ypix[soma_crop], xpix[soma_crop]] = 1
        roi_masks.append(mask)

    return roi_masks

###################################################################################################
# metrics
###################################################################################################


def get_roi_center(mask):
    """Get the x, y center of mass of roi mask"""
    ypix, xpix = np.nonzero(mask)
    x_center = np.mean(xpix)
    y_center = np.mean(ypix)

    return x_center, y_center


def calc_pixel_accuracy(roi_gt, roi_pred):
    """

    Parameters
    ----------
    roi_gt : numpy array
        ground truth roi image

    roi_pred : numpy array
        predicted roi image
    """
    roi_labeled_b = roi_gt > 0
    roi_pred_b = roi_pred > 0
    pixel_accuracy = np.sum(roi_labeled_b == roi_pred_b) / np.prod(roi_labeled_b.shape)
    return pixel_accuracy


def calc_iou(roi_gt, roi_pred):
    """
    Calculate intersection over union of two roi masks
    """
    roi_labeled_b = roi_gt > 0
    roi_pred_b = roi_pred > 0

    intersection = np.sum(roi_labeled_b * roi_pred_b)
    union = np.sum(roi_labeled_b) + np.sum(roi_pred_b) - intersection
    iou = intersection / union

    return iou


def roi_bounding_box(roi_mask, pad: int = None):
    """Returns the bounding box of a roi mask"""
    roi_mask = roi_mask.astype(np.uint8)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)

    if pad:
        x = x - pad
        y = y - pad
        w = w + pad * 2
        h = h + pad * 2
    return x, y, w, h

###################################################################################################
# Plotting
###################################################################################################


def plot_contours_overlap_two_masks(mask1: np.ndarray,
                                    mask2: np.ndarray,
                                    img: np.ndarray = None,
                                    colors: list = None,
                                    ax=None) -> plt.axes:
    """Given two masks, plot the contours of the masks on an image

    Parameters
    ----------
    mask1 : np.ndarray
        mask 1
    mask2 : np.ndarray
        mask 2
    img : np.ndarray, optional
        background image
    colors : list, optional
        list of colors for each mask, by default None
    ax : plt.axes, optional
        axis to plot on

    Returns
    -------
    plt.axes
    """
    assert mask1.shape == mask2.shape, "masks must be same shape"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if colors is None:
        colors = ['red', 'green']

    # background image
    if img is not None:
        assert img.shape == mask1.shape
        vmax = np.percentile(img, 99.6)
        ax.imshow(img, vmax=vmax, cmap=plt.cm.gray)
    else:
        bg = np.ones_like(mask1)
        ax.imshow(bg, cmap=plt.cm.gray, vmin=0, vmax=1)

    # binarize masks, NOTE: adjacent rois will be merged
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    contours_labeled = measure.find_contours(mask1, 0.5)
    contours_pred = measure.find_contours(mask2, 0.5)

    for n, contour in enumerate(contours_labeled):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[0])

    for n, contour in enumerate(contours_pred):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[1])

    ax.set_facecolor('white')

    return ax


# TODO: typical plots for 
# def plot_rois_gt_vs_pred(mask1: np.ndarray,
#                         mask2: np.ndarray,
#                         img: np.ndarray = None,
#                         white_bg: bool = False,
#                         title: str = '',
#                         ax=None,
#                         save_dir: str = None,
#                         save: bool = False):
    
#     ax = plot_contours_overlap_two_masks(mask1, mask2, img, ax=ax, colors = [''])
        

    
#     ax.set_xticks([])
#     ax.set_yticks([])
#     if title:
#         ax.set_title(title)
#     else:
#         title = "red=gt_green_pred"
#         ax.set_title("red = gt, green = pred")
#     if save:
#         output_fn = save_dir + title + ".png"
#         plt.savefig(output_fn, bbox_inches='tight', dpi=300, facecolor='white', transparent=False)

###################################################################################################
# Ground truth v Prediction evalulation
###################################################################################################


def classify_single_pred_roi_by_iou(roi_mask_pred: np.ndarray,
                                    all_gt_masks: np.ndarray,
                                    tp_thresh: float = .3) -> Tuple(float, int, str):
    """Calculate iou of single mask with all other masks from a predicted set

    roi_mask_pred: np.array
        mask of single roi from predicted image
    all_gt_masks: np.array
        array of all gt masks from single plane
    tp_thresh: float
        threshold for iou to be considered a true positive

    Returns
    max_iou: float
        max iou of roi_mask_pred with all other masks
    max_roi: int
        roi id of mask with max iou
    classification: str
        classification of roi based on max_iou

    """
    roi_ids_gt = np.unique(all_gt_masks)
    roi_id_pred = np.unique(roi_mask_pred)[0]

    all_ious_dict = {}

    ious = []
    # binarize roi_mask_pred
    roi_mask_pred = roi_mask_pred > 0
    for roi in roi_ids_gt:

        roi_pred_mask = all_gt_masks == roi

        # binarize roi_pred_mask (need?)
        roi_pred_mask = roi_pred_mask > 0

        # calc intersection
        intersection = np.sum(roi_mask_pred * roi_pred_mask)
        # calc union
        union = np.sum(roi_mask_pred) + np.sum(roi_pred_mask) - intersection
        # calc iou
        iou = intersection / union

        ious.append(iou)

    all_ious_dict.update({roi_id_pred: ious})

    # get max iou
    max_iou = np.round(np.max(ious), 3)
    max_roi = roi_ids_gt[np.argmax(ious)]

    # if multiple rois have max iou over thresh, then MP
    if np.sum(np.array(ious) > tp_thresh) > 1:
        classification = "MP"  # "Multiple Positive"
    elif max_iou >= tp_thresh:
        classification = "TP"  # "True Positive"
    elif max_iou < tp_thresh:
        classification = "FP"  # "False Negative"

    return max_iou, max_roi, classification, all_ious_dict


def calc_roi_classification_stats(roi_masks_pred, roi_masks_gt, tp_thresh = .3):
    roi_class_dict = {}
    all_ious_dict = {}

    roi_ids_gt = np.unique(roi_masks_gt)
    roi_ids_pred = np.unique(roi_masks_pred)

    for roi_id_pred in roi_ids_pred:
        roi_mask = roi_masks_pred == roi_id_pred
        max_iou, gt_roi_id, classification, ious_dict = classify_single_pred_roi_by_iou(roi_mask, 
                                                                                       roi_masks_gt,
                                                                                       tp_thresh = tp_thresh)

        roi_class_dict.update({roi_id_pred: {"iou": max_iou,
                               "gt_roi_id": gt_roi_id,
                               "classification": classification}})
        print(f"iou for roi_pred: {roi_id_pred} is {max_iou} with roi_gt: {gt_roi_id}")

        all_ious_dict.update({roi_id_pred: ious_dict})

    # convert to dataframe
    roi_class_df = pd.DataFrame.from_dict(roi_class_dict, orient="table")
    roi_class_df = roi_class_df.reset_index().rename(columns={"index": "pred_roi_id"})

    # add FN to new df
    fp_roi_ids = np.setdiff1d(roi_ids_gt, roi_class_df["gt_roi_id"].unique())
    fp_df = pd.DataFrame({"pred_roi_id": fp_roi_ids,
                          "classification": "FN"})
    roi_class_df_fn = pd.concat([roi_class_df, fp_df], ignore_index=True)

    return roi_class_df_fn


