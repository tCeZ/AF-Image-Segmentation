import numpy as np
from scipy.sparse import csr_matrix
from aicsimageio import AICSImage
from pathlib import Path
import pickle
from scipy import ndimage
import bz2
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy import ndimage
from cellpose import plot, utils
import cv2
from skimage import (
    filters, measure, morphology, segmentation
)
import numba as nb
from numba.typed import Dict as nbDict
from numba.typed import List as nbList
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter


# assl = [0.01, 0.01, 0.01, 0.023, 0.02, 0.051, 0.032, 0.043, 0.054, 0.062, 0.043, 0.055, 0.043, 0.044, 0.045, 0.054, 0.042, 0.061, 0.060, 0.065, 0.053, 0.074, 0.102, 0.082, 0.092, 0.104, 0.111, 0.123, 0.148, 0.133, 0.134, 0.101, 0.09]
# praucl = [0.002, 0.002, 0.002, 0.001, 0.002, 0.008, 0.01, 0.013, 0.015, 0.021, 0.012, 0.018, 0.012, 0.014, 0.016, 0.019, 0.016, 0.022, 0.021, 0.025, 0.02, 0.026, 0.030, 0.028, 0.031, 0.033, 0.035, 0.036, 0.0399, 0.036, 0.0366, 0.032, 0.03]
# apopcl = [0.22, 0.22, 0.22, 0.25, 0.23, 0.38, 0.29, 0.33, 0.39, 0.31, 0.32, 0.29, 0.030, 0.29, 0.029, 0.31, 0.32, 0.35, 0.34, 0.39, 0.34, 0.41, 0.42, 0.40, 0.44, 0.44, 0.41, 0.41, 0.56, 0.58, 0.56, 0.56, 0.44]


def compute_M(data):
  cols = np.arange(data.size)
  return csr_matrix((cols, (data.ravel(), cols)),
            shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
  M = compute_M(data)
  return [np.unravel_index(row.data, data.shape) for row in M]

def check_match_iou(ref_arr, que_arr, j_thre):
  a = set((tuple(i) for i in ref_arr))
  b = set((tuple(i) for i in que_arr))
  match_pixel_num = len(list(a & b))
  union_pixel_num = len(list(a | b))
  j_idx = match_pixel_num / union_pixel_num
  if j_idx > j_thre:
    return True
  else:
    return False


  
def check_match_overlap(ref_arr, que_arr):
  a = set((tuple(i) for i in ref_arr))
  b = set((tuple(i) for i in que_arr))
  match_pixel_num = len(list(a & b))
  ref_pixel_num = len(list(a))
  if match_pixel_num > (0.5 * ref_pixel_num):
    return True
  else:
    return False
    
def get_f1_score(reference_mask, query_mask, jac_thre):
  reference_mask_coords = get_indices_sparse(reference_mask)[1:]
  query_mask_coords = get_indices_sparse(query_mask)[1:]
  reference_mask_coords = list(map(lambda x: np.array(x).T, reference_mask_coords))
  query_mask_coords = list(map(lambda x: np.array(x).T, query_mask_coords))

  reference_mask_matched_index_list = []
  query_mask_matched_index_list = []

  TP = 0
  for i in range(len(reference_mask_coords)):
    # print(i)
    if len(reference_mask_coords[i]) != 0:
      current_reference_mask_coords = reference_mask_coords[i]
      query_mask_search_num = np.unique(list(map(lambda x: query_mask[tuple(x)], current_reference_mask_coords)))
      # print(query_mask_search_num)
      for j in query_mask_search_num:
        current_query_mask_coords = query_mask_coords[j-1]
        if j != 0:
          if (j-1 not in query_mask_matched_index_list) and (i not in reference_mask_matched_index_list):
            matched_bool = check_match_iou(current_reference_mask_coords, current_query_mask_coords, jac_thre)
            if matched_bool == True:
              TP += 1
              i_ind = i
              j_ind = j - 1
              reference_mask_matched_index_list.append(i_ind)
              query_mask_matched_index_list.append(j_ind)
              break

  reference_mask_cell_num = len(reference_mask_coords)
  query_mask_cell_num = len(query_mask_coords)
  FN_FP = reference_mask_cell_num + query_mask_cell_num - TP * 2
  FN = reference_mask_cell_num - TP
  FP = query_mask_cell_num - TP
  f1_score = TP / (TP + 0.5 * FN_FP)

  return f1_score, TP, FP, FN

def calculate_jaccard(ref_arr, que_arr):
  a = set((tuple(i) for i in ref_arr))
  b = set((tuple(i) for i in que_arr))
  match_pixel_num = len(list(a & b))
  union_pixel_num = len(list(a | b))
  j_idx = match_pixel_num / union_pixel_num
  return j_idx


def get_seg_score(reference_mask, query_mask):
  reference_mask_coords = get_indices_sparse(reference_mask)[1:]
  query_mask_coords = get_indices_sparse(query_mask)[1:]
  reference_mask_coords = list(map(lambda x: np.array(x).T, reference_mask_coords))
  query_mask_coords = list(map(lambda x: np.array(x).T, query_mask_coords))
  
  query_mask_matched_index_list = []
  
  seg_score_list = []
  for i in range(len(reference_mask_coords)):
    if len(reference_mask_coords[i]) != 0:
      current_reference_mask_coords = reference_mask_coords[i]
      query_mask_search_num = np.unique(list(map(lambda x: query_mask[tuple(x)], current_reference_mask_coords)))
      best_jaccard = 0
      for j in query_mask_search_num:
        current_query_mask_coords = query_mask_coords[j-1]
        if j != 0:
          if (j-1 not in query_mask_matched_index_list):
            match_bool = check_match_overlap(current_reference_mask_coords, current_query_mask_coords)
            if match_bool == True:
              current_jaccard = calculate_jaccard(current_reference_mask_coords, current_query_mask_coords)
              if current_jaccard > best_jaccard:
                best_jaccard = current_jaccard
                # i_ind_best = i
                j_ind_best = j-1
        
      if best_jaccard > 0:
        query_mask_matched_index_list.append(j_ind_best)
      seg_score_list.append(best_jaccard)

  
  return np.average(seg_score_list)

def prauc_calc(c):
  # [f1_score, TP, FP, FN] - input

  # i = np.asarray(c)

  current_precision = c[:, 1] / (c[:, 1] + c[:, 2])
  current_recall = c[:, 1] / (c[:, 1] + c[:, 3])

  pr_matrix = np.stack((current_precision, current_recall), axis=0).T
  pr_matrix_sorted = pr_matrix[(-pr_matrix[:, 0]).argsort()]
  current_PRAUC = auc(pr_matrix_sorted[:, 0], pr_matrix_sorted[:, 1])

  return current_PRAUC

@nb.njit()
def nb_populate_dict(cell_num, cell_num_idx):
    d = nbDict.empty(nb.types.int64, nb.types.int64)

    for i in range(0, len(cell_num)):
        d[cell_num[i]] = cell_num_idx[i]

    return d

def check_sequential(mask_data):
  # find cell index - if not sequential
  cell_num = np.unique(mask_data)
  maxvalue = len(cell_num)
  # mask.set_cell_index(cell_num[1:])
  fmask_data = mask_data

  if maxvalue - 1 != np.max(mask_data):
    cell_num_idx = np.arange(0, len(cell_num))
    # cell_num_dict = dict(zip(cell_num, cell_num_idx))
    cell_num_dict = nb_populate_dict(cell_num, cell_num_idx)
    fmask_data = mask_data.reshape(-1)

    # for i in range(0, len(fmask_data)):
    #     fmask_data[i] = cell_num_dict.get(fmask_data[i])
    cell_num_index_map(fmask_data, cell_num_dict)

    fmask_data = fmask_data.reshape((mask_data.shape[0], mask_data.shape[1]))
    # mask.set_data(fmask_data)
    # mask_data = mask.get_data()

    cell_num = np.unique(mask_data)
    maxvalue = len(cell_num)

  assert (maxvalue - 1) == np.max(mask_data)

  return fmask_data

@nb.njit(parallel=True)
def cell_num_index_map(flat_mask, cell_num_dict):
  for i in nb.prange(0, len(flat_mask)):
    flat_mask[i] = cell_num_dict.get(flat_mask[i])

def divide_into_tiles(arr, gcd):
  # Convert the input array to a NumPy array if it is not already one
  arr = np.array(arr)

  # Find the number of rows and columns in the array
  num_rows, num_cols = arr.shape

  # Find the greatest common divisor of the number of rows and columns
  # gcd = np.gcd(num_rows, num_cols)
  #
  # if gcd == 1:
  #   gcd = gcd_m

  # Divide the number of rows and columns by the greatest common divisor to get the number of tiles in each direction
  num_tiles_rows = num_rows // gcd
  num_tiles_cols = num_cols // gcd

  # Initialize a list to hold the tiles
  tiles = []

  # Iterate over the array and divide it into tiles
  for i in range(num_tiles_rows):
    row_start = i * gcd
    row_end = row_start + gcd
    for j in range(num_tiles_cols):
      col_start = j * gcd
      col_end = col_start + gcd
      tile = arr[row_start:row_end, col_start:col_end]
      tiles.append(tile)

  return tiles


if __name__ == '__main__':



  #TRAIN #
  mask_train = AICSImage('/Users/tedzhang/Desktop/CMU/murphylab/cellpose/stitched_images/bfconvert_mask.ome.tiff')
  # img_crop_1 = AICSImage('/Users/tedzhang/Desktop/CMU/murphylab/cellpose/stitched_images/Cyc001_Reg001_Ch002/fused_tp_0_ch_0.tif')
  # img_crop_2 = AICSImage('/Users/tedzhang/Desktop/CMU/murphylab/cellpose/stitched_images/Cyc001_Reg001_Ch003/fused_tp_0_ch_0.tif')
  img_crop_3 = AICSImage('/Users/tedzhang/Desktop/CMU/murphylab/cellpose/stitched_images/Cyc001_Reg001_Ch004/fused_tp_0_ch_0.tif')

  #CROP
  img_crop = img_crop_3.data[0, 0, 0, :, :]
  mask = mask_train.data[0, 0, 0, :, :]
  crop = divide_into_tiles(img_crop, 1000)
  mcrop = divide_into_tiles(mask, 1000)

  #save the cropped images
  output_dir = Path('/Users/tedzhang/Desktop/CMU/murphylab/cellpose/build/train_3')
  # out_img_dir = output_dir / 'img_crop'
  # out_mask_dir = output_dir / 'mask_crop'

  writer = OmeTiffWriter()
  for i in range(len(crop)):
    writer.save(crop[i], output_dir / ('img_' + str(i) + '.tif'))
    writer.save(mcrop[i], output_dir / ('img_' + str(i) + '_masks.tif'))

  print('end of crop')



  #TEST#
  mask = AICSImage('/Users/tedzhang/Desktop/cellpose/mask_c.ome.tiff')
  mask_2 = np.load('/Users/tedzhang/Desktop/cellpose/reg001_X01_Y01_t001_z001_c001_seg.npy', allow_pickle=True).item()

  # openfile = bz2.BZ2File('/Users/tedzhang/Desktop/cellpose/mask_deepcell_membrane-0.12.3.pickle', 'rb')
  # test_mask = pickle.load(openfile)
  #
  # openfile_2 = bz2.BZ2File('/Users/tedzhang/Desktop/cellpose/mask_expert1.pickle', "rb")
  # test_mask_2 = pickle.load(openfile_2)
  maskref = mask.data[0, 0, 0, :, :]
  maskpred = mask_2['masks']

  #quality control
  maskref = check_sequential(maskref)
  maskpred = check_sequential(maskpred)

  #erode
  f1 = []
  praucl = []
  for j in range(1, 8, 2):
    e = ndimage.binary_erosion(maskref, iterations=j)
    e = e * maskref

    c = []
    for i in np.arange(0, 1, 0.05):
      c.append(get_f1_score(maskref, e, i))

    c = np.asarray(c)
    prauc = prauc_calc(c)

    f1.append(np.average(c[:, 0]))
    praucl.append(prauc)

###################
  a = get_seg_score(maskref, maskpred) #SEG
  b = get_seg_score(maskpred, maskref)
  avg_seg_score = (a+b) / 2 #SEG-PRIME
###################
  # jac = calculate_jaccard(test_mask, test_mask_2)

  c = []
  for i in np.arange(0, 1, 0.05):
    c.append(get_f1_score(maskref, maskpred, i))

  prauc = prauc_calc(c)

  #PLOT


  print('end')

