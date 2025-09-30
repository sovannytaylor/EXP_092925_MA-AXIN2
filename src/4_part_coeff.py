"""Detect puncta, measure features, visualize data
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage.color import label2rgb
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest, skew
from skimage import morphology
from skimage.measure import regionprops
from skimage.morphology import dilation, disk
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'
proof_folder = 'python_results/proofs/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

if not os.path.exists(proof_folder):
    os.mkdir(proof_folder)

def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))


def remove_large_objects(label_image, max_size):
    out = np.zeros_like(label_image)
    for region in regionprops(label_image):
        if region.area <= max_size:
            out[label_image == region.label] = region.label
    return out


# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# Assumes images[key] shape is (C, H, W) and masks[key] shape is (1, H, W) or (H, W)
image_mask_dict = {
    key: np.stack([
        images[key][0],  # Channel 0 - 647-TNK-FLAG
        images[key][1],  # Channel 1 - 594 - TNK -antibody
        images[key][2],  # Channel 2 - AXIN2 GFP condensate
        images[key][3],  # Channel 3 - Dapi
        masks[key][0] if masks[key].ndim == 3 else masks[key]  # Handle 2D or 3D mask
    ])
    for key in masks
}
# ----------------collect feature information----------------


# ---------------- filtering steps ----------------


#saturation filtering of channel 0 and channel 1 (TNK flag and antibody)
not_saturated = {}
# structure element for eroding masks
structure_element = np.ones((16, 16)).astype(int)

for name, image in image_mask_dict.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # loop to remove saturated masks (>5% px values = 65535 in ch0 or ch1)
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[-1, :, :] == label)

        # define mask for this label
        TNK_0 = np.where(image[-1, :, :] == label, label, 0)
        # erode mask to avoid bright puncta at periphery
        mask_eroded = morphology.erosion(TNK_0, structure_element)

        # extract intensities for ch0 and ch1 inside mask
        TINK_0 = np.where(mask_eroded == label, image[0, :, :], 0)
        TINK_1 = np.where(mask_eroded == label, image[1, :, :], 0)

        # check saturation in ch0
        ch0_saturated_count = np.count_nonzero(TINK_0 == 65535)
        # check saturation in ch1
        ch1_saturated_count = np.count_nonzero(TINK_1 == 65535)

        # keep only if BOTH channels are under saturation threshold
        if ((ch0_saturated_count / pixel_count) < 0.05 and
            (ch1_saturated_count / pixel_count) < 0.05):
            labels_filtered.append(mask_eroded)

    # add all eroded, non-saturated masks together
    masks_filtered = np.sum(labels_filtered, axis=0)

    # stack the filtered masks with original image channels
    # stack the filtered mask with all original image channels
    masks_filtered_stack = np.stack(
     (image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :], masks_filtered)
    )
    not_saturated[name] = masks_filtered_stack


# # now filter out cells that are not FLAG positive 
# cell_intensities = {}

# for name, image in not_saturated.items():
#     mask = image[-1, :, :]        # filtered mask (labels per cell)
#     ch0 = image[0, :, :]          # channel 0 = 647-TNK-FLAG

#     intensities = []
#     for label in np.unique(mask):
#         if label == 0:  # skip background
#             continue
#         # average intensity of this label in channel 0
#         mean_intensity = ch0[mask == label].mean()
#         intensities.append(mean_intensity)
    
#     cell_intensities[name] = intensities


# #now look at the global distribution
# all_intensities = [i for vals in cell_intensities.values() for i in vals]
# all_intensities = np.array(all_intensities)

# print("Min:", all_intensities.min())
# print("Max:", all_intensities.max())
# print("Mean:", all_intensities.mean())
# print("Median:", np.median(all_intensities))

# # plot channel 0 so we can get a good grasp on what the range for the flag intensity is 

# plt.hist(all_intensities, bins=100)
# plt.xlabel("Mean Ch0 intensity per cell (647-TNK-FLAG)")
# plt.ylabel("Cell count")
# plt.title("Distribution of FLAG intensities")
# plt.show()

# ^ we plotted this but ended up using imagej to look at good threshold numbers 

# ---- threshold FLAG positve and GFP positive cells ----
# Threshold values
ch0_thresh = 2000  # TNK-FLAG
ch2_thresh = 500  # AXIN2 GFP / condensate

filtered_cells = {}

for name, image in not_saturated.items():
    mask = image[-1, :, :]  # filtered mask from saturation step
    ch0 = image[0, :, :]
    ch2 = image[2, :, :]

    labels_filtered = []

    for label in np.unique(mask)[1:]:  # skip background
        cell_mask = (mask == label)

        mean_ch0 = np.mean(ch0[cell_mask])
        mean_ch2 = np.mean(ch2[cell_mask])

        if mean_ch0 > ch0_thresh and mean_ch2 > ch2_thresh:
            # keep this cell
            labels_filtered.append(cell_mask.astype(int) * label)

    # combine kept cells into a new mask
    if labels_filtered:
        new_mask = np.sum(labels_filtered, axis=0)
    else:
        new_mask = np.zeros_like(mask)

    # stack with original image channels
    filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :], new_mask)
    )

    filtered_cells[name] = filtered_stack

# Now filtered_cells contains only cells above the mean intensity thresholds

#checking to make sure the filtering step worked 

# --- Count cells BEFORE intensity filtering ---
total_before = 0
print("Cells per image BEFORE intensity filtering:")
for name, image in not_saturated.items():  # use the mask after saturation filtering
    mask = image[-1, :, :]
    num_cells = len(np.unique(mask)) - 1  # subtract 1 for background
    total_before += num_cells
    print(f"{name}: {num_cells} cells")
print(f"Total cells BEFORE intensity filtering: {total_before}\n")

# --- Count cells AFTER intensity filtering ---
total_after = 0
print("Cells per image AFTER intensity filtering:")
for name, image in filtered_cells.items():  # use mask after mean intensity threshold
    mask = image[-1, :, :]
    num_cells = len(np.unique(mask)) - 1  # subtract 1 for background
    total_after += num_cells
    print(f"{name}: {num_cells} cells")
print(f"Total cells AFTER intensity filtering: {total_after}")


# now collect condensate masks and features info
logger.info('collecting feature info')
feature_information_list = []
for name, image in filtered_cells.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[-1, :, :] != 0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]

    # loop to extract params from cells
    for num in unique_val[1:]:
        # last channel (-1) is always the mask
        cell_mask = np.where(image[-1, :, :] == num, image[-1, :, :], 0)

        # channel 0 = TNK-FLAG intensity
        tnk_flag = np.where(image[-1, :, :] == num, image[0, :, :], 0)
        tnk_flag_mean = np.mean(tnk_flag[tnk_flag != 0])

        # channel 1 = TNK-antibody intensity
        tnk_ab = np.where(image[-1, :, :] == num, image[1, :, :], 0)
        tnk_ab_mean = np.mean(tnk_ab[tnk_ab != 0])

        # channel 2 = making the mask for condensates
        condenschan = np.where(image[-1, :, :] == num, image[2, :, :], 0)
        condenschan_std = np.std(condenschan[condenschan != 0])
        condenschan_mean = np.mean(condenschan[condenschan != 0])

        # thresholding to define condensates
        # i decided to not do global cut off per cell but basing it off of the local mean intensity
        threshold = condenschan_mean + (2 * condenschan_std)
        binary = (condenschan > threshold).astype(int)
        binary = dilation(binary, disk(1))  # expand puncta edges
        condens_masks = measure.label(binary)
        condens_masks = remove_small_objects(condens_masks, 9)
        # remove large objects (>500 area)
        condens_masks = remove_large_objects(condens_masks, 500)

        # === PROOF PLOTTING ===
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # left: raw condensate channel, inverted (white background, dark puncta)
        axes[0].imshow(condenschan, cmap='gray_r')  # note the "_r" = reversed colormap
        axes[0].set_title("Channel 2 (Condensate)")
        axes[0].axis('off')

        # right: masks overlayed on inverted grayscale
        overlay = label2rgb(condens_masks, image=condenschan, bg_label=0, alpha=0.3, bg_color=(1,1,1))
        axes[1].imshow(overlay, cmap='gray_r')
        axes[1].set_title("Condensate Masks")
        axes[1].axis('off')

        plt.suptitle(f"Proof: {name}, Cell {num}", fontsize=12)
        plt.tight_layout()

        # Save instead of show
        plt.savefig(os.path.join(proof_folder, f"{name}_cell{num}.png"), dpi=200)
        plt.close()
        # =======================

        # measure properties of condensate masks
        condens_properties = feature_extractor(condens_masks).add_prefix('condens_')

        # make list for cov and skew, add as columns to properties for TNK-FLAG and TNK-ab channels
        tnk_flag_condens_cv_list = []
        tnk_flag_condens_intensity_list = []

        tnk_ab_condens_cv_list = []
        tnk_ab_condens_intensity_list = []

        for condens_num in np.unique(condens_masks)[1:]:
            # per-condensate TNK-FLAG measurements
            tnk_flag_condens = np.where(condens_masks == condens_num, image[0, :, :], 0)
            tnk_flag_condens = tnk_flag_condens[tnk_flag_condens != 0]
            tnk_flag_condens_cv = np.std(tnk_flag_condens) / np.mean(tnk_flag_condens)
            tnk_flag_condens_cv_list.append(tnk_flag_condens_cv)
            tnk_flag_condens_intensity_list.append(np.mean(tnk_flag_condens))

            # per-condensate TNK-antibody measurements
            tnk_ab_condens = np.where(condens_masks == condens_num, image[1, :, :], 0)
            tnk_ab_condens = tnk_ab_condens[tnk_ab_condens != 0]
            tnk_ab_condens_cv = np.std(tnk_ab_condens) / np.mean(tnk_ab_condens)
            tnk_ab_condens_cv_list.append(tnk_ab_condens_cv)
            tnk_ab_condens_intensity_list.append(np.mean(tnk_ab_condens))

        # store measurements
        condens_properties['tnk_flag_condens_cv'] = tnk_flag_condens_cv_list
        condens_properties['tnk_flag_condens_intensity'] = tnk_flag_condens_intensity_list

        condens_properties['tnk_ab_condens_cv'] = tnk_ab_condens_cv_list
        condens_properties['tnk_ab_condens_intensity'] = tnk_ab_condens_intensity_list

        # if no condensates, fill with 0
        if len(condens_properties) < 1:
            condens_properties.loc[len(condens_properties)] = 0

        # make df and add cell and image info
        properties = pd.concat([condens_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell_mask[cell_mask != 0])
        properties['cell_tng_flag_mean'] = tnk_flag_mean
        properties['cell_tnk_ab_mean'] = tnk_ab_mean

        # add cell outlines to coords
        properties['cell_coords'] = [contour] * len(properties)

        feature_information_list.append(properties)

feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')


# adding columns based on image_name so adding variant
feature_information['variant'] = feature_information['image_name'].str.split('_').str[2]


# add aspect ratio (like asking 12x5 or 12/5) and circularity
feature_information['condens_aspect_ratio'] = feature_information['condens_minor_axis_length'] / feature_information['condens_major_axis_length']
feature_information['condens_circularity'] = (12.566*feature_information['condens_area'])/(feature_information['condens_perimeter']**2)

# add partitioning coefficient
feature_information['tnk_flag_part_coeff'] = feature_information['tnk_flag_condens_intensity'] / feature_information['cell_tng_flag_mean']

# add partitioning coefficient
feature_information['tnk_ab_part_coeff'] = feature_information['tnk_ab_condens_intensity'] / feature_information['cell_tnk_ab_mean']


# save data for plotting coords
feature_information.to_csv(f'{output_folder}AXIN2_GFP-pos_TNKflag-pos_puncta-detection_feature_info.csv')

# make additional df for avgs per replicate
features_of_interest = ['condens_area', 'condens_eccentricity',
'condens_major_axis_length', 'condens_minor_axis_length',
'tnk_flag_condens_cv', 'tnk_flag_condens_intensity',
'tnk_ab_condens_cv', 'tnk_ab_condens_intensity','cell_size', 'cell_tng_flag_mean','cell_tnk_ab_mean','tnk_ab_part_coeff', 'tnk_flag_part_coeff', 'condens_aspect_ratio','condens_circularity']

condens_summary_percell = []
for col in features_of_interest:
    reps_table = feature_information.groupby(['variant','cell_number']).mean(numeric_only=True)[f'{col}']
    condens_summary_percell.append(reps_table)
condens_summary_percell_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['variant','cell_number'], how='outer'), condens_summary_percell).reset_index()

condens_summary_percell_df.to_csv(f'{output_folder}AXIN2_GFP-pos_TNKflag-pos_puncta-detection_feature_info_percell.csv')

# --------------visualize calculated parameters - raw --------------

x = 'variant'
order = ['wt','E66K', 'G67R', 'P50S', 'R68Q', 'R77Q', 'V40G']

plots_per_fig = 6
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

for fig_num in range(num_figures):
    # Create a new figure
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Calculated Parameters - per condensate (Fig {fig_num + 1})', fontsize=18, y=0.99)

    # Get the current slice of features
    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)
        sns.stripplot(data=feature_information, x=x, y=parameter, dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
        sns.boxplot(data=feature_information, x=x, y=parameter, palette=['.9'], order=order, ax=ax)
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    output_path = f'{output_folder}/puncta-features_percondensate_raw_fig{fig_num + 1}.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


#---Plotting specific plots --------------

# Count number of cells per variant
cell_counts = condens_summary_percell_df['variant'].value_counts().to_dict()


# Add a new column with the labels for plotting
condens_summary_percell_df['variant_labeled'] = condens_summary_percell_df['variant'].map(
    lambda v: f"{v} (n={cell_counts.get(v, 0)})"
)

# Then update plotting variable
x = 'variant_labeled'
order = ['wt','E66K', 'G67R', 'P50S', 'R68Q', 'R77Q', 'V40G']

# ✅ Only include the features you want to plot
features_of_interest = ['tnk_ab_part_coeff', 'tnk_flag_part_coeff']

plots_per_fig = 2
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

# Find global min and max across all features
ymin = condens_summary_percell_df[features_of_interest].min().min()
ymax = condens_summary_percell_df[features_of_interest].max().max()

for fig_num in range(num_figures):
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Partitioning Coeff - per cell', fontsize=18, y=0.99)

    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)

        sns.stripplot(
            data=condens_summary_percell_df, 
            x=x, y=parameter,
            dodge=True, edgecolor='white', linewidth=1,
            size=8, alpha=0.4, order=order, ax=ax
        )
        sns.boxplot(
            data=condens_summary_percell_df, 
            x=x, y=parameter,
            palette=['.9'], order=order, ax=ax
        )

        ax.set_ylim(ymin, ymax)  # 🔑 ensure same y-axis across plots
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    output_path = f'{output_folder}/partition_coeffs_percell.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()



# ## Below will measure and plot per cell instead of per nucleolus
# # --------------Grab major and minor_axis_length for punctas--------------
# minor_axis = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_minor_axis_length'].mean()
# major_axis = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_major_axis_length'].mean()

# # --------------Calculate average size of punctas per nuc--------------
# puncta_avg_area = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].mean().reset_index()

# # --------------Calculate proportion of area in punctas--------------
# nuc_size = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nuc_size'].mean()
# puncta_area = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].sum()
# puncta_proportion = ((puncta_area / nuc_size) *
#                    100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# # --------------Calculate number of 'punctas' per nuc--------------
# puncta_count = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].count()

# # --------------Calculate average size of punctas per nuc--------------
# avg_eccentricity = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_eccentricity'].mean().reset_index()

# # --------------Grab nuc nucleol cov --------------
# nucleol_cv_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_cv'].mean()

# # --------------Grab nuc nucleol skew --------------
# nucleol_skew_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_skew'].mean()

# # --------------Grab nuc nucleol partition coeff --------------
# partition_coeff = feature_information.groupby(
#     ['image_name', 'nuc_number'])['partition_coeff'].mean()

# # --------------Grab nuc intensity mean --------------
# nuc_intensity_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nuc_intensity_mean'].mean()

# # --------------Summarise, save to csv--------------
# summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'nuc_number'], how='outer'), [nuc_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, nucleol_cv_mean, nucleol_skew_mean, partition_coeff, nuc_intensity_mean])
# summary.columns = ['image_name', 'nuc_number',  'nuc_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'nucleol_cv_mean', 'nucleol_skew_mean', 'partition_coeff', 'nuc_intensity_mean']

# # --------------tidy up dataframe--------------
# # add columns for sorting
# # add peptide name
# summary['peptide'] = summary['image_name'].str.split('_').str[1].str.split('-').str[-1]

# # save
# summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

# # make df where all puncta features are normalized to mean nuc intensity
# normalized_summary = summary.copy()
# for column in normalized_summary.columns[3:-3]:
#     column
#     normalized_summary[column] = normalized_summary[column] / normalized_summary['nuc_intensity_mean']

# # --------------visualize calculated parameters - raw --------------
# features_of_interest = ['mean_puncta_area',
#        'puncta_area_proportion', 'puncta_count', 'avg_eccentricity', 'nucleol_cv_mean', 'nucleol_skew_mean', 'partition_coeff', 'nuc_intensity_mean']
# plt.figure(figsize=(20, 15))
# plt.subplots_adjust(hspace=0.5)
# plt.suptitle('calculated parameters - per nuc', fontsize=18, y=0.99)
# # loop through the length of tickers and keep track of index
# for n, parameter in enumerate(features_of_interest):
#     # add a new subplot iteratively
#     ax = plt.subplot(3, 4, n + 1)

#     sns.stripplot(data=summary, x=x, y=parameter, dodge='True',
#                     edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
#     sns.boxplot(data=summary, x=x, y=parameter,
#                 palette=['.9'], order=order, ax=ax)
    
#     # # statannot stats
#     # annotator = Annotator(ax, pairs, data=summary, x=x, y=parameter, order=order)
#     # annotator.configure(test='t-test_ind', verbose=2)
#     # annotator.apply_test()
#     # annotator.annotate()

#     # formatting
#     sns.despine()
#     ax.set_xlabel('')

# plt.tight_layout()
# plt.savefig(f'{output_folder}puncta-features_pernuc_raw.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# # --------------visualize calculated parameters - normalized --------------
# plt.figure(figsize=(15, 15))
# plt.subplots_adjust(hspace=0.5)
# plt.suptitle('calculated parameters - per nuc, normalized to cytoplasm intensity', fontsize=18, y=0.99)
# # loop through the length of tickers and keep track of index
# for n, parameter in enumerate(summary.columns.tolist()[3:-3]):
#     # add a new subplot iteratively
#     ax = plt.subplot(3, 4, n + 1)

#     # filter df and plot ticker on the new subplot axis
#     sns.stripplot(data=normalized_summary, x=x, y=parameter, dodge='True',
#                     edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
#     sns.boxplot(data=normalized_summary, x=x, y=parameter,
#                 palette=['.9'], order=order, ax=ax)
    
#     # statannot stats
#     annotator = Annotator(ax, pairs, data=normalized_summary, x=x, y=parameter, order=order)
#     annotator.configure(test='t-test_ind', verbose=2)
#     annotator.apply_test()
#     annotator.annotate()

#     # formatting
#     sns.despine()
#     ax.set_xlabel('')

# plt.tight_layout()
# plt.savefig(f'{output_folder}puncta-features_pernuc_normalized.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# -------------- plotting proofs --------------
# plot proofs
for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # extract coords
    nuc = np.where(image[2, :, :] != 0, image[0, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        nuc_contour = image_df['nuc_coords'].iloc[0]
        coord_list = np.array(image_df.nucleol_coords)

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(image[1])
        ax2.imshow(image[0])
        ax3.imshow(nuc)
        for nuc_line in nuc_contour:
            ax3.plot(nuc_line[:, 1], nuc_line[:, 0], linewidth=0.5, c='w')
        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax3.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
        for ax in fig.get_axes():
            ax.label_outer()

        # # create scale bar TODO need to update scale value
        # scalebar = ScaleBar(0.0779907, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='w', length_fraction=0.3)
        # ax3.add_artist(scalebar)

        # title and save
        fig.suptitle(name, y=0.67, size=14)
        fig.tight_layout()

        fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()