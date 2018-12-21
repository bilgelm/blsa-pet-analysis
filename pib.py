# # PiB-PET PROCESSING IN PET SPACE USING ANATOMICAL LABELS
#
# The processing steps can be summarized as follows:
# 1. Time frame alignment
# 2. MRI-PET coregistration (MRIs have already been processed and anatomical regions have been defined)
# 3. Partial volume correction of time frame data
# 4. Extraction of early amyloid (EA), SUVR, kinetic parameter images
# 5. ROI summary calculation
# 6. Spatial normalization of all output images to MNI space
#
# Steps 4-6 will be performed with and without partial volume correction.

# running on nilab10 using petpipeline_05152018 conda environment
# before running this code, make sure that you've activated this conda env:
#   source activate petpipeline_05152018

# Import packages
import os, sys, logging
import pandas as pd
import numpy as np
import scipy as sp
from collections import OrderedDict

# nipype
import nipype.interfaces.io as nio
from nipype.interfaces import spm, fsl, petpvc, c3, ants
from nipype.pipeline.engine import Workflow, Node, JoinNode, MapNode
from nipype.interfaces.utility import Function, IdentityInterface, Merge
from nipype import config, logging
config.enable_debug_mode()
logging.update_logging(config)

# custom nipype wrappers and functions
from temporalimage.nipype_wrapper import SplitTimeSeries, DynamicMean
from kineticmodel.nipype_wrapper import KineticModel
from nipype_misc_funcs import Pad4DImage, Unpad4DImage, CombineROIs, \
                              get_base_filename, to_div_string, ROI_stats_to_spreadsheet
from nipype_snapshot_funcs import realign_snapshots, coreg_snapshots, \
                                  labels_snapshots, refReg_snapshots, \
                                  triplanar_snapshots, mosaic
from nipype.utils.filemanip import split_filename

# this needs to be more flexible
from MUSE_label_scheme import whole_brain_ROI_grouping, reference_ROI_grouping, \
                              pvc_ROI_groupings, ROIs
singleROIs = OrderedDict({k: v for k, v in ROIs.items() if type(v) is int})
compositeROIs = OrderedDict({k: v for k, v in ROIs.items() if type(v) is list})

import argparse

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("pet4D", help="path to 4D PiB-PET image")
parser.add_argument("pettiming", help="path to csv file describing PET timing info")
parser.add_argument("mri", help="path to preprocessed MRI image")
parser.add_argument("label", help="path to anatomical label image")
parser.add_argument("mnitransform", help="path to composite (deformable) transform that takes MRI to MNI space")
parser.add_argument("outputdir", help="output directory")

# optional arguments
parser.add_argument("--t_start", type=float, default=0.75,
                    help="frames prior to this time (in min) will be excluded from analysis")
parser.add_argument("--t_end_realign", type=float, default=2,
                    help="time frame alignment will use the average of frames prior to this time (in min) as the target")
parser.add_argument("--t_end_coreg", type=float, default=20,
                    help="MRI coregistration will use the average of the frames prior to this time (in min) as the source")
parser.add_argument("--t_end_EA", type=float, default=2.5,
                    help="early amyloid image will be computed as the average of frames prior to this time (in min)")
parser.add_argument("--t_start_SUVR", type=float, default=50,
                    help="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)")
parser.add_argument("--t_end_SUVR", type=float, default=70,
                    help="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)")
parser.add_argument("--t_end_kinetic_model", type=float, default=70,
                    help="Kinetic parameter image computation will be based on frames prior to this time (in min)")

parser.add_argument("--no_pvc", action="store_true",
                    help="do not perform partial volume correction")

parser.add_argument("-x", "--psf_fwhm_x", type=float,
                    help="PET scanner PSF FWHM along x (in mm)")
parser.add_argument("-y", "--psf_fwhm_y", type=float,
                    help="PET scanner PSF FWHM along y (in mm)")
parser.add_argument("-z", "--psf_fwhm_z", type=float,
                    help="PET scanner PSF FWHM along z (in mm)")

parser.add_argument("-s","--smooth_fwhm", type=float,
                    help="FWHM of Gaussian smoothing filter (in mm)")

parser.add_argument("-n","--n_procs", type=int, default=12,
                    help="number of parallel processes")

args = parser.parse_args()

# number of parallel processes
n_procs = args.n_procs

# Set up standalone SPM
spm.SPMCommand.set_mlab_paths(matlab_cmd=os.environ['SPMMCRCMD'], use_mcr=True)

template = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
template_brainmask = fsl.Info.standard_image('MNI152_T1_1mm_brain_mask_dil.nii.gz')


template_dir = '/templates/UPENN_BLSA_templates'
# study-specific template in MNI space (note that this is not used as a target image in any registration - used only as reference)
blsa_template_mni = os.path.join(template_dir,'BLSA_2_MNI',
                   'BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152_reoriented.nii.gz')
blsa_template_mni_brainmask = os.path.join(template_dir,'BLSA_2_MNI',
                   'BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152_reoriented_brainmask.nii.gz')

# output directory - create if doesn't exist
output_dir = args.outputdir
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

C11_halflife = 20.334 # mins

dyn_mean_wts = None

# ## 1. INPUTS

# We set up the nipype Nodes that will act as the inputs to our Workflows.
# Nodes allow for the passing of the 4D PiB PET, processed MRI,
# and label images given the scan IDs
# (`getpet4D`, `getmri`, `getlabel`, respectively),
# as well as the retrieval of the text files detailing the PET frame timing
# information (`getpettiming`) to downstream Nodes.

# get full path to PiB scan corresponding to idvi from spreadsheet
getpet4D = Node(interface=IdentityInterface(fields=['pet4D']), name="getpet4D")
getpet4D.inputs.pet4D = args.pet4D

getpettiming = Node(interface=IdentityInterface(fields=['pettiming']), name="getpettiming")
getpettiming.inputs.pettiming = args.pettiming

getmri = Node(interface=IdentityInterface(fields=['mri']), name="getmri")
getmri.inputs.mri = args.mri

getlabel = Node(interface=IdentityInterface(fields=['label']), name="getlabel")
getlabel.inputs.label = args.label


# ## 2. REALIGN

# The goal of the realign workflow is to compute a spatially-aligned dynamic
# PiB-PET image by removing subject motion.
#
# * `reorient`: We start by reorienting the PET image to match the orientation
#   of MNI152 templates. This is not spatial normalization - no registration is
#   performed. We simply apply 90, 180, or 270 degree rotations as needed about
#   the $x,y,z$ axes to match the MNI152 orientation.
#
# We will align each time frame to the average of the first 2 minutes of acquisition. We use the first 2 minutes because they do not reflect amyloid binding but instead reflect blood flow, which has better anatomical definition that aids in the registration. We assume that the time frames in the first 2 minutes are in alignment, so we will not apply any spatial transformations to these frames.
#
# * `split_time_2mins`: Split the dynamic scan into $< 2$ min and $\geq 2$ min.
# * `dynamic_mean_2min`: Compute the average of the time frames in the $< 2$ mins.
#
# We are going to use SPM's Realign function to perform time frame alignment, using the "align to first time frame" option. Thus, we create an artificial image where the first frame is the 2-min average, and the following frames are the dynamic scan frames $\geq 2$ min.
#
# * `merge_lists`: Concatenates the full paths to the 2-min average image and time frames $\geq 2$ min, separated by a space.
# * `merge_time`: Takes the concatenated string and performs the image concatenation.
# * `pad`: In order to minimize interpolation artifacts at image edges, we pad each time frame on each of the 6 sides with the nearest slice prior to time frame alignment.
# * `realign`: This is the SPM function to do the actual work.
# * `unpad`: We reverse the padding to return to original image dimensions.
# * `drop_first_timeframe`: We had artifically put this first time frame (which is the 2-min average) to get SPM's realign function to work. We remove it after realignment.
# * `nan_to_0`: In case SPM transformations introduced any nan's (due to voxels coming into view from outside of the original image), we set these to 0 to avoid problems down the line.
# * `merge_lists2` and `merge_time2`: We need to add back the first 2-minutes of time frames to the beginning of the realigned image.
#
# _Note:_ For acquisitions that do not include time frames $< 2$ minutes post-injection, we align to the mean of the entire acquisition. This is achieved directly by setting the proper SPM Realign options. SPM uses a two-pass procedure when aligning to the mean image.

# Reorient
reorient = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorient")

# Split the dynamic scan at the 2 min mark
split_time_2mins = Node(interface=SplitTimeSeries(splitTime=args.t_end_realign), name="split_time_2mins")

# Compute 0.75-2 min average that will be used for time frame alignment
dynamic_mean_2min = Node(interface=DynamicMean(startTime=args.t_start,
                                               endTime=args.t_end_realign,
                                               weights=dyn_mean_wts),
                         name="dynamic_mean_2min")

merge_lists = Node(interface=Merge(2), name="merge_lists")
merge_time = Node(interface=fsl.Merge(dimension='t', output_type='NIFTI_PAIR'), name="merge_time")
# output type is NIFTI_PAIR because that is what SPM (used for realign) likes

pad = Node(interface=Pad4DImage(padsize=1),name="pad")
unpad = Node(interface=Unpad4DImage(padsize=1),name="unpad")

# Realign time frames
realign = Node(interface=spm.Realign(), name="realign")
# === Estimate options ===
realign.inputs.quality = 1
realign.inputs.separation = 2 # used to be 4. Separation in mm b/w points sampled in reference
                              # image. Smaller more accurate but slower
realign.inputs.fwhm = 8 # used to be 7. FWHM in mm of the Gaussian smoothing kernel applied
                        # to the images before estimating realignment parameters
realign.inputs.register_to_mean = False # align to the first time frame,
                                        # which we've hacked to be the 2 min avg
realign.inputs.interp = 2 # degree of interpolation. Higher is better but slower
realign.inputs.wrap = [0, 0, 0] # no wrap around in PET
# === Reslice options ===
realign.inputs.write_which = [1, 0] # do not to reslice the first timeframe
realign.inputs.write_interp = 4
realign.inputs.write_mask = True
realign.inputs.write_wrap = [0, 0, 0]
realign.inputs.out_prefix = 'r'
realign.inputs.jobtype = 'estwrite'
realign.use_mcr = True # run using standalone SPM (without MATLAB)

# After alignment, drop first time frame, which is the 2 min average
drop_first_timeframe = Node(interface=fsl.ExtractROI(t_min=1,t_size=-1), name="drop_first_timeframe")

# nan mask
nan_mask_4D = Node(interface=fsl.ImageMaths(op_string=' -nanm', suffix='_nanmask'), name='nan_mask_4D')
nan_mask = Node(interface=fsl.ImageMaths(op_string=' -Tmax', suffix='_Tmax'), name='nan_mask')
mulneg1 = Node(interface=fsl.ImageMaths(op_string=' -mul -1', suffix='_mul'), name='mulneg1')
add1 = Node(interface=fsl.ImageMaths(op_string=' -add 1', suffix='_add'), name='add1')

# Replace nan values after realignment with 0
nan_to_0 = Node(interface=fsl.ImageMaths(op_string=' -nan', suffix='_nanto0'), name='nan_to_0')

merge_lists2 = Node(interface=Merge(2), name="merge_lists2")
merge_time2 = Node(interface=fsl.Merge(dimension='t'), name="merge_time2")

mask4D = Node(interface=fsl.ImageMaths(op_string=' -mas', suffix='_masked'), name='mask4D')

# subtraction of .5 is a "hack" fix that works specifically for this dataset with the choice of 2 mins for t_end_realign
# the idea is to get to the start time of the time frame right before t_end_realign
realign_qc = Node(interface=realign_snapshots(splitTime=args.t_end_realign-.5), name="realign_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','realign_wf')
datasink.inputs.substitutions = [('_roi',''),
                                 ('_merged',''),
                                 ('mean','avg2min'),
                                 ('_reoriented',''),
                                 ('_padded',''),
                                 ('_masked','')]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

realign_workflow = Workflow(name="realign_workflow")
realign_workflow.base_dir = os.path.join(output_dir,'realign_workingdir')
realign_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'realign_crashdumps')}}
realign_workflow.connect([(getpet4D, reorient, [('pet4D','in_file')]),
                          # get time frames beyond first two minutes (we assume first 2 mins are free of motion)
                          (reorient, split_time_2mins, [('out_file','timeSeriesImgFile')]),
                          (getpettiming, split_time_2mins, [('pettiming','frameTimingCsvFile')]),

                          # compute 0.75-2 min average (initial frames are excluded due to insufficient SNR)
                          (reorient, dynamic_mean_2min, [('out_file','timeSeriesImgFile')]),
                          (getpettiming, dynamic_mean_2min, [('pettiming','frameTimingCsvFile')]),

                          # perform time alignment to the average of first 2 mins
                          (dynamic_mean_2min, merge_lists, [('meanImgFile','in1')]),
                          (split_time_2mins, merge_lists, [('secondImgFile','in2')]),
                          (merge_lists, merge_time, [('out','in_files')]),

                          (merge_time, pad, [('merged_file','timeSeriesImgFile')]),
                          (pad, realign, [('paddedImgFile','in_files')]),
                          (realign, unpad, [('realigned_files','timeSeriesImgFile')]),
                          (unpad, drop_first_timeframe, [('unpaddedImgFile','in_file')]),

                          (drop_first_timeframe, nan_mask_4D, [('roi_file','in_file')]),
                          (nan_mask_4D, nan_mask, [('out_file','in_file')]),
                          (nan_mask, mulneg1, [('out_file','in_file')]),
                          (mulneg1, add1, [('out_file','in_file')]),

                          (drop_first_timeframe, nan_to_0, [('roi_file','in_file')]),

                          # put together the first 2 minutes with the rest of the time frames,
                          #  which have been motion-corrected
                          (split_time_2mins, merge_lists2, [('firstImgFile','in1')]),
                          (nan_to_0, merge_lists2, [('out_file','in2')]),
                          (merge_lists2, merge_time2, [('out','in_files')]),

                          (merge_time2, mask4D, [('merged_file','in_file')]),
                          (add1, mask4D, [('out_file','in_file2')]),

                          # QC plots and snapshots
                          (getpettiming, realign_qc, [('pettiming','frameTimingCsvFile')]),
                          (realign, realign_qc, [('realignment_parameters','realignParamsFile')]),
                          #(merge_time2, realign_qc, [('merged_file','petrealignedfile')]),
                          (mask4D, realign_qc, [('out_file','petrealignedfile')]),

                          # save outputs
                          (dynamic_mean_2min, datasink, [('meanImgFile','avg2min')]), # 0.75-2min average (3D image) used in time frame alignment
                          #(merge_time2, datasink, [('merged_file','realigned')]), # realigned time series (4D image)
                          (mask4D, datasink, [('out_file','realigned')]), # realigned time series (4D image)
                          (realign, datasink, [('realignment_parameters','realigned.@par')]), # realignment parameters
                          (realign_qc, datasink, [('realign_param_plot','QC'), # QC plots and snapshots
                                                  ('realigned_img_snap','QC.@snap')])
                         ])

realign_workflow.write_graph('realign.dot', graph2use='colored', simple_form=True)



# ## 3. MRI-PET COREGISTRATION
#
# Our goal is to perform image processing in native PET space to produce parametric images. We have chosen this approach (rather than processing in native MRI space) for two reasons:
# 1. This approach avoids the use of PET data interpolated to match the voxel size of the MRI scans for generating parametric images. Such an interpolation is undesirable because of the large difference between PET and MRI voxel sizes.
# 2. Fewer brain voxels in native PET space allows for faster computation of voxelwise kinetic parameters for the whole brain.
#
# It should be noted that PET space processing is not without disadvantages. Anatomical labels have to be interpolated to match the PET voxel size, which yields inaccuracies. While these inaccuracies are not important for reference region or ROI definitions for computing averages, they are influential for partial volume correction.
#
# We have structural MRIs that have already been preprocessed and anatomically labeled. To bring anatomical labels to PET space, we will perform coregistration of the PET and the MRI.
#
# * `dynamic_mean_20min`: We compute the average of PiB time frames $<20$ mins, which will be used in the coregistration.
# * `reorientmri` and `reorientlabel`:  Apply 90, 180, or 270 degree rotations as needed about the $x,y,z$ axes to match the MNI152 orientation.
# * `pet_to_mri`: We use the image with finer spatial resolution (MRI) as the reference, and the PiB 20-min average as the moving image, to perform rigid alignment with normalized mutual information cost function, using FSL's FLIRT method.
# * `invertTransform`: Since we want anatomical labels in PET space, we invert the rigid transformation.
# * `mri_to_pet` and `labels_to_pet`: We apply the inverted transformation to the MRI and anatomical labels to bring them to PET space.
#
# _Note:_ For acquisitions that do not allow for the computation of a 20-min average image, we use the mean of the entire acquisition to perform coregistration with the MRI.


# a placeholder for the realigned 4D PiB
realignedpib = Node(interface=IdentityInterface(fields=['pib']), name="realignedpib")

# Compute 0.75-20 min average that will be used for MRI coregistration
dynamic_mean_20min = Node(interface=DynamicMean(startTime=args.t_start,
                                                endTime=args.t_end_coreg,
                                                weights=dyn_mean_wts),
                          name="dynamic_mean_20min")

# Reorient MRI and label
reorientmri = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientmri")
reorientlabel = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientlabel")

# MRI coregistration, rigid, with normalized mutual information
pet_to_mri = Node(interface=fsl.FLIRT(cost='normmi', dof=6,
                                      searchr_x=[-30,30], searchr_y=[-30,30], searchr_z=[-30,30],
                                      coarse_search=15, fine_search=6),
                  name="pet_to_mri")

invertTransform = Node(interface=fsl.ConvertXFM(invert_xfm=True), name="invertTransform")
mri_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True), name="mri_to_pet")
labels_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, interp='nearestneighbour'), name="labels_to_pet")

coreg_qc = Node(interface=coreg_snapshots(), name="coreg_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','coreg_wf')
datasink.inputs.substitutions = [('_roi',''),
                                 ('_merged',''),
                                 ('mean','avg20min'),
                                 ('flirt','coreg'),
                                 ('_reoriented',''),
                                 ('_masked','')]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

coreg_workflow = Workflow(name="coreg_workflow")
coreg_workflow.base_dir = os.path.join(output_dir,'coreg_workingdir')
coreg_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'coreg_crashdumps')}}
coreg_workflow.connect([(getmri, reorientmri, [('mri','in_file')]),
                        (getlabel, reorientlabel, [('label','in_file')]),

                        (realignedpib, dynamic_mean_20min, [('pib','timeSeriesImgFile')]),
                        (getpettiming, dynamic_mean_20min, [('pettiming','frameTimingCsvFile')]),

                        (dynamic_mean_20min, pet_to_mri, [('meanImgFile','in_file')]),
                        (reorientmri, pet_to_mri, [('out_file','reference')]),

                        (pet_to_mri, invertTransform, [('out_matrix_file','in_file')]),

                        (reorientmri, mri_to_pet, [('out_file','in_file')]),
                        (dynamic_mean_20min, mri_to_pet, [('meanImgFile','reference')]),
                        (invertTransform, mri_to_pet, [('out_file','in_matrix_file')]),

                        (reorientlabel, labels_to_pet, [('out_file','in_file')]),
                        (dynamic_mean_20min, labels_to_pet, [('meanImgFile','reference')]),
                        (invertTransform, labels_to_pet, [('out_file','in_matrix_file')]),

                        (dynamic_mean_20min, coreg_qc, [('meanImgFile','petavgfile')]),
                        (mri_to_pet, coreg_qc, [('out_file','mriregfile')]),

                        # save outputs
                        (dynamic_mean_20min, datasink, [('meanImgFile','avg20min')]), # 0.75-20min average (3D image) used for MRI coregistration
                        (pet_to_mri, datasink, [('out_file','coreg_avg20min'),
                                                ('out_matrix_file','coreg_avg20min.@param')]),
                        (mri_to_pet, datasink, [('out_file','coreg_mri')]), # MRI registered onto PET
                        (labels_to_pet, datasink, [('out_file','coreg_labels')]), # anatomical labels on PET
                        (coreg_qc, datasink, [('coreg_edges','QC'),
                                              ('coreg_overlay_sagittal','QC.@sag'),
                                              ('coreg_overlay_coronal','QC.@cor'),
                                              ('coreg_overlay_axial','QC.@ax')])
                       ])
coreg_workflow.write_graph('coreg.dot', graph2use='colored', simple_form=True)

# We connect the inputs to the realignment workflow:
pib_workflow = Workflow(name="pib_workflow")
pib_workflow.base_dir = output_dir
pib_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'pib_crashdumps')}}

pib_workflow.connect([# PET-to-MRI registration
                      (realign_workflow, coreg_workflow, [('mask4D.out_file','realignedpib.pib')])
                     ])


# ## 4. LABELS
#
# There are two streams of processing we will pursue. First, we generate a conservative reference region definition:
# * `reference_region`: Combines the selected MUSE labels to generate a binary mask.
#
# Second, we generate the set of labels that will be used in partial volume correction. MUSE labels do not include a sulcal CSF label, but this is an important label for PVC. We approximate the sulcal CSF label as the rim around the brain. To this end, we dilate the brain mask, and subtract from it the original brain mask. We designate a label value of $-1$ to this rim, and include it with the ventricle and CSF ROI for PVC.
# * `brainmask`: Threshold the MUSE label image to get a binary brain mask.
# * `dilate`: Dilate the brain mask using a $4\times4\times4$ mm box kernel.
# * `difference`: Subtract dilated brain mask from the orginal mask to get the rim around the brain. This subtraction assigns a value of $-1$ to the rim.
# * `add`: We add the rim image to the MUSE label image. Since the MUSE label image has value $0$ where the rim image has non-zero values, the result is a label image that preserves all the MUSE labels and additionally has a "sulcal CSF" label with value $-1$.
# * `pvc_labels`: We combine the ROIs to generate a collection of binary masks. The result is a 4D volume (with all the binary 3D masks concatenated along 4th dimension). This 4D volume will be an input to the PVC methods.

# placeholder
#muselabel = Node(interface=IdentityInterface(fields=['muselabel']), name="muselabel")
muselabel_unmasked = Node(interface=IdentityInterface(fields=['muselabel_unmasked']), name="muselabel_unmasked")
realign_mask = Node(interface=IdentityInterface(fields=['realign_mask']), name="realign_mask")

muselabel = Node(interface=fsl.ImageMaths(op_string=' -mas ', suffix="_masked"), name="muselabel")

reference_region = Node(interface=CombineROIs(ROI_groupings=list(reference_ROI_grouping.values())),
                        name="reference_region")
whole_brain = Node(interface=CombineROIs(ROI_groupings=list(whole_brain_ROI_grouping.values())),
                   name="whole_brain")

brainmask = Node(interface=fsl.ImageMaths(op_string=' -bin', suffix='_brainmask'), name='brainmask')
dilate = Node(interface=fsl.DilateImage(operation='max', kernel_shape='box', kernel_size=4), name='dilate')
difference = Node(interface=fsl.ImageMaths(op_string=' -sub ', suffix='_diff'), name='difference')
add = Node(interface=fsl.ImageMaths(op_string=' -add ', suffix='_add'), name='add')
pvc_labels = Node(interface=CombineROIs(ROI_groupings=list(pvc_ROI_groupings.values())), name="pvc_labels")

labels_qc = Node(interface=labels_snapshots(labelnames=list(pvc_ROI_groupings.keys())), name="labels_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','labels_wf')
datasink.inputs.substitutions = [('flirt','coreg'),
                                 ('_'+'{:d}'.format(len(reference_ROI_grouping))+'combinedROIs','_refRegion'),
                                 ('_'+'{:d}'.format(len(pvc_ROI_groupings))+'combinedROIs','_pvcLabels'),
                                 ('_add',''),
                                 ('_masked','')
                                ]

labels_workflow = Workflow(name="labels_workflow")
labels_workflow.base_dir = os.path.join(output_dir,'labels_workingdir')
labels_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'labels_crashdumps')}}
labels_workflow.connect([(muselabel_unmasked, muselabel, [('muselabel_unmasked','in_file')]),
                         (realign_mask, muselabel, [('realign_mask','in_file2')]),

                         #(muselabel, reference_region, [('muselabel', 'labelImgFile')]),
                         #(muselabel, whole_brain, [('muselabel', 'labelImgFile')]),
                         (muselabel, reference_region, [('out_file', 'labelImgFile')]),
                         (muselabel, whole_brain, [('out_file', 'labelImgFile')]),

                         # Assign a value of -1 to voxels surrounding the brain
                         # this is an approximation for sulcal CSF label
                         #(muselabel, brainmask, [('muselabel','in_file')]),
                         (muselabel, brainmask, [('out_file','in_file')]),
                         (brainmask, dilate, [('out_file','in_file')]),
                         (brainmask, difference, [('out_file','in_file')]),
                         (dilate, difference, [('out_file','in_file2')]),
                         #(muselabel, add, [('muselabel','in_file')]),
                         (muselabel, add, [('out_file','in_file')]),
                         (difference, add,[('out_file','in_file2')]),
                         (add, pvc_labels, [('out_file','labelImgFile')]),

                         (pvc_labels, labels_qc, [('roi4DMaskFile','labelfile')]),

                         (reference_region, datasink, [('roi4DMaskFile','reference_region')]),
                         (whole_brain, datasink, [('roi4DMaskFile','whole_brain')]),
                         (pvc_labels, datasink, [('roi4DMaskFile','pvc_labels')]),
                         (labels_qc, datasink, [('label_snap','QC')])
                        ])

labels_workflow.write_graph('labels.dot', graph2use='colored', simple_form=True)

refReg_qc = Node(interface=refReg_snapshots(), name="refReg_qc")

pib_workflow.connect([# Anatomical label manipulation
                      (realign_workflow, labels_workflow, [('add1.out_file','realign_mask.realign_mask')]),
                      #(coreg_workflow, labels_workflow, [('labels_to_pet.out_file','muselabel.muselabel')]),
                      (coreg_workflow, labels_workflow, [('labels_to_pet.out_file','muselabel_unmasked.muselabel_unmasked')]),

                      (labels_workflow, refReg_qc, [('reference_region.roi4DMaskFile','maskfile')]),
                      (realign_workflow, refReg_qc, [('mask4D.out_file','petrealignedfile'),
                                                     ('getpettiming.pettiming','frameTimingCsvFile')]),
                      (coreg_workflow, refReg_qc, [('dynamic_mean_20min.meanImgFile','petavgfile')]),
                      (refReg_qc, datasink, [('maskOverlay_axial','QC.@ax'),
                                             ('maskOverlay_coronal','QC.@cor'),
                                             ('maskOverlay_sagittal','QC.@sag'),
                                             ('mask_TAC','QC.@tac')])
                     ])




# ## 5. VOXELWISE PARTIAL VOLUME CORRECTION
if not args.no_pvc:
    timesplit = Node(interface=fsl.Split(dimension='t'), name="timesplit")

    pvc = MapNode(interface=petpvc.PETPVC(pvc='RBV',
                                          fwhm_x=args.psf_fwhm_x,
                                          fwhm_y=args.psf_fwhm_y,
                                          fwhm_z=args.psf_fwhm_z),
                  iterfield=['in_file'], name="pvc")

    timemerge = Node(interface=fsl.Merge(dimension='t'), name="timemerge")


    datasink = Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = os.path.join('output','PVC_wf')
    datasink.inputs.substitutions = [('_roi',''),
                                     ('_merged',''),
                                     ('mean','avg20min'),
                                     ('flirt','coreg'),
                                     ('_reoriented',''),
                                     ('_add',''),
                                     ('_0000',''),
                                     ('_masked','')
                                    ]
    datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

    pvc_workflow = Workflow(name="pvc_workflow")
    pvc_workflow.base_dir = os.path.join(output_dir,'pvc_workingdir')
    pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'pvc_crashdumps')}}

    pvc_workflow.connect([(timesplit, pvc, [('out_files','in_file')]),
                          (pvc, timemerge, [('out_file','in_files')]),

                          (timemerge, datasink, [('merged_file','@')])
                         ])

    pib_workflow.connect([(coreg_workflow, pvc_workflow, [(('realignedpib.pib',get_base_filename),'timesplit.out_base_name'),
                                                           ('realignedpib.pib', 'timesplit.in_file')]),
                          (labels_workflow, pvc_workflow, [('pvc_labels.roi4DMaskFile', 'pvc.mask_file')])
                         ])

# ## 6a. EARLY AMYLOID IMAGE

dynamic_mean_EA = Node(interface=DynamicMean(startTime = args.t_start,
                                             endTime = args.t_end_EA),
                       name="dynamic_mean_EA")
#dynamic_mean_EA.iterables = [('startTime', startTimeList_EA),
#                             ('endTime', endTimeList_EA)]
#dynamic_mean_EA.synchronize = True

# cerebellar GM as reference
ROImean = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean") # note that this is not a trimmed mean!
SUVR_EA = Node(interface=fsl.ImageMaths(), name="SUVR_EA")

ROImeans_EA = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                      ROI_names=list(singleROIs.keys()),
                                                      additionalROIs=list(compositeROIs.values()),
                                                      additionalROI_names=list(compositeROIs.keys()),
                                                      stat='mean'),
                   name="ROImeans_EA")

# whole brain as reference
ROImean_wb = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean_wb") # note that this is not a trimmed mean!
SUVR_EA_wb = Node(interface=fsl.ImageMaths(), name="SUVR_EA_wb")

ROImeans_EA_wb = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                         ROI_names=list(singleROIs.keys()),
                                                         additionalROIs=list(compositeROIs.values()),
                                                         additionalROI_names=list(compositeROIs.keys()),
                                                         stat='mean'),
                      name="ROImeans_EA_wb")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','EA_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_flirt','_coreg'),
                                 ('_reoriented',''),
                                 ('_mean','_EA'),
                                 ('_maths','_intensityNormalized'),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

EA_workflow = Workflow(name="EA_workflow")
EA_workflow.base_dir = os.path.join(output_dir,'EA_workingdir')
EA_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'EA_crashdumps')}}
EA_workflow.connect([(dynamic_mean_EA, ROImean, [('meanImgFile','in_file')]),
                     (dynamic_mean_EA, SUVR_EA, [('meanImgFile','in_file')]),
                     (ROImean, SUVR_EA, [(('out_stat',to_div_string),'op_string')]),
                     (SUVR_EA, ROImeans_EA, [('out_file','imgFile')]),

                     (dynamic_mean_EA, ROImean_wb, [('meanImgFile','in_file')]),
                     (dynamic_mean_EA, SUVR_EA_wb, [('meanImgFile','in_file')]),
                     (ROImean_wb, SUVR_EA_wb, [(('out_stat',to_div_string),'op_string')]),
                     (SUVR_EA_wb, ROImeans_EA_wb, [('out_file','imgFile')]),
                    ])

EA_workflow.write_graph('EA.dot', graph2use='colored', simple_form=True)


pib_workflow.connect([# Early amyloid image
                      (coreg_workflow, EA_workflow, [('realignedpib.pib', 'dynamic_mean_EA.timeSeriesImgFile')]),
                      (realign_workflow, EA_workflow, [('getpettiming.pettiming','dynamic_mean_EA.frameTimingCsvFile')]),
                      (labels_workflow, EA_workflow, [('add.out_file','ROImeans_EA.labelImgFile'),
                                                      ('add.out_file','ROImeans_EA_wb.labelImgFile'),
                                                      ('reference_region.roi4DMaskFile','ROImean.mask_file'),
                                                      ('whole_brain.roi4DMaskFile','ROImean_wb.mask_file')]),
                     ])

# ## 6b. EARLY AMYLOID IMAGE WITH PVC
if not args.no_pvc:
    dynamic_mean_EA_pvc = Node(interface=DynamicMean(startTime = args.t_start,
                                                     endTime = args.t_end_EA),
                               name="dynamic_mean_EA_pvc")
    #dynamic_mean_EA_pvc.iterables = [('startTime', startTimeList_EA),
    #                             ('endTime', endTimeList_EA)]
    #dynamic_mean_EA_pvc.synchronize = True

    # cerebellar GM as reference
    ROImean_pvc = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean_pvc") # note that this is not a trimmed mean!
    SUVR_EA_pvc = Node(interface=fsl.ImageMaths(), name="SUVR_EA_pvc")

    ROImeans_EA_pvc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                          ROI_names=list(singleROIs.keys()),
                                                          additionalROIs=list(compositeROIs.values()),
                                                          additionalROI_names=list(compositeROIs.keys()),
                                                          stat='mean'),
                       name="ROImeans_EA_pvc")

    # whole brain as reference
    ROImean_pvc_wb = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean_pvc_wb") # note that this is not a trimmed mean!
    SUVR_EA_pvc_wb = Node(interface=fsl.ImageMaths(), name="SUVR_EA_pvc_wb")

    ROImeans_EA_pvc_wb = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                             ROI_names=list(singleROIs.keys()),
                                                             additionalROIs=list(compositeROIs.values()),
                                                             additionalROI_names=list(compositeROIs.keys()),
                                                             stat='mean'),
                          name="ROImeans_EA_pvc_wb")

    datasink = Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = os.path.join('output','EA_wf')
    datasink.inputs.substitutions = [('_merged',''),
                                     ('_flirt','_coreg'),
                                     ('_reoriented',''),
                                     ('_mean','_EA'),
                                     ('_maths','_intensityNormalized'),
                                     ('_masked','')
                                    ]
    datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

    EA_pvc_workflow = Workflow(name="EA_pvc_workflow")
    EA_pvc_workflow.base_dir = os.path.join(output_dir,'EA_workingdir')
    EA_pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'EA_crashdumps')}}
    EA_pvc_workflow.connect([(dynamic_mean_EA_pvc, ROImean_pvc, [('meanImgFile','in_file')]),
                         (dynamic_mean_EA_pvc, SUVR_EA_pvc, [('meanImgFile','in_file')]),
                         (ROImean_pvc, SUVR_EA_pvc, [(('out_stat',to_div_string),'op_string')]),
                         (SUVR_EA_pvc, ROImeans_EA_pvc, [('out_file','imgFile')]),

                         (dynamic_mean_EA_pvc, ROImean_pvc_wb, [('meanImgFile','in_file')]),
                         (dynamic_mean_EA_pvc, SUVR_EA_pvc_wb, [('meanImgFile','in_file')]),
                         (ROImean_pvc_wb, SUVR_EA_pvc_wb, [(('out_stat',to_div_string),'op_string')]),
                         (SUVR_EA_pvc_wb, ROImeans_EA_pvc_wb, [('out_file','imgFile')]),
                        ])

    EA_pvc_workflow.write_graph('EA.dot', graph2use='colored', simple_form=True)


    pib_workflow.connect([# Early amyloid image
                          (pvc_workflow, EA_pvc_workflow, [('timemerge.merged_file', 'dynamic_mean_EA_pvc.timeSeriesImgFile')]),
                          (realign_workflow, EA_pvc_workflow, [('getpettiming.pettiming','dynamic_mean_EA_pvc.frameTimingCsvFile')]),
                          (labels_workflow, EA_pvc_workflow, [('add.out_file','ROImeans_EA_pvc.labelImgFile'),
                                                              ('add.out_file','ROImeans_EA_pvc_wb.labelImgFile'),
                                                              ('reference_region.roi4DMaskFile','ROImean_pvc.mask_file'),
                                                              ('whole_brain.roi4DMaskFile','ROImean_pvc_wb.mask_file')]),
                         ])

# ## 7a. SUVR IMAGE

# Compute 50 to 70 min mean image for SUVR computation
dynamic_mean = Node(interface=DynamicMean(startTime=args.t_start_SUVR,
                                          endTime=args.t_end_SUVR,
                                          weights=dyn_mean_wts),
                              name="dynamic_mean")

ROImean = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean")
SUVR = Node(interface=fsl.ImageMaths(), name="SUVR")

ROImeans = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                   ROI_names=list(singleROIs.keys()),
                                                   additionalROIs=list(compositeROIs.values()),
                                                   additionalROI_names=list(compositeROIs.keys()),
                                                   stat='mean'),
                name="ROImeans")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','SUVR_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_flirt','_coreg'),
                                 ('_maths','_suvr'),
                                 ('_reoriented',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

SUVR_workflow = Workflow(name="SUVR_workflow")
SUVR_workflow.base_dir = os.path.join(output_dir,'SUVR_workingdir')
SUVR_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_crashdumps')}}
SUVR_workflow.connect([(getpettiming, dynamic_mean, [('pettiming','frameTimingCsvFile')]),

                       (dynamic_mean, ROImean, [('meanImgFile','in_file')]),
                       (dynamic_mean, SUVR, [('meanImgFile','in_file')]),
                       (ROImean, SUVR, [(('out_stat',to_div_string),'op_string')]),
                       (SUVR, ROImeans, [('out_file','imgFile')]),

                       #(dynamic_mean, datasink, [('meanImgFile','avg50to70min')]),
                       (SUVR, datasink, [('out_file','SUVR')])
                      ])

SUVR_workflow.write_graph('SUVR.dot', graph2use='colored', simple_form=True)

pib_workflow.connect([# SUVR computation
                      (coreg_workflow, SUVR_workflow, [('realignedpib.pib', 'dynamic_mean.timeSeriesImgFile')]),
                      (labels_workflow, SUVR_workflow, [('reference_region.roi4DMaskFile','ROImean.mask_file'),
                                                        ('add.out_file','ROImeans.labelImgFile')])
                     ])

# ## 7b. SUVR IMAGE with PVC
if not args.no_pvc:
    # Compute 50 to 70 min mean image for SUVR computation
    dynamic_mean_pvc = Node(interface=DynamicMean(startTime=args.t_start_SUVR,
                                                  endTime=args.t_end_SUVR,
                                                  weights=dyn_mean_wts),
                            name="dynamic_mean_pvc")

    ROImean_pvc = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean_pvc")
    SUVR_pvc = Node(interface=fsl.ImageMaths(), name="SUVR_pvc")

    ROImeans_pvc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                           ROI_names=list(singleROIs.keys()),
                                                           additionalROIs=list(compositeROIs.values()),
                                                           additionalROI_names=list(compositeROIs.keys()),
                                                           stat='mean'),
                        name="ROImeans_pvc")

    datasink = Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = os.path.join('output','SUVR_wf')
    datasink.inputs.substitutions = [('_merged',''),
                                     ('_flirt','_coreg'),
                                     ('_maths','_suvr'),
                                     ('_reoriented',''),
                                     ('_0000',''),
                                     ('_masked','')
                                    ]
    datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

    SUVR_pvc_workflow = Workflow(name="SUVR_pvc_workflow")
    SUVR_pvc_workflow.base_dir = os.path.join(output_dir,'SUVR_workingdir')
    SUVR_pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_crashdumps')}}
    SUVR_pvc_workflow.connect([(dynamic_mean_pvc, ROImean_pvc, [('meanImgFile','in_file')]),
                               (dynamic_mean_pvc, SUVR_pvc, [('meanImgFile','in_file')]),
                               (ROImean_pvc, SUVR_pvc, [(('out_stat',to_div_string),'op_string')]),
                               (SUVR_pvc, ROImeans_pvc, [('out_file','imgFile')]),

                               (dynamic_mean_pvc, datasink, [('meanImgFile','avg50to70min_pvc')]),
                               (SUVR_pvc, datasink, [('out_file','SUVR_pvc')])
                          ])

    SUVR_pvc_workflow.write_graph('SUVR_pvc.dot', graph2use='colored', simple_form=True)

    pib_workflow.connect([# SUVR computation with PVC
                          (pvc_workflow, SUVR_pvc_workflow, [('timemerge.merged_file', 'dynamic_mean_pvc.timeSeriesImgFile')]),
                          (realign_workflow, SUVR_pvc_workflow, [('getpettiming.pettiming','dynamic_mean_pvc.frameTimingCsvFile')]),
                          (labels_workflow, SUVR_pvc_workflow, [('reference_region.roi4DMaskFile','ROImean_pvc.mask_file'),
                                                                ('add.out_file','ROImeans_pvc.labelImgFile')])
                         ])

# ## 8a. KINETIC PARAMETER IMAGES

kinetic_model = Node(interface=KineticModel(model='SRTM_Zhou2003',
                                  startTime=args.t_start, endTime=args.t_end_kinetic_model,
                                  time_unit='min', startActivity='flat',
                                  weights='frameduration',
                                  halflife=C11_halflife,
                                  fwhm=args.smooth_fwhm), name="kinetic_model")

DVR = Node(interface=fsl.ImageMaths(op_string='-add 1', suffix="add1"), name="DVR")

dvr_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="dvr_nan")
r1_wlr_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="r1_wlr_nan")
r1_lrsc_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="r1_lrsc_nan")

ROImeans_dvr = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                       ROI_names=list(singleROIs.keys()),
                                                       additionalROIs=list(compositeROIs.values()),
                                                       additionalROI_names=list(compositeROIs.keys()),
                                                       stat='mean'),
                    name="ROImeans_dvr")

ROImeans_r1_wlr = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                          ROI_names=list(singleROIs.keys()),
                                                          additionalROIs=list(compositeROIs.values()),
                                                          additionalROI_names=list(compositeROIs.keys()),
                                                          stat='mean'),
                       name="ROImeans_r1_wlr")

ROImeans_r1_lrsc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                           ROI_names=list(singleROIs.keys()),
                                                           additionalROIs=list(compositeROIs.values()),
                                                           additionalROI_names=list(compositeROIs.keys()),
                                                           stat='mean'),
                        name="ROImeans_r1_lrsc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','kinetic_model_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_masked',''),
                                 ('_BPadd1','_DVR'),
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

kinetic_model_workflow = Workflow(name="kinetic_model_workflow")
kinetic_model_workflow.base_dir = os.path.join(output_dir,'kinetic_model_workingdir')
kinetic_model_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'kinetic_model_crashdumps')}}
kinetic_model_workflow.connect([
                                (getpettiming, kinetic_model, [('pettiming','frameTimingCsvFile')]),
                                (kinetic_model, DVR, [('BP','in_file')]),
                                (DVR, ROImeans_dvr, [('out_file','imgFile')]),
                                (kinetic_model, ROImeans_r1_wlr, [('R1','imgFile')]),
                                (kinetic_model, ROImeans_r1_lrsc, [('R1_lrsc','imgFile')]),

                                (DVR, dvr_nan, [('out_file','in_file')]),
                                (kinetic_model, r1_wlr_nan, [('R1','in_file')]),
                                (kinetic_model, r1_lrsc_nan, [('R1_lrsc','in_file')])
                               ])

kinetic_model_workflow.write_graph('kinetic_model.dot', graph2use='colored', simple_form=True)

pib_workflow.connect([# DVR computation
                      (coreg_workflow, kinetic_model_workflow, [('realignedpib.pib', 'kinetic_model.timeSeriesImgFile')]),
                      (labels_workflow, kinetic_model_workflow, [('reference_region.roi4DMaskFile','kinetic_model.refRegionMaskFile'),
                                                                 ('add.out_file','ROImeans_dvr.labelImgFile'),
                                                                 ('add.out_file','ROImeans_r1_wlr.labelImgFile'),
                                                                 ('add.out_file','ROImeans_r1_lrsc.labelImgFile')])
                     ])

# ## 8b. KINETIC PARAMETER IMAGES with PVC
if not args.no_pvc:
    kinetic_model_pvc = Node(interface=KineticModel(model='SRTM_Zhou2003',
                                                    startTime=args.t_start, endTime=args.t_end_kinetic_model,
                                                    time_unit='min', startActivity='flat',
                                                    weights='frameduration',
                                                    halflife=C11_halflife,
                                                    fwhm=args.smooth_fwhm), name="kinetic_model_pvc")

    DVR_pvc = Node(interface=fsl.ImageMaths(op_string='-add 1', suffix="add1"), name="DVR_pvc")

    ROImeans_dvr_pvc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                               ROI_names=list(singleROIs.keys()),
                                                               additionalROIs=list(compositeROIs.values()),
                                                               additionalROI_names=list(compositeROIs.keys()),
                                                               stat='mean'),
                             name="ROImeans_dvr_pvc")

    ROImeans_r1_wlr_pvc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                                  ROI_names=list(singleROIs.keys()),
                                                                  additionalROIs=list(compositeROIs.values()),
                                                                  additionalROI_names=list(compositeROIs.keys()),
                                                                  stat='mean'),
                               name="ROImeans_r1_wlr_pvc")

    ROImeans_r1_lrsc_pvc = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                                   ROI_names=list(singleROIs.keys()),
                                                                   additionalROIs=list(compositeROIs.values()),
                                                                   additionalROI_names=list(compositeROIs.keys()),
                                                                   stat='mean'),
                                name="ROImeans_r1_lrsc_pvc")

    dvr_pvc_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="dvr_pvc_nan")
    r1_wlr_pvc_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="r1_wlr_pvc_nan")
    r1_lrsc_pvc_nan = Node(interface=fsl.ImageMaths(op_string='-nan'),name="r1_lrsc_pvc_nan")

    datasink = Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = os.path.join('output','DVR_wf')
    datasink.inputs.substitutions = [('_merged',''),
                                     ('_reoriented',''),
                                     ('_0000',''),
                                     ('_masked',''),
                                     ('_BPadd1','_DVR'),
                                    ]
    datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                            (r'_\d+\.\d+min',r'')]

    kinetic_model_pvc_workflow = Workflow(name="kinetic_model_pvc_workflow")
    kinetic_model_pvc_workflow.base_dir = os.path.join(output_dir,'kinetic_model_workingdir')
    kinetic_model_pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'kinetic_model_crashdumps')}}
    kinetic_model_pvc_workflow.connect([
                                    (getpettiming, kinetic_model_pvc, [('pettiming','frameTimingCsvFile')]),
                                    (kinetic_model_pvc, DVR_pvc, [('BP','in_file')]),
                                    (DVR_pvc, ROImeans_dvr_pvc, [('out_file','imgFile')]),
                                    (kinetic_model_pvc, ROImeans_r1_wlr_pvc, [('R1','imgFile')]),
                                    (kinetic_model_pvc, ROImeans_r1_lrsc_pvc, [('R1_lrsc','imgFile')]),
                                    (DVR_pvc, dvr_pvc_nan, [('out_file','in_file')]),
                                    (kinetic_model_pvc, r1_wlr_pvc_nan, [('R1','in_file')]),
                                    (kinetic_model_pvc, r1_lrsc_pvc_nan, [('R1_lrsc','in_file')]),
                                   ])

    kinetic_model_pvc_workflow.write_graph('kinetic_model_pvc.dot', graph2use='colored', simple_form=True)

    pib_workflow.connect([# DVR computation
                          (pvc_workflow, kinetic_model_pvc_workflow, [('timemerge.merged_file', 'kinetic_model_pvc.timeSeriesImgFile')]),
                          (labels_workflow, kinetic_model_pvc_workflow, [('reference_region.roi4DMaskFile','kinetic_model_pvc.refRegionMaskFile'),
                                                                         ('add.out_file','ROImeans_dvr_pvc.labelImgFile'),
                                                                         ('add.out_file','ROImeans_r1_wlr_pvc.labelImgFile'),
                                                                         ('add.out_file','ROImeans_r1_lrsc_pvc.labelImgFile')])
                         ])

# ## 9. MNI SPACE

# placeholders
mri = Node(interface=IdentityInterface(fields=['mri']), name="mri")
pib20min = Node(interface=IdentityInterface(fields=['pib20min']), name="pib20min")

# Quick registration to MNI template
mri_to_mni = Node(interface=fsl.FLIRT(dof=12,reference=template), name="mri_to_mni")

mergexfm = Node(interface=fsl.ConvertXFM(concat_xfm=True), name="mergexfm")

warp_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, reference=template), name='warp_pet')

warp_dvr = warp_pet.clone(name='warp_dvr')
warp_r1_wlr = warp_pet.clone(name='warp_r1_wlr')
warp_r1_lrsc = warp_pet.clone(name='warp_r1_lrsc')
warp_suvr = warp_pet.clone(name='warp_suvr')
warp_ea = warp_pet.clone(name='warp_ea')
warp_ea_wb = warp_pet.clone(name='warp_ea_wb')


# Gaussian smoothing
smooth_ea = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_ea")
smooth_ea_wb = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_ea_wb")
smooth_suvr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_suvr")

smooth_dvr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_dvr")
smooth_r1_wlr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_r1_wlr")
smooth_r1_lrsc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_r1_lrsc")

# Triplanar snapshots
dvr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5), name="dvr_qc")

r1_wlr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.0), name="r1_wlr_qc")
r1_lrsc_qc = r1_wlr_qc.clone(name="r1_lrsc_qc")

suvr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=4.0), name="suvr_qc")
ea_qc = suvr_qc.clone(name="ea_qc")
ea_wb_qc = suvr_qc.clone(name="ea_wb_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('flirt','mni'),
                                 ('_0000',''),
                                 ('_masked',''),
                                 ('_BPadd1','_DVR'),
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

MNI_workflow = Workflow(name="MNI_workflow")
MNI_workflow.base_dir = os.path.join(output_dir,'MNI_workingdir')
MNI_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'MNI_crashdumps')}}
MNI_workflow.connect([(mri, mri_to_mni, [('mri','in_file')]),

                      (mri_to_mni, mergexfm, [('out_matrix_file','in_file2')]),

                      (pib20min, warp_pet, [('pib20min','in_file')]),
                      (mergexfm, warp_pet, [('out_file', 'in_matrix_file')]),

                      (mergexfm, warp_dvr, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_r1_wlr, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_r1_lrsc, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_suvr, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_ea, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_ea_wb, [('out_file','in_matrix_file')]),

                      (warp_dvr, smooth_dvr, [('out_file','in_file')]),
                      (warp_r1_wlr, smooth_r1_wlr, [('out_file','in_file')]),
                      (warp_r1_lrsc, smooth_r1_lrsc, [('out_file','in_file')]),
                      (warp_suvr, smooth_suvr, [('out_file','in_file')]),
                      (warp_ea, smooth_ea, [('out_file','in_file')]),
                      (warp_ea_wb, smooth_ea_wb, [('out_file','in_file')]),

                      (mri_to_mni, dvr_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, r1_wlr_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, r1_lrsc_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, suvr_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, ea_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, ea_wb_qc, [('out_file','bgimgfile')]),

                      (warp_dvr, dvr_qc, [('out_file','imgfile')]),
                      (warp_r1_wlr, r1_wlr_qc, [('out_file','imgfile')]),
                      (warp_r1_lrsc, r1_lrsc_qc, [('out_file','imgfile')]),
                      (smooth_suvr, suvr_qc, [('smoothed_file','imgfile')]),
                      (smooth_ea, ea_qc, [('smoothed_file','imgfile')]),
                      (smooth_ea_wb, ea_wb_qc, [('smoothed_file','imgfile')]),

                      (mri_to_mni, datasink, [('out_file','warped_mri'),
                                              ('out_matrix_file','warped_mri.@param')]),
                      (warp_pet, datasink, [('out_file','warped_pet')]),

                      (warp_dvr, datasink, [('out_file','warped_dvr')]),
                      (warp_r1_wlr, datasink, [('out_file','warped_r1_wlr')]),
                      (warp_r1_lrsc, datasink, [('out_file','warped_r1_lrsc')]),
                      (warp_suvr, datasink, [('out_file','warped_suvr')]),
                      (warp_ea, datasink, [('out_file','warped_ea')]),
                      (warp_ea_wb, datasink, [('out_file','warped_ea_wb')]),
                      (smooth_suvr, datasink, [('smoothed_file','warped_suvr.@smooth')]),
                      (smooth_ea, datasink, [('smoothed_file','warped_ea.@smooth')]),
                      (smooth_ea_wb, datasink, [('smoothed_file','warped_ea_wb.@smooth')]),

                      (dvr_qc, datasink, [('triplanar','QC')]),
                      (r1_wlr_qc, datasink, [('triplanar','QC.@r1_wlr')]),
                      (r1_lrsc_qc, datasink, [('triplanar','QC.@r1_lrsc')]),
                      (suvr_qc, datasink, [('triplanar','QC@SUVR')]),
                      (ea_qc, datasink, [('triplanar','QC@EA')]),
                      (ea_wb_qc, datasink, [('triplanar','QC@EA_wb')]),
                     ])

pib_workflow.connect([# MNI space normalization
                      (coreg_workflow, MNI_workflow, [('reorientmri.out_file','mri.mri'),
                                                      ('dynamic_mean_20min.meanImgFile','pib20min.pib20min'),
                                                      ('pet_to_mri.out_matrix_file','mergexfm.in_file')]),
                      #(kinetic_model_workflow, MNI_workflow, [('DVR.out_file','warp_dvr.in_file'),
                      #                                        ('kinetic_model.R1','warp_r1_wlr.in_file'),
                      #                                        ('kinetic_model.R1_lrsc','warp_r1_lrsc.in_file')]),
                      (kinetic_model_workflow, MNI_workflow, [('dvr_nan.out_file','warp_dvr.in_file'),
                                                              ('r1_wlr_nan.out_file','warp_r1_wlr.in_file'),
                                                              ('r1_lrsc_nan.out_file','warp_r1_lrsc.in_file')]),
                      (SUVR_workflow, MNI_workflow, [('SUVR.out_file','warp_suvr.in_file')]),
                      (EA_workflow, MNI_workflow, [('SUVR_EA.out_file','warp_ea.in_file'),
                                                   ('SUVR_EA_wb.out_file','warp_ea_wb.in_file')]),
                     ])

if not args.no_pvc:
    warp_dvr_pvc = warp_pet.clone(name='warp_dvr_pvc')
    warp_r1_wlr_pvc = warp_pet.clone(name='warp_r1_wlr_pvc')
    warp_r1_lrsc_pvc = warp_pet.clone(name='warp_r1_lrsc_pvc')
    warp_suvr_pvc = warp_pet.clone(name='warp_suvr_pvc')
    warp_ea_pvc = warp_pet.clone(name='warp_ea_pvc')
    warp_ea_pvc_wb = warp_pet.clone(name='warp_ea_pvc_wb')


    # Gaussian smoothing
    smooth_ea_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_ea_pvc")
    smooth_ea_pvc_wb = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_ea_pvc_wb")
    smooth_suvr_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_suvr_pvc")

    smooth_dvr_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_dvr_pvc")
    smooth_r1_wlr_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_r1_wlr_pvc")
    smooth_r1_lrsc_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_r1_lrsc_pvc")

    # Triplanar snapshots
    dvr_pvc_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5), name="dvr_pvc_qc")

    r1_wlr_pvc_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.0), name="r1_wlr_pvc_qc")
    r1_lrsc_pvc_qc = r1_wlr_qc.clone(name="r1_lrsc_pvc_qc")

    suvr_pvc_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=4.0), name="suvr_pvc_qc")
    ea_pvc_qc = suvr_qc.clone(name="ea_pvc_qc")
    ea_pvc_wb_qc = suvr_qc.clone(name="ea_pvc_wb_qc")

    MNI_workflow.connect([(mergexfm, warp_dvr_pvc, [('out_file','in_matrix_file')]),
                          (mergexfm, warp_r1_wlr_pvc, [('out_file','in_matrix_file')]),
                          (mergexfm, warp_r1_lrsc_pvc, [('out_file','in_matrix_file')]),
                          (mergexfm, warp_suvr_pvc, [('out_file','in_matrix_file')]),
                          (mergexfm, warp_ea_pvc, [('out_file','in_matrix_file')]),
                          (mergexfm, warp_ea_pvc_wb, [('out_file','in_matrix_file')]),

                          (warp_dvr_pvc, smooth_dvr_pvc, [('out_file','in_file')]),
                          (warp_r1_wlr_pvc, smooth_r1_wlr_pvc, [('out_file','in_file')]),
                          (warp_r1_lrsc_pvc, smooth_r1_lrsc_pvc, [('out_file','in_file')]),
                          (warp_suvr_pvc, smooth_suvr_pvc, [('out_file','in_file')]),
                          (warp_ea_pvc, smooth_ea_pvc, [('out_file','in_file')]),
                          (warp_ea_pvc_wb, smooth_ea_pvc_wb, [('out_file','in_file')]),

                          (mri_to_mni, dvr_pvc_qc, [('out_file','bgimgfile')]),
                          (mri_to_mni, r1_wlr_pvc_qc, [('out_file','bgimgfile')]),
                          (mri_to_mni, r1_lrsc_pvc_qc, [('out_file','bgimgfile')]),
                          (mri_to_mni, suvr_pvc_qc, [('out_file','bgimgfile')]),
                          (mri_to_mni, ea_pvc_qc, [('out_file','bgimgfile')]),
                          (mri_to_mni, ea_pvc_wb_qc, [('out_file','bgimgfile')]),

                          (warp_dvr_pvc, dvr_pvc_qc, [('out_file','imgfile')]),
                          (warp_r1_wlr_pvc, r1_wlr_pvc_qc, [('out_file','imgfile')]),
                          (warp_r1_lrsc_pvc, r1_lrsc_pvc_qc, [('out_file','imgfile')]),
                          (smooth_suvr_pvc, suvr_pvc_qc, [('smoothed_file','imgfile')]),
                          (smooth_ea_pvc, ea_pvc_qc, [('smoothed_file','imgfile')]),
                          (smooth_ea_pvc_wb, ea_pvc_wb_qc, [('smoothed_file','imgfile')]),

                          (warp_dvr_pvc, datasink, [('out_file','warped_dvr_pvc')]),
                          (warp_r1_wlr_pvc, datasink, [('out_file','warped_r1_wlr_pvc')]),
                          (warp_r1_lrsc_pvc, datasink, [('out_file','warped_r1_lrsc_pvc')]),
                          (warp_suvr_pvc, datasink, [('out_file','warped_suvr_pvc')]),
                          (warp_ea_pvc, datasink, [('out_file','warped_ea_pvc')]),
                          (warp_ea_pvc_wb, datasink, [('out_file','warped_ea_pvc_wb')]),
                          (smooth_suvr_pvc, datasink, [('smoothed_file','warped_suvr_pvc.@smooth')]),
                          (smooth_ea_pvc, datasink, [('smoothed_file','warped_ea_pvc.@smooth')]),
                          (smooth_ea_pvc_wb, datasink, [('smoothed_file','warped_ea_pvc_wb.@smooth')]),

                          (dvr_pvc_qc, datasink, [('triplanar','QC.@dvr_pvc')]),
                          (r1_wlr_pvc_qc, datasink, [('triplanar','QC.@r1_wlr_pvc')]),
                          (r1_lrsc_pvc_qc, datasink, [('triplanar','QC.@r1_lrsc_pvc')]),
                          (suvr_pvc_qc, datasink, [('triplanar','QC@SUVR_pvc')]),
                          (ea_pvc_qc, datasink, [('triplanar','QC@EA_pvc')]),
                          (ea_pvc_wb_qc, datasink, [('triplanar','QC@EA_pvc_wb')]),
                         ])

    pib_workflow.connect([(kinetic_model_pvc_workflow, MNI_workflow, [('dvr_pvc_nan.out_file','warp_dvr_pvc.in_file'),
                                                                      ('r1_wlr_pvc_nan.out_file','warp_r1_wlr_pvc.in_file'),
                                                                      ('r1_lrsc_pvc_nan.out_file','warp_r1_lrsc_pvc.in_file')]),
                          (SUVR_pvc_workflow, MNI_workflow, [('SUVR_pvc.out_file','warp_suvr_pvc.in_file')]),
                          (EA_pvc_workflow, MNI_workflow, [('SUVR_EA_pvc.out_file','warp_ea_pvc.in_file'),
                                                           ('SUVR_EA_pvc_wb.out_file','warp_ea_pvc_wb.in_file')]),
                         ])

MNI_workflow.write_graph('MNI.dot', graph2use='colored', simple_form=True)

# ## 10. MNI SPACE -- deformable alignment onto study-specific template

# convert the affine transform from PET to MRI obtained via FSL to ANTs format
convert2itk = Node(interface=c3.C3dAffineTool(fsl2ras=True, itk_transform=True), name="convert2itk")

# merge affine transform to MRI with composite warp to study-specific template in MNI space
# first should be composite
# second should be pet-to-mri
merge_list = Node(Merge(2), name='merge_list')
merge_list.inputs.in1 = args.mnitransform

warp_dvr = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni),
                    name="warp_dvr")
warp_r1_wlr = warp_dvr.clone(name='warp_r1_wlr')
warp_r1_lrsc = warp_dvr.clone(name='warp_r1_lrsc')
warp_ea = warp_dvr.clone(name='warp_ea')
warp_ea_wb = warp_dvr.clone(name='warp_ea_wb')
warp_suvr = warp_dvr.clone(name='warp_suvr')

smooth_dvr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_dvr")
smooth_r1_wlr = smooth_dvr.clone(name='smooth_r1_wlr')
smooth_r1_lrsc = smooth_dvr.clone(name='smooth_r1_lrsc')
smooth_ea = smooth_dvr.clone(name='smooth_ea')
smooth_ea_wb = smooth_dvr.clone(name='smooth_ea_wb')
smooth_suvr = smooth_dvr.clone(name='smooth_suvr')

# Triplanar snapshots
dvr_qc = Node(interface=triplanar_snapshots(bgimgfile = blsa_template_mni,
                                            alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5),
                  name="dvr_qc")
r1_wlr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.0), name="r1_wlr_qc")
r1_lrsc_qc = r1_wlr_qc.clone(name="r1_lrsc_qc")

suvr_qc = Node(interface=triplanar_snapshots(bgimgfile = blsa_template_mni,
                                             alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=4.0),
               name="suvr_qc")
ea_qc = suvr_qc.clone(name="ea_qc")
ea_wb_qc = suvr_qc.clone(name="ea_wb_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_deform_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('flirt','mni'),
                                 ('_0000',''),
                                 ('_masked',''),
                                 ('_BPadd1','_DVR'),
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

MNI_deform_workflow = Workflow(name="MNI_deform_workflow")
MNI_deform_workflow.base_dir = os.path.join(output_dir,'MNI_deform_workingdir')
MNI_deform_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'MNI_deform_crashdumps')}}
MNI_deform_workflow.connect([(convert2itk, merge_list, [('itk_transform','in2')]),

                             (merge_list, warp_dvr, [('out','transforms')]),
                             (merge_list, warp_r1_wlr, [('out','transforms')]),
                             (merge_list, warp_r1_lrsc, [('out','transforms')]),
                             (merge_list, warp_ea, [('out','transforms')]),
                             (merge_list, warp_ea_wb, [('out','transforms')]),
                             (merge_list, warp_suvr, [('out','transforms')]),

                             (warp_dvr, smooth_dvr, [('output_image','in_file')]),
                             (warp_r1_wlr, smooth_r1_wlr, [('output_image','in_file')]),
                             (warp_r1_lrsc, smooth_r1_lrsc, [('output_image','in_file')]),
                             (warp_ea, smooth_ea, [('output_image','in_file')]),
                             (warp_ea_wb, smooth_ea_wb, [('output_image','in_file')]),
                             (warp_suvr, smooth_suvr, [('output_image','in_file')]),

                             (warp_dvr, dvr_qc, [('output_image','imgfile')]),
                             (warp_r1_wlr, r1_wlr_qc, [('output_image','imgfile')]),
                             (warp_r1_lrsc, r1_lrsc_qc, [('output_image','imgfile')]),
                             (smooth_ea, ea_qc, [('smoothed_file','imgfile')]),
                             (smooth_ea_wb, ea_wb_qc, [('smoothed_file','imgfile')]),
                             (smooth_suvr, suvr_qc, [('smoothed_file','imgfile')]),

                             (warp_dvr, datasink, [('output_image','warped_dvr')]),
                             (warp_r1_wlr, datasink, [('output_image','warped_r1_wlr')]),
                             (warp_r1_lrsc, datasink, [('output_image','warped_r1_lrsc')]),
                             (warp_ea, datasink, [('output_image','warped_ea')]),
                             (warp_ea_wb, datasink, [('output_image','warped_ea_wb')]),
                             (warp_suvr, datasink, [('output_image','warped_suvr')]),

                             (dvr_qc, datasink, [('triplanar','QC')]),
                             (r1_wlr_qc, datasink, [('triplanar','QC.@r1_wlr')]),
                             (r1_lrsc_qc, datasink, [('triplanar','QC.@r1_lrsc')]),
                             (ea_qc, datasink, [('triplanar','QC@EA')]),
                             (ea_wb_qc, datasink, [('triplanar','QC@EA_wb')]),
                             (suvr_qc, datasink, [('triplanar','QC@SUVR')]),
                            ])

pib_workflow.connect([(coreg_workflow, MNI_deform_workflow, [('dynamic_mean_20min.meanImgFile','convert2itk.source_file'),
                                                             ('reorientmri.out_file','convert2itk.reference_file'),
                                                             ('pet_to_mri.out_matrix_file','convert2itk.transform_file')]),
                      (kinetic_model_workflow, MNI_deform_workflow, [('DVR.out_file','warp_dvr.input_image'),
                                                                     ('kinetic_model.R1','warp_r1_wlr.input_image'),
                                                                     ('kinetic_model.R1_lrsc','warp_r1_lrsc.input_image')]),
                      (SUVR_workflow, MNI_deform_workflow, [('SUVR.out_file','warp_suvr.input_image')]),
                      (EA_workflow, MNI_deform_workflow, [('SUVR_EA.out_file','warp_ea.input_image'),
                                                          ('SUVR_EA_wb.out_file','warp_ea_wb.input_image')]),
                     ])

if not args.no_pvc:
    warp_dvr_pvc = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni),
                        name="warp_dvr_pvc")
    warp_r1_wlr_pvc = warp_dvr.clone(name='warp_r1_wlr_pvc')
    warp_r1_lrsc_pvc = warp_dvr.clone(name='warp_r1_lrsc_pvc')
    warp_ea_pvc = warp_dvr.clone(name='warp_ea_pvc')
    warp_ea_pvc_wb = warp_dvr.clone(name='warp_ea_pvc_wb')
    warp_suvr_pvc = warp_dvr.clone(name='warp_suvr_pvc')

    smooth_dvr_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_dvr_pvc")
    smooth_r1_wlr_pvc = smooth_dvr.clone(name='smooth_r1_wlr_pvc')
    smooth_r1_lrsc_pvc = smooth_dvr.clone(name='smooth_r1_lrsc_pvc')
    smooth_ea_pvc = smooth_dvr.clone(name='smooth_ea_pvc')
    smooth_ea_pvc_wb = smooth_dvr.clone(name='smooth_ea_pvc_wb')
    smooth_suvr_pvc = smooth_dvr.clone(name='smooth_suvr_pvc')

    # Triplanar snapshots
    dvr_pvc_qc = Node(interface=triplanar_snapshots(bgimgfile = blsa_template_mni,
                                                alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5),
                      name="dvr_pvc_qc")
    r1_wlr_pvc_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.0), name="r1_wlr_pvc_qc")
    r1_lrsc_pvc_qc = r1_wlr_qc.clone(name="r1_lrsc_pvc_qc")

    suvr_pvc_qc = Node(interface=triplanar_snapshots(bgimgfile = blsa_template_mni,
                                                 alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=4.0),
                   name="suvr_pvc_qc")
    ea_pvc_qc = suvr_qc.clone(name="ea_pvc_qc")
    ea_pvc_wb_qc = suvr_qc.clone(name="ea_pvc_wb_qc")

    MNI_deform_workflow.connect([(merge_list, warp_dvr_pvc, [('out','transforms')]),
                                 (merge_list, warp_r1_wlr_pvc, [('out','transforms')]),
                                 (merge_list, warp_r1_lrsc_pvc, [('out','transforms')]),
                                 (merge_list, warp_ea_pvc, [('out','transforms')]),
                                 (merge_list, warp_ea_pvc_wb, [('out','transforms')]),
                                 (merge_list, warp_suvr_pvc, [('out','transforms')]),

                                 (warp_dvr_pvc, smooth_dvr_pvc, [('output_image','in_file')]),
                                 (warp_r1_wlr_pvc, smooth_r1_wlr_pvc, [('output_image','in_file')]),
                                 (warp_r1_lrsc_pvc, smooth_r1_lrsc_pvc, [('output_image','in_file')]),
                                 (warp_ea_pvc, smooth_ea_pvc, [('output_image','in_file')]),
                                 (warp_ea_pvc_wb, smooth_ea_pvc_wb, [('output_image','in_file')]),
                                 (warp_suvr_pvc, smooth_suvr_pvc, [('output_image','in_file')]),

                                 (warp_dvr_pvc, dvr_pvc_qc, [('output_image','imgfile')]),
                                 (warp_r1_wlr_pvc, r1_wlr_pvc_qc, [('output_image','imgfile')]),
                                 (warp_r1_lrsc_pvc, r1_lrsc_pvc_qc, [('output_image','imgfile')]),
                                 (smooth_ea_pvc, ea_pvc_qc, [('smoothed_file','imgfile')]),
                                 (smooth_ea_pvc_wb, ea_pvc_wb_qc, [('smoothed_file','imgfile')]),
                                 (smooth_suvr_pvc, suvr_pvc_qc, [('smoothed_file','imgfile')]),

                                 (warp_dvr_pvc, datasink, [('output_image','warped_dvr_pvc')]),
                                 (warp_r1_wlr_pvc, datasink, [('output_image','warped_r1_wlr_pvc')]),
                                 (warp_r1_lrsc_pvc, datasink, [('output_image','warped_r1_lrsc_pvc')]),
                                 (warp_ea_pvc, datasink, [('output_image','warped_ea_pvc')]),
                                 (warp_ea_pvc_wb, datasink, [('output_image','warped_ea_pvc_wb')]),
                                 (warp_suvr_pvc, datasink, [('output_image','warped_suvr_pvc')]),

                                 (dvr_pvc_qc, datasink, [('triplanar','QC.@dvr_pvc')]),
                                 (r1_wlr_pvc_qc, datasink, [('triplanar','QC.@r1_wlr_pvc')]),
                                 (r1_lrsc_pvc_qc, datasink, [('triplanar','QC.@r1_lrsc_pvc')]),
                                 (ea_pvc_qc, datasink, [('triplanar','QC@EA_pvc')]),
                                 (ea_pvc_wb_qc, datasink, [('triplanar','QC@EA_pvc_wb')]),
                                 (suvr_pvc_qc, datasink, [('triplanar','QC@SUVR_pvc')]),
                                ])

    pib_workflow.connect([(kinetic_model_pvc_workflow, MNI_deform_workflow, [('DVR_pvc.out_file','warp_dvr_pvc.input_image'),
                                                                             ('kinetic_model_pvc.R1','warp_r1_wlr_pvc.input_image'),
                                                                             ('kinetic_model_pvc.R1_lrsc','warp_r1_lrsc_pvc.input_image')]),
                          (SUVR_pvc_workflow, MNI_deform_workflow, [('SUVR_pvc.out_file','warp_suvr_pvc.input_image')]),
                          (EA_pvc_workflow, MNI_deform_workflow, [('SUVR_EA_pvc.out_file','warp_ea_pvc.input_image'),
                                                                  ('SUVR_EA_pvc_wb.out_file','warp_ea_pvc_wb.input_image')]),
                         ])

MNI_deform_workflow.write_graph('MNI_deform.dot', graph2use='colored', simple_form=True)

pib_workflow.write_graph('pib_MUSE.dot', graph2use='colored', simple_form=True)

result = pib_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
