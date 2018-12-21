# AV1451 PET PROCESSING IN PET SPACE USING MUSE LABELS
#
# The processing steps can be summarized as follows:
# 1. Time frame alignment
# 2. MRI-PET coregistration (MRIs have already been processed with FreeSurfer to define anatomical regions)
# 3. Extraction of SUVR image
# 4. ROI summary calculation
# 5. Spatial normalization of SUVR images to MNI space
#
# Step 2 is performed in two stages: first, MRI with skull is used to perform and initial coregistration. AV1451 is masked using a dilated brainmask obtained from the MRI, and coregistration is repeated using skull-stripped MRI and masked AV1451.
#
# Steps 3-5 will be performed with and without partial volume correction.

# running on nilab10 using petpipeline_05152018 conda environment
# before running this code, make sure that you've activated this conda env:
#   source activate petpipeline_05152018


# import packages
import math, os, sys, logging
import pandas as pd
import numpy as np
import scipy as sp
from glob import glob
from collections import OrderedDict

# nipype
import nipype.interfaces.io as nio
from nipype.interfaces import spm, fsl, petpvc, ants, c3
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import Function, IdentityInterface, Merge
from nipype import config, logging
config.enable_debug_mode()
logging.update_logging(config)

# custom nipype wrappers and functions
from temporalimage.nipype_wrapper import SplitTimeSeries, DynamicMean
# Pad4DImage, Unpad4DImage,
from nipype_misc_funcs import CombineROIs, Unpad4DImage, \
                              get_base_filename, to_div_string, ROI_stats_to_spreadsheet
from nipype_snapshot_funcs import realign_snapshots, coreg_snapshots, \
                                  labels_snapshots, refReg_snapshots, \
                                  triplanar_snapshots, mosaic

# this needs to be more flexible
from MUSE_label_scheme import whole_brain_ROI_grouping, reference_ROI_grouping, \
                              pvc_ROI_groupings, ROIs
from SUIT_label_scheme import inf_cerebellum_ROI_grouping, sup_cerebellum_ROI_grouping
singleROIs = OrderedDict({k: v for k, v in ROIs.items() if type(v) is int})
compositeROIs = OrderedDict({k: v for k, v in ROIs.items() if type(v) is list})

import argparse

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("pet4D", help="path to 4D AV1451-PET image")
parser.add_argument("pettiming", help="path to csv file describing PET timing info")
parser.add_argument("mriskull", help="path to preprocessed MRI image with skull")
parser.add_argument("mri", help="path to preprocessed MRI image without skull")
parser.add_argument("label", help="path to anatomical label image (in MRI space)")
parser.add_argument("mnitransform", help="path to composite (deformable) transform that takes MRI to MNI space")
parser.add_argument("outputdir", help="output directory")


# optional arguments
parser.add_argument("--t_start_SUVR", type=float, default=80,
                    help="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)")
parser.add_argument("--t_end_SUVR", type=float, default=100,
                    help="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)")

parser.add_argument("--unpad_size", type=int, default=7,
                    help="number of slices to remove from each side of the PET image")
parser.add_argument("--signal_threshold", type=float, default=0.,
                    help="ignore signal below this threshold when computing ROI averages")

parser.add_argument("--no_pvc", action="store_true",
                    help="do not perform partial volume correction")

parser.add_argument("-x", "--psf_fwhm_x", type=float, default=2.5,
                    help="PET scanner PSF FWHM along x (in mm)")
parser.add_argument("-y", "--psf_fwhm_y", type=float, default=2.5,
                    help="PET scanner PSF FWHM along y (in mm)")
parser.add_argument("-z", "--psf_fwhm_z", type=float, default=2.5,
                    help="PET scanner PSF FWHM along z (in mm)")

parser.add_argument("-s","--smooth_fwhm", type=float, default=4.0,
                    help="FWHM of Gaussian smoothing filter (in mm)")

parser.add_argument("-n","--n_procs", type=int, default=12,
                    help="number of parallel processes")

args = parser.parse_args()

# number of parallel processes
n_procs = args.n_procs


# Set up standalone SPM
spm.SPMCommand.set_mlab_paths(matlab_cmd=os.environ['SPMMCRCMD'], use_mcr=True)

suit_labels = os.path.join(fsl.Info.standard_image(''),os.pardir,'atlases','Cerebellum','Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz')

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

F18_halflife = 109.771 # mins

dyn_mean_wts = None


'''
# PVC smoothing parameters: PET scanner PSF FWHM (in mm)
pvc_fwhm_x = 2.5
pvc_fwhm_y = 2.5
pvc_fwhm_z = 2.5

# Smoothing parameter (in mm) used for SUVR
smooth_fwhm = 4
'''


# ## 1. INPUTS

# We set up the nipype Nodes that will act as the inputs to our Workflows.
# Nodes allow for the passing of the 4D AV1451 PET, processed MRI,
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

getmriskull = Node(interface=IdentityInterface(fields=['mriskull']), name="getmriskull")
getmriskull.inputs.mriskull = args.mriskull

getlabel = Node(interface=IdentityInterface(fields=['label']), name="getlabel")
getlabel.inputs.label = args.label


# 2. REALIGN
#
# The goal of the realign workflow is to compute a spatially-aligned dynamic AV1451-PET image by removing subject motion.
#
# * `reorient`: We start by reorienting the PET image to match the orientation of MNI152 templates. This is not spatial normalization - no registration is performed. We simply apply 90, 180, or 270 degree rotations as needed about the $x,y,z$ axes to match the MNI152 orientation.
#
# We are interested in data acquired between 80-100 min.
# * `extract_time`: Get the time frames corresponding to 80-100 minutes of scanning.
#
# We will align each time frame to the average of the 80-100 min acquisition, and compute the mean of the aligned time frames.
# * `realign`: This is the SPM function to do the actual work. SPM uses a two-pass procedure for aligning to the mean.
# * `dynamic_mean`: Compute the average of the aligned time frames.
# * `smooth`: Finally, we smooth the mean image.

# Reorient
reorient = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorient")

unpad = Node(interface=Unpad4DImage(padsize=args.unpad_size), name="unpad")

# Realign time frames to mean frame
realign = Node(interface=spm.Realign(), name="realign")
# === Estimate options ===
realign.inputs.quality = 1
realign.inputs.separation = 2 # Separation in mm b/w points sampled in reference
                              # image. Smaller more accurate but slower
realign.inputs.fwhm = 7 # FWHM in mm of the Gaussian smoothing kernel applied
                        # to the images before estimating realignment parameters
realign.inputs.register_to_mean = True # align to the first time frame,
realign.inputs.interp = 2 # degree of interpolation. Higher is better but slower
realign.inputs.wrap = [0, 0, 0] # no wrap around in PET
# === Reslice options ===
realign.inputs.write_which = [2, 1] #
realign.inputs.write_interp = 4
realign.inputs.write_mask = True
realign.inputs.write_wrap = [0, 0, 0]
realign.inputs.out_prefix = 'r'
realign.inputs.jobtype = 'estwrite'
realign.use_mcr = True # run using standalone SPM (without MATLAB)

# nan mask
nan_mask_4D = Node(interface=fsl.ImageMaths(op_string=' -nanm', suffix='_nanmask'), name='nan_mask_4D')
nan_mask = Node(interface=fsl.ImageMaths(op_string=' -Tmax', suffix='_Tmax'), name='nan_mask')
mulneg1 = Node(interface=fsl.ImageMaths(op_string=' -mul -1', suffix='_mul'), name='mulneg1')
notnan_mask = Node(interface=fsl.ImageMaths(op_string=' -add 1', suffix='_add'), name='notnan_mask')

# Replace nan values after realignment with 0
nan_to_0 = Node(interface=fsl.ImageMaths(op_string=' -nan', suffix='_nanto0'), name='nan_to_0')

pet4D_masked = Node(interface=fsl.ImageMaths(op_string=' -mas', suffix='_masked'), name='pet4D_masked')

dynamic_mean = Node(interface=DynamicMean(startTime=args.t_start_SUVR,
                                          endTime=args.t_end_SUVR,
                                          weights=dyn_mean_wts), name="dynamic_mean")

# Gaussian smoothing
smooth = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth")

realign_qc = Node(interface=realign_snapshots(splitTime=0), name="realign_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','realign_wf')
datasink.inputs.substitutions = [('_roi',''),
                                 ('_merged',''),
                                 ('_reoriented',''),
                                 ('_unpadded',''),
                                 ('_nanto0',''),
                                 ('_masked','')]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

realign_workflow = Workflow(name="realign_workflow")
realign_workflow.base_dir = os.path.join(output_dir,'realign_workingdir')
realign_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'realign_crashdumps')}}
realign_workflow.connect([(getpet4D, reorient, [('pet4D','in_file')]),

                          (reorient, realign, [('out_file','in_files')]),

                          # remove low signal slices at image boundary
                          (realign, unpad, [('realigned_files','timeSeriesImgFile')]),

                          # get mask of within-field voxels after realignment
                          (unpad, nan_mask_4D, [('unpaddedImgFile','in_file')]),
                          #(realign, nan_mask_4D, [('realigned_files','in_file')]),
                          (nan_mask_4D, nan_mask, [('out_file','in_file')]),
                          # invert nan mask to get not-nan mask
                          (nan_mask, mulneg1, [('out_file','in_file')]),
                          (mulneg1, notnan_mask, [('out_file','in_file')]),

                          (unpad, nan_to_0, [('unpaddedImgFile','in_file')]),
                          #(realign, nan_to_0, [('realigned_files','in_file')]),

                          (nan_to_0, pet4D_masked, [('out_file','in_file')]),
                          (notnan_mask, pet4D_masked, [('out_file','in_file2')]),


                          (pet4D_masked, dynamic_mean, [('out_file','timeSeriesImgFile')]),
                          (getpettiming, dynamic_mean, [('pettiming','frameTimingCsvFile')]),

                          (dynamic_mean, smooth, [('meanImgFile','in_file')]),

                          # QC plots and snapshots
                          (getpettiming, realign_qc, [('pettiming','frameTimingCsvFile')]),
                          (pet4D_masked, realign_qc, [('out_file','petrealignedfile')]),
                          (realign, realign_qc, [('realignment_parameters','realignParamsFile')]),

                          # save outputs
                          # datasink paths are not working as intended. 8/24/2018
                          (dynamic_mean, datasink, [('meanImgFile',
                                                     'avg'+str(args.t_start_SUVR)+'to'+str(args.t_end_SUVR)+'min')]), # average (3D image)
                          (smooth, datasink, [('smoothed_file','avg'+str(args.t_start_SUVR)+'to'+str(args.t_end_SUVR)+'min.@smooth')]), # smoothed 80-100min average
                          (pet4D_masked, datasink, [('out_file','realigned')]), # realigned 80-100min time series (4D image)
                          (realign, datasink, [('realignment_parameters','realigned.@par')]), # realignment parameters
                          (realign_qc, datasink, [('realign_param_plot','QC'), # QC plots and snapshots
                                                  ('realigned_img_snap','QC.@snap')])
                         ])

realign_workflow.write_graph('realign.dot', graph2use='colored', simple_form=True)


# 3. MRI-PET COREGISTRATION
#
# Our goal is to perform image processing in native PET space to produce parametric images. We have structural MRIs that have already been preprocessed and anatomically labeled. To bring anatomical labels to PET space, we will perform coregistration of the PET and the MRI. We first register the MRI with skull onto the AV1451 mean image, use the brain mask obtained on the MRI to mask the AV1451 image, then repeat the registration using the skull-stripped MRI and the masked AV1451 image.
#
# * `av1451_mean`: Placeholder for the mean image computed at the end of the realign workflow.
# * `mriconvert_FSmri_init`, `mriconvert_FSmri`, `mriconvert_FSlabel`: We convert the `mgz` format to `nii`.
# * `reorientmri_init`, `reorientmri`, and `reorientlabel`:  Apply 90, 180, or 270 degree rotations as needed about the $x,y,z$ axes to match the MNI152 orientation.
# * `brainmask_init`: Brain mask in MRI space, computed by thresholding the label image.
# * `pet_to_mri_init`: We use the image with finer spatial resolution (MRI) as the reference, and the mean AV1451 image average as the moving image, to perform rigid alignment with normalized mutual information cost function, using FSL's FLIRT method. This initial registration uses MRI with skull.
# * `invertTransform_init`: Since we want anatomical labels in PET space, we invert the rigid transformation.
# * `brainmask_to_pet`: We apply the inverted transformation to the brain mask in MRI space to bring it into PET space.
# * `brainmask`: We dilate the transformed brain mask using a $3 \times 3 \times 3$ mm box kernel.
# * `av1451_mean_brainmasked`: We apply the dilated brain mask to the AV1451 mean image.
# * `pet_to_mri`: We repeat the PET to MRI registration, this time using the skull stripped MRI and brain masked PET image.
# * `invertTransform`: Since we want anatomical labels in PET space, we invert the rigid transformation.
# * `mri_to_pet` and `labels_to_pet`: We apply the inverted transformation to the MRI and anatomical labels to bring them to PET space.
# * `petmask`: An approximate brain mask in PET space, computed by thresholding the PET image. This is used to make sure that ROI averages do not include voxels where signal is very low (at image edges).
# * `multiply`: We apply the PET brain mask to the labels that have been transformed into PET space.

# a placeholder for the 80-100min average
av1451_mean = Node(interface=IdentityInterface(fields=['av1451_mean']), name="av1451_mean")

# Reorient MRI and label
reorientmri = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientmri")
reorientmriskull = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientmriskull")
reorientlabel = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientlabel")

# MRI coregistration, rigid, with mutual information
pet_to_mri = Node(interface=fsl.FLIRT(cost='mutualinfo', dof=6, searchr_x=[-30,30], searchr_y=[-30,30], searchr_z=[-30,30],
                                 coarse_search=15, fine_search=6), name="pet_to_mri")

invertTransform = Node(interface=fsl.ConvertXFM(invert_xfm=True), name="invertTransform")
mri_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True), name="mri_to_pet")
labels_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, interp='nearestneighbour'), name="labels_to_pet")

# this is not working as intended -- it simply binarizes the image. 8/24/2018
av1451_threshold_binarized = Node(interface=fsl.Threshold(nan2zeros=True,
                                                          thresh=args.signal_threshold,
                                                          args=' -bin'),
                                  name='av1451_threshold_binarized')
petmask = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul'), name='petmask')
labels_masked = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul'), name='labels_masked')

coreg_qc = Node(interface=coreg_snapshots(), name="coreg_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','coreg_wf')
datasink.inputs.substitutions = [('_roi',''),
                                 ('_merged',''),
                                 ('flirt','coreg'),
                                 ('_reoriented',''),
                                 ('_unpadded',''),
                                 ('_nanto0',''),
                                 ('_masked','')]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

coreg_workflow = Workflow(name="coreg_workflow")
coreg_workflow.base_dir = os.path.join(output_dir,'coreg_workingdir')
coreg_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'coreg_crashdumps')}}
coreg_workflow.connect([(getmri, reorientmri, [('mri','in_file')]),
                        (getmriskull, reorientmriskull, [('mriskull','in_file')]),
                        (getlabel, reorientlabel, [('label','in_file')]),

                        (av1451_mean, pet_to_mri, [('av1451_mean','in_file')]),
                        (reorientmriskull, pet_to_mri, [('out_file','reference')]),

                        (pet_to_mri, invertTransform, [('out_matrix_file','in_file')]),

                        (reorientmriskull, mri_to_pet, [('out_file','in_file')]),
                        (av1451_mean, mri_to_pet, [('av1451_mean','reference')]),
                        (invertTransform, mri_to_pet, [('out_file','in_matrix_file')]),

                        (reorientlabel, labels_to_pet, [('out_file','in_file')]),
                        (av1451_mean, labels_to_pet, [('av1451_mean','reference')]),
                        (invertTransform, labels_to_pet, [('out_file','in_matrix_file')]),

                        # Create a mask by thresholding av1451 mean
                        (av1451_mean, av1451_threshold_binarized, [('av1451_mean','in_file')]),
                        # Multiply this mask by the not-nan mask from realign
                        (av1451_threshold_binarized, petmask, [('out_file','in_file')]),
                        # Multiply label image by this combined mask
                        (labels_to_pet, labels_masked, [('out_file','in_file')]),
                        (petmask, labels_masked, [('out_file','in_file2')]),

                        (av1451_mean, coreg_qc, [('av1451_mean','petavgfile')]),
                        (mri_to_pet, coreg_qc, [('out_file','mriregfile')]),

                        # save outputs
                        (pet_to_mri, datasink, [('out_file','coreg_avg80to100min'),
                                                ('out_matrix_file','coreg_avg80to100min.@param')]),
                        (mri_to_pet, datasink, [('out_file','coreg_mri')]), # MRI registered onto PET
                        (labels_to_pet, datasink, [('out_file','coreg_labels')]), # anatomical labels on PET
                        (labels_masked, datasink, [('out_file','coreg_labels.@masked')]), # masked anatomical labels on PET
                        (coreg_qc, datasink, [('coreg_edges','QC'),
                                              ('coreg_overlay_sagittal','QC.@sag'),
                                              ('coreg_overlay_coronal','QC.@cor'),
                                              ('coreg_overlay_axial','QC.@ax')])
                       ])

coreg_workflow.write_graph('coreg.dot', graph2use='colored', simple_form=True)

av1451_workflow = Workflow(name="av1451_workflow")
av1451_workflow.base_dir = output_dir
av1451_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'av1451_crashdumps')}}

av1451_workflow.connect([# PET-to-MRI registration
                         (realign_workflow, coreg_workflow, [('smooth.smoothed_file','av1451_mean.av1451_mean'),
                                                             ('notnan_mask.out_file','petmask.in_file2')
                                                            ])
                        ])

# 4. SUIT CEREBELLAR ATLAS
# inverse transform the SUIT cerebellar atlas from MNI space to subject's PET (via subject's MRI)
# use affine registration for speed - this is not precise but SUITable for our purposes

# Quick registration to MNI template
mri_to_mni = Node(interface=fsl.FLIRT(dof=12,reference=template), name="mri_to_mni")
invertTransform_mni = Node(interface=fsl.ConvertXFM(invert_xfm=True), name="invertTransform_mni")

suit_to_mni = Node(interface=fsl.ApplyXFM(in_file=suit_labels, reference=template,
                                          apply_xfm=True, uses_qform=True),
                   name="suit_to_mni")
mergexfm = Node(interface=fsl.ConvertXFM(concat_xfm=True), name="mergexfm")
mergexfm2 = Node(interface=fsl.ConvertXFM(concat_xfm=True), name="mergexfm2")

suit_to_pet = Node(interface=fsl.ApplyXFM(in_file=suit_labels,
                                          apply_xfm=True,
                                          interp='nearestneighbour'), name="suit_to_pet")

# the following is based on Baker et al. 2017
inf_cerebellum = Node(interface=CombineROIs(ROI_groupings=list(inf_cerebellum_ROI_grouping.values())),
                      name="inf_cerebellum")
sup_cerebellum = Node(interface=CombineROIs(ROI_groupings=list(sup_cerebellum_ROI_grouping.values())),
                      name="sup_cerebellum")

# Gaussian smoothing of the masks
smooth_inf_cerebellum = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_inf_cerebellum")
smooth_sup_cerebellum = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_sup_cerebellum")

inf_minus_sup = Node(interface=fsl.ImageMaths(op_string=' -sub '), name="inf_minus_sup")
inf_greater_than_sup = Node(interface=fsl.ImageMaths(op_string=' -bin'), name="inf_greater_than_sup")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('_maths','_suvr'),
                                 ('_unpadded',''),
                                 ('flirt','mni'),
                                 ('_nanto0',''),
                                 ('_masked',''),
                                 ('_mul','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

SUIT_workflow = Workflow(name="SUIT_workflow")
SUIT_workflow.base_dir = os.path.join(output_dir,'SUIT_workingdir')
SUIT_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUIT_crashdumps')}}
SUIT_workflow.connect([(mri_to_mni, invertTransform_mni, [('out_matrix_file','in_file')]),
                       (suit_to_mni, mergexfm, [('out_matrix_file','in_file')]),
                       (invertTransform_mni, mergexfm, [('out_file','in_file2')]),

                       (mergexfm, mergexfm2, [('out_file','in_file')]),
                       (mergexfm2, suit_to_pet, [('out_file', 'in_matrix_file')]),

                       (suit_to_pet, inf_cerebellum, [('out_file','labelImgFile')]),
                       (suit_to_pet, sup_cerebellum, [('out_file','labelImgFile')]),

                       (inf_cerebellum, smooth_inf_cerebellum, [('roi4DMaskFile','in_file')]),
                       (sup_cerebellum, smooth_sup_cerebellum, [('roi4DMaskFile','in_file')]),

                       (smooth_inf_cerebellum, inf_minus_sup, [('smoothed_file','in_file')]),
                       (smooth_sup_cerebellum, inf_minus_sup, [('smoothed_file','in_file2')]),

                       (inf_minus_sup, inf_greater_than_sup, [('out_file','in_file')]),

                       (mri_to_mni, datasink, [('out_file','transformed_mri'),
                                               ('out_matrix_file','transformed_mri.@param')]),
                       (suit_to_pet, datasink, [('out_file','transformed_SUIT')])
                      ])

SUIT_workflow.write_graph('MNI.dot', graph2use='colored', simple_form=True)

av1451_workflow.connect([## MNI space normalization
                         (coreg_workflow, SUIT_workflow, [('reorientmri.out_file','mri_to_mni.in_file'),
                                                         ('invertTransform.out_file','mergexfm2.in_file2'),
                                                         ('av1451_mean.av1451_mean','suit_to_pet.reference')])
                        ])


# 5. LABELS
#
# There are two streams of processing we will pursue. First, we generate a conservative reference region definition:
# * `reference_region`: Combines the selected FS labels to generate a binary mask.
#
# Second, we generate the set of labels that will be used in partial volume correction. FS labels do not include a sulcal CSF label, but this is an important label for PVC. We approximate the sulcal CSF label as the rim around the brain. To this end, we dilate the brain mask, and subtract from it the original brain mask. We designate a label value of $-1$ to this rim, and include it with the ventricle and CSF ROI for PVC.
# * `brainmask`: Threshold the MUSE label image to get a binary brain mask.
# * `dilate`: Dilate the brain mask using a $3\times3\times3$ mm box kernel.
# * `difference`: Subtract dilated brain mask from the orginal mask to get the rim around the brain. This subtraction assigns a value of $-1$ to the rim.
# * `add`: We add the rim image to the FS label image. Since the FS label image has value $0$ where the rim image has non-zero values, the result is a label image that preserves all the MUSE labels and additionally has a "sulcal CSF" label with value $-1$.
# * `pvc_labels`: We combine the ROIs to generate a collection of binary masks. The result is a 4D volume (with all the binary 3D masks concatenated along 4th dimension). This 4D volume will be an input to the PVC methods.

# placeholder
label_unmasked = Node(interface=IdentityInterface(fields=['label_unmasked']), name="label_unmasked")
label_masked = Node(interface=IdentityInterface(fields=['label_masked']), name="label_masked")

reference_region = Node(interface=CombineROIs(ROI_groupings=list(reference_ROI_grouping.values())),
                        name="reference_region")
reference_region_inferior = Node(interface=fsl.ImageMaths(op_string=' -mul '), name="reference_region_inferior")

brainmask = Node(interface=fsl.ImageMaths(op_string=' -bin', suffix='_brainmask'), name='brainmask')
dilate = Node(interface=fsl.DilateImage(operation='max', kernel_shape='box', kernel_size=3), name='dilate')
difference = Node(interface=fsl.ImageMaths(op_string=' -sub ', suffix='_diff'), name='difference')
difference_masked = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul'), name='difference_masked')
add = Node(interface=fsl.ImageMaths(op_string=' -add ', suffix='_add'), name='add')
multiply = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul'), name='multiply')
pvc_labels = Node(interface=CombineROIs(ROI_groupings=list(pvc_ROI_groupings.values())), name="pvc_labels")

labels_qc = Node(interface=labels_snapshots(labelnames=list(pvc_ROI_groupings.keys())), name="labels_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','labels_wf')
datasink.inputs.substitutions = [('flirt','coreg'),
                                 ('_'+'{:d}'.format(len(reference_ROI_grouping))+'combinedROIs','_refRegion'),
                                 ('_'+'{:d}'.format(len(pvc_ROI_groupings))+'combinedROIs','_pvcLabels'),
                                 ('_mul',''),
                                 ('_add',''),
                                 ('_reoriented',''),
                                 ('_unpadded',''),
                                 ('_nanto0',''),
                                 ('_masked',''),
                                ]

labels_workflow = Workflow(name="labels_workflow")
labels_workflow.base_dir = os.path.join(output_dir,'labels_workingdir')
labels_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'labels_crashdumps')}}
labels_workflow.connect([# Assign a value of -1 to voxels surrounding the brain
                         # this is an approximation for sulcal CSF label
                         (label_unmasked, brainmask, [('label_unmasked','in_file')]),
                         (brainmask, dilate, [('out_file','in_file')]),
                         (brainmask, difference, [('out_file','in_file')]),
                         (dilate, difference, [('out_file','in_file2')]),
                         (difference, difference_masked, [('out_file','in_file')]),

                         (label_masked, add, [('label_masked','in_file')]),
                         (difference_masked, add,[('out_file','in_file2')]),
                         (add, pvc_labels, [('out_file','labelImgFile')]),

                         (label_masked, reference_region, [('label_masked', 'labelImgFile')]),
                         (reference_region, reference_region_inferior, [('roi4DMaskFile','in_file')]),

                         (pvc_labels, labels_qc, [('roi4DMaskFile','labelfile')]),

                         (reference_region_inferior, datasink, [('out_file','reference_region')]),
                         (pvc_labels, datasink, [('roi4DMaskFile','pvc_labels')]),
                         (labels_qc, datasink, [('label_snap','QC')])
                        ])

labels_workflow.write_graph('labels.dot', graph2use='colored', simple_form=True)

refReg_qc = Node(interface=refReg_snapshots(), name="refReg_qc")

av1451_workflow.connect([# Anatomical label manipulation
                         (coreg_workflow, labels_workflow, [('labels_to_pet.out_file','label_unmasked.label_unmasked'),
                                                            ('labels_masked.out_file','label_masked.label_masked'),
                                                            ('petmask.out_file','difference_masked.in_file2')]),
                         (SUIT_workflow, labels_workflow, [('inf_greater_than_sup.out_file','reference_region_inferior.in_file2')]),

                         # Reference region QC
                         (labels_workflow, refReg_qc, [('reference_region_inferior.out_file','maskfile')]),
                         (realign_workflow, refReg_qc, [('pet4D_masked.out_file','petrealignedfile'),
                                                        ('getpettiming.pettiming','frameTimingCsvFile')]),
                         (coreg_workflow, refReg_qc, [('av1451_mean.av1451_mean','petavgfile')]),
                         (refReg_qc, datasink, [('maskOverlay_axial','QC.@ax'),
                                                ('maskOverlay_coronal','QC.@cor'),
                                                ('maskOverlay_sagittal','QC.@sag'),
                                                ('mask_TAC','QC.@tac')])
                        ])

# WRITE OUT 4D TACs by ROI--

# 6a. SUVR IMAGE

# a placeholder for the 80-100min average
av1451_mean = Node(interface=IdentityInterface(fields=['av1451_mean']), name="av1451_mean")

ROImean = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean") # note that this is not a trimmed mean!
SUVR = Node(interface=fsl.ImageMaths(), name="SUVR")

ROImeans = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                   ROI_names=list(singleROIs.keys()),
                                                   additionalROIs=list(compositeROIs.values()),
                                                   additionalROI_names=list(compositeROIs.keys()),
                                                   stat='mean'),
                name="ROImeans")

ROI_Q3 = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                 ROI_names=list(singleROIs.keys()),
                                                 additionalROIs=list(compositeROIs.values()),
                                                 additionalROI_names=list(compositeROIs.keys()),
                                                 stat='Q3'),
              name="ROI_Q3")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','SUVR_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_flirt','_coreg'),
                                 ('_maths','_suvr'),
                                 ('_unpadded',''),
                                 ('_reoriented',''),
                                 ('_nanto0',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

SUVR_workflow = Workflow(name="SUVR_workflow")
SUVR_workflow.base_dir = os.path.join(output_dir,'SUVR_workingdir')
SUVR_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_crashdumps')}}
SUVR_workflow.connect([(av1451_mean, ROImean, [('av1451_mean','in_file')]),
                       (av1451_mean, SUVR, [('av1451_mean','in_file')]),
                       (ROImean, SUVR, [(('out_stat',to_div_string),'op_string')]),

                       (SUVR, ROImeans, [('out_file','imgFile')]),
                       (SUVR, ROI_Q3, [('out_file','imgFile')]),

                       (SUVR, datasink, [('out_file','SUVR')]),
                      ])

SUVR_workflow.write_graph('SUVR.dot', graph2use='colored', simple_form=True)

av1451_workflow.connect([# SUVR computation
                         (realign_workflow, SUVR_workflow, [('dynamic_mean.meanImgFile','av1451_mean.av1451_mean')]),
                         (labels_workflow, SUVR_workflow, [('reference_region_inferior.out_file','ROImean.mask_file'),
                                                           ('add.out_file','ROImeans.labelImgFile'),
                                                           ('add.out_file','ROI_Q3.labelImgFile'),
                                                          ])
                        ])

# 6b. SUVR IMAGE with PVC
if not args.no_pvc:
    # a placeholder for the 80-100min average
    av1451_mean = Node(interface=IdentityInterface(fields=['av1451_mean']), name="av1451_mean")

    pvc = Node(interface=petpvc.PETPVC(pvc='RBV', fwhm_x=args.psf_fwhm_x, fwhm_y=args.psf_fwhm_y, fwhm_z=args.psf_fwhm_z), name="pvc")

    ROImean = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean") # note that this is not a trimmed mean!
    SUVR = Node(interface=fsl.ImageMaths(), name="SUVR")

    ROImeans = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                 ROI_names=list(singleROIs.keys()),
                                                 additionalROIs=list(compositeROIs.values()),
                                                 additionalROI_names=list(compositeROIs.keys()),
                                                 stat='mean'),
                        name="ROImeans")

    ROI_Q3 = Node(interface=ROI_stats_to_spreadsheet(ROI_list=list(singleROIs.values()),
                                                 ROI_names=list(singleROIs.keys()),
                                                 additionalROIs=list(compositeROIs.values()),
                                                 additionalROI_names=list(compositeROIs.keys()),
                                                 stat='Q3'),
                        name="ROI_Q3")

    datasink = Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = os.path.join('output','SUVR_wf')
    datasink.inputs.substitutions = [('_merged',''),
                                     ('_flirt','_coreg'),
                                     ('_maths','_suvr'),
                                     ('_unpadded',''),
                                     ('_reoriented',''),
                                     ('_nanto0',''),
                                     ('_masked','')
                                    ]
    datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

    SUVR_pvc_workflow = Workflow(name="SUVR_pvc_workflow")
    SUVR_pvc_workflow.base_dir = os.path.join(output_dir,'SUVR_pvc_workingdir')
    SUVR_pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_pvc_crashdumps')}}
    SUVR_pvc_workflow.connect([(av1451_mean, pvc, [('av1451_mean','in_file')]),
                               (pvc, ROImean, [('out_file','in_file')]),
                               (pvc, SUVR, [('out_file','in_file')]),
                               (ROImean, SUVR, [(('out_stat',to_div_string),'op_string')]),

                               (SUVR, ROImeans, [('out_file','imgFile')]),
                               (SUVR, ROI_Q3, [('out_file','imgFile')]),

                               (SUVR, datasink, [('out_file','SUVR_pvc')])
                              ])

    SUVR_pvc_workflow.write_graph('SUVR_pvc.dot', graph2use='colored', simple_form=True)

    av1451_workflow.connect([# SUVR computation with pvc
                             (realign_workflow, SUVR_pvc_workflow, [('dynamic_mean.meanImgFile','av1451_mean.av1451_mean')]),
                             (labels_workflow, SUVR_pvc_workflow, [('reference_region_inferior.out_file','ROImean.mask_file'),
                                                                   ('pvc_labels.roi4DMaskFile','pvc.mask_file'),
                                                                   ('add.out_file','ROImeans.labelImgFile'),
                                                                   ('add.out_file','ROI_Q3.labelImgFile'),
                                                                  ])
                           ])



# 7. MNI SPACE
# Quick registration to MNI template
mri_to_mni = Node(interface=fsl.FLIRT(dof=12,reference=template), name="mri_to_mni")
mergexfm = Node(interface=fsl.ConvertXFM(concat_xfm=True), name="mergexfm")

transform_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, reference=template), name='transform_pet')
transform_suvr = transform_pet.clone(name='transform_suvr')

smooth_suvr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_suvr")

mask_suvr = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul', in_file2=template_brainmask),
                 name='mask_suvr')

suvr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5), name="suvr_qc")

mosaic_suvr = Node(interface=mosaic(vmin=0, vmax=2.5, cmap='jet', alpha=.5), name='mosaic_suvr')

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('_maths','_suvr'),
                                 ('_unpadded',''),
                                 ('flirt','mni'),
                                 ('_nanto0',''),
                                 ('_masked',''),
                                 ('_mul','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

MNI_workflow = Workflow(name="MNI_workflow")
MNI_workflow.base_dir = os.path.join(output_dir,'MNI_workingdir')
MNI_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'MNI_crashdumps')}}
MNI_workflow.connect([(mri_to_mni, mergexfm, [('out_matrix_file','in_file2')]),

                      (mergexfm, transform_pet, [('out_file', 'in_matrix_file')]),
                      (mergexfm, transform_suvr, [('out_file','in_matrix_file')]),

                      (transform_suvr, smooth_suvr, [('out_file','in_file')]),

                      (smooth_suvr, mask_suvr, [('smoothed_file','in_file')]),

                      (mri_to_mni, suvr_qc, [('out_file','bgimgfile')]),
                      (mask_suvr, suvr_qc, [('out_file','imgfile')]),

                      (mri_to_mni, mosaic_suvr, [('out_file','bgimgfile')]),
                      (mask_suvr, mosaic_suvr, [('out_file','imgfile')]),

                      (mri_to_mni, datasink, [('out_file','transformed_mri'),
                                              ('out_matrix_file','transformed_mri.@param')]),
                      (transform_pet, datasink, [('out_file','transformed_pet')]),
                      (transform_suvr, datasink, [('out_file','transformed_suvr')]),

                      (suvr_qc, datasink, [('triplanar','QC')]),
                      (mosaic_suvr, datasink, [('mosaic','QC.@mosaic_suvr')]),
                     ])

av1451_workflow.connect([## MNI space normalization
                         (coreg_workflow, MNI_workflow, [('reorientmri.out_file','mri_to_mni.in_file'),
                                                         ('pet_to_mri.out_matrix_file','mergexfm.in_file')]),
                         (realign_workflow, MNI_workflow, [('dynamic_mean.meanImgFile','transform_pet.in_file')]),
                         (SUVR_workflow, MNI_workflow, [('SUVR.out_file','transform_suvr.in_file')]),
                        ])

if not args.no_pvc:
    transform_suvr_pvc = transform_pet.clone(name='transform_suvr_pvc')
    smooth_suvr_pvc = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_suvr_pvc")
    mask_suvr_pvc = mask_suvr.clone(name="mask_suvr_pvc")
    suvr_pvc_qc = suvr_qc.clone(name="suvr_pvc_qc")
    mosaic_suvr_pvc = mosaic_suvr.clone(name="mosaic_suvr_pvc")

    MNI_workflow.connect([(mergexfm, transform_suvr_pvc, [('out_file','in_matrix_file')]),

                          (transform_suvr_pvc, smooth_suvr_pvc, [('out_file','in_file')]),

                          (smooth_suvr_pvc, mask_suvr_pvc, [('smoothed_file','in_file')]),

                          (mri_to_mni, suvr_pvc_qc, [('out_file','bgimgfile')]),
                          (mask_suvr_pvc, suvr_pvc_qc, [('out_file','imgfile')]),

                          (mri_to_mni, mosaic_suvr_pvc, [('out_file','bgimgfile')]),
                          (mask_suvr_pvc, mosaic_suvr_pvc, [('out_file','imgfile')]),

                          (transform_suvr_pvc, datasink, [('out_file','transformed_suvr_pvc')]),

                          (suvr_pvc_qc, datasink, [('triplanar','QC.@suvr_pvc')]),
                          (mosaic_suvr_pvc, datasink, [('mosaic','QC.@mosaic_suvr_pvc')]),
                         ])

    av1451_workflow.connect([(SUVR_pvc_workflow, MNI_workflow, [('SUVR.out_file','transform_suvr_pvc.in_file')]),
                            ])

MNI_workflow.write_graph('MNI.dot', graph2use='colored', simple_form=True)


# 7. MNI SPACE -- deformable alignment onto study-specific template
# placeholder
pet = Node(interface=IdentityInterface(fields=['pet']), name="pet")

# convert the affine transform from PET to MRI obtained via FSL to ANTs format
convert2itk = Node(interface=c3.C3dAffineTool(fsl2ras=True, itk_transform=True), name="convert2itk")

# merge affine transform to MRI with composite warp to study-specific template in MNI space
# first should be composite
# second should be pet-to-mri
merge_list = Node(Merge(2), name='merge_list')
merge_list.inputs.in1 = args.mnitransform

warp_pet = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni),
                name="warp_pet")
warp_suvr = warp_pet.clone(name='warp_suvr')

mask_pet = Node(interface=fsl.ImageMaths(op_string=' -mul ', suffix='_mul', in_file2=blsa_template_mni_brainmask), name='mask_pet')
mask_suvr = mask_pet.clone(name='mask_suvr')

smooth_suvr = Node(interface=fsl.Smooth(fwhm=args.smooth_fwhm), name="smooth_suvr")

# Triplanar snapshots
suvr_qc = Node(interface=triplanar_snapshots(bgimgfile = blsa_template_mni,
                                             alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=2.5),
                  name="suvr_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_deform_wf')
datasink.inputs.substitutions = [('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('flirt','mni'),
                                 ('_0000',''),
                                 ('_unpadded',''),
                                 ('_maths','_suvr'),
                                 ('_masked',''),
                                 ('_nanto0',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

MNI_deform_workflow = Workflow(name="MNI_deform_workflow")
MNI_deform_workflow.base_dir = os.path.join(output_dir,'MNI_deform_workingdir')
MNI_deform_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'MNI_deform_crashdumps')}}
MNI_deform_workflow.connect([(pet, convert2itk, [('pet','source_file')]),

                             (convert2itk, merge_list, [('itk_transform','in2')]),

                             (pet, warp_pet, [('pet','input_image')]),
                             (merge_list, warp_pet, [('out','transforms')]),
                             (merge_list, warp_suvr, [('out','transforms')]),

                             (warp_suvr, smooth_suvr, [('output_image','in_file')]),

                             (warp_pet, mask_pet, [('output_image','in_file')]),
                             (warp_suvr, mask_suvr, [('output_image','in_file')]),

                             (smooth_suvr, suvr_qc, [('smoothed_file','imgfile')]),

                             (warp_pet, datasink, [('output_image','warped_pet')]),
                             (warp_suvr, datasink, [('output_image','warped_suvr')]),

                             (suvr_qc, datasink, [('triplanar','QC')]),
                            ])

av1451_workflow.connect([# Deformable spatial normalization
                         (coreg_workflow, MNI_deform_workflow, [('pet_to_mri.out_matrix_file','convert2itk.transform_file'),
                                                                ('reorientmri.out_file','convert2itk.reference_file')]),
                         (realign_workflow, MNI_deform_workflow, [('dynamic_mean.meanImgFile','pet.pet')]),
                         (SUVR_workflow, MNI_deform_workflow, [('SUVR.out_file','warp_suvr.input_image')]),
                        ])

if not args.no_pvc:
    warp_suvr_pvc = warp_pet.clone(name='warp_suvr_pvc')
    mask_suvr_pvc = mask_pet.clone(name='mask_suvr_pvc')
    smooth_suvr_pvc = smooth_suvr.clone(name='smooth_suvr_pvc')
    suvr_pvc_qc = suvr_qc.clone(name='suvr_pvc_qc')

    MNI_deform_workflow.connect([(merge_list, warp_suvr_pvc, [('out','transforms')]),
                                 (warp_suvr_pvc, smooth_suvr_pvc, [('output_image','in_file')]),
                                 (warp_suvr_pvc, mask_suvr_pvc, [('output_image','in_file')]),
                                 (smooth_suvr_pvc, suvr_pvc_qc, [('smoothed_file','imgfile')]),
                                 (warp_suvr_pvc, datasink, [('output_image','warped_suvr_pvc')]),
                                 (suvr_pvc_qc, datasink, [('triplanar','QC.@suvr_pvc')])
                                ])

    av1451_workflow.connect([(SUVR_pvc_workflow, MNI_deform_workflow, [('SUVR.out_file','warp_suvr_pvc.input_image')])])

MNI_deform_workflow.write_graph('MNI_deform.dot', graph2use='colored', simple_form=True)

av1451_workflow.write_graph('av1451_MUSE.dot', graph2use='colored', simple_form=True)
result = av1451_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
