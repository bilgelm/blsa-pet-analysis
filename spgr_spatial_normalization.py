# import packages
import os
import nipype.interfaces.io as nio
from nipype.interfaces import ants, fsl
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Function, IdentityInterface, Merge
from nipype_misc_funcs import get_value, reverse_list # user-defined functions
from nipype_snapshot_funcs import triplanar_snapshots
from nipype import config, logging
import pandas as pd
import numpy as np
import math
config.enable_debug_mode()
logging.update_logging(config)

# number of parallel processes
n_procs = 8

# directory to store the workflow results
output_dir = '/output/mri'

# prefix for the data collection site
sitePrefix = 'BLSA'

template_dir = '/templates/UPENN_BLSA_templates'

# study-specific template in template space
blsa_template = os.path.join(template_dir,'BLSA_avg_temp_space',
                'BLSA_SPGR_AllBaselines_Age60-80_Random100_averagetemplate_final_short_reoriented.nii.gz')

# fixed transformation from template space to MNI space (affine followed by deformable warp)
affine2mni = os.path.join(template_dir,'BLSA_2_MNI',
             'BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152Affine.txt')
deform2mni = os.path.join(template_dir,'BLSA_2_MNI',
             'BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152Warp.nii.gz')

# study-specific template in MNI space (note that this is not used as a target image in any registration - used only as reference)
blsa_template_mni = os.path.join(template_dir,'BLSA_2_MNI',
                   'BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152_reoriented.nii.gz')

# spreadsheet with the following columns: blsaid, blsavi, musemripath, muselabelpath
organization_spreadsheet = '/input/PETstatus_09Nov2018.xlsx'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Read in the organization spreadsheet
NAN_VALUES = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan',''] # do not include NA as a null value as it is a valid EMSID
data_table = pd.read_excel(organization_spreadsheet, keep_default_na=False, na_values=NAN_VALUES)
required_cols = ['blsaid','blsavi','musemripath']
for col in required_cols:
    if not col in data_table.columns:
        sys.exit('Required column '+col+' is not present in the data organization spreadsheet '+organization_spreadsheet+'!')

# Find all visits with PiB-PET (excluding incomplete/interrupted PiB scans) and MUSE labels
data_table = data_table[required_cols].dropna(axis=0, how='any').reset_index()

# Identify MUSE 1.5T SPGRs (scanners 1, 2, 3)
mri15T = np.zeros(data_table.shape[0], dtype=bool)
for i, row in data_table.iterrows():
    if int(os.path.basename(row['musemripath'])[15:17]) in [1,2,3]:
        mri15T[i] = True

data_table = data_table[mri15T]

subjID = data_table['blsaid'].values.tolist()
visNo = data_table['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
blsavi_decimal = [ str(x).split('.')[1] for x in visNo ]
idvi_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,blsavi_integer,blsavi_decimal)]

musemri_list = data_table['musemripath'].values.tolist()
musemri_dict = dict(zip(idvi_list, musemri_list))

# Identify baseline MUSE 1.5T MRIs
data_table_baseline = data_table.groupby('blsaid').first().reset_index()

subjID = data_table_baseline['blsaid'].values.tolist()
visNo = data_table_baseline['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
blsavi_decimal = [ str(x).split('.')[1] for x in visNo ]
idvi_baseline_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,blsavi_integer,blsavi_decimal)]

# Identify non-baseline MUSE 1.5T MRIs
isnotbaseline = [x not in data_table_baseline['musemripath'].values.tolist() for x in data_table['musemripath'].values.tolist()]
data_table_nonbaseline = data_table[isnotbaseline]

data_table_baseline.rename(columns={'blsavi': 'baseblsavi'}, inplace=True)
# Merge the conversion and data spreadsheets
data_table_nonbaseline = pd.merge(data_table_nonbaseline,data_table_baseline[['blsaid','baseblsavi']], how='inner', on=['blsaid'])

subjID = data_table_nonbaseline['blsaid'].values.tolist()
visNo = data_table_nonbaseline['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
blsavi_decimal = [ str(x).split('.')[1] for x in visNo ]
idvi_nonbaseline_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,blsavi_integer,blsavi_decimal)]

# Map each nonbaseline to its baseline (add column to data_table_nonbaseline)
basevisNo = data_table_nonbaseline['baseblsavi'].values.tolist()
baseblsavi_integer = [ math.floor(x) for x in basevisNo ]
baseblsavi_decimal = [ str(x).split('.')[1] for x in basevisNo ]
idvi_basenonbaseline_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,baseblsavi_integer,baseblsavi_decimal)]
nonbaseline_to_baseline = dict(zip(idvi_nonbaseline_list, idvi_basenonbaseline_list))

# Save spreadsheet of nonbaseline-to-baseline mappings
data_table_nonbaseline['idvi_basenonbaseline_list'] = idvi_basenonbaseline_list
data_table_nonbaseline.to_csv(os.path.join(output_dir,'baseline_mapping_spgr.csv'),index=False)

## ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ INPUTS ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ##
# placeholder Node to enable iteration over scans
infosource_baseline = Node(interface=IdentityInterface(fields=['idvi']), name="infosource_baseline")
infosource_baseline.iterables = ('idvi', idvi_baseline_list)

# get full path to MRI corresponding to idvi
getmusemri_baseline = Node(Function(input_names=['key','dict'],output_names=['musemri'],function=get_value),
                           name='getmusemri_baseline')
getmusemri_baseline.inputs.dict = musemri_dict

# Step 1: Spatial normalization of baseline 1.5T MRI onto study-specific template

# Reorient: this simply applies 90, 180, or 270 degree rotations about each axis to make the image orientation
# the same as the FSL standard
reorient = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorient")

# Use antsRegistration to compute registration between subject's baseline 1.5T MRI and study-specific template
antsreg = Node(ants.Registration(args='--float',
                                 collapse_output_transforms=True,
                                 fixed_image=blsa_template,
                                 initial_moving_transform_com=True,
                                 num_threads=1,
                                 output_inverse_warped_image=True,
                                 output_warped_image=True,
                                 smoothing_sigmas=[[3, 2, 1, 0]]*3,
                                 sigma_units=['vox']*3,
                                 transforms=['Rigid', 'Affine', 'SyN'],
                                 terminal_output='file',
                                 winsorize_lower_quantile=0.005,
                                 winsorize_upper_quantile=0.995,
                                 convergence_threshold=[1e-06],
                                 convergence_window_size=[10],
                                 metric=['MI', 'MI', 'CC'],
                                 metric_weight=[1.0]*3,
                                 number_of_iterations=[[1000, 500, 250, 100],
                                                       [1000, 500, 250, 100],
                                                       [100, 70, 50, 20]],
                                 radius_or_number_of_bins=[32, 32, 4],
                                 sampling_percentage=[0.25, 0.25, 1],
                                 sampling_strategy=['Regular',
                                                    'Regular',
                                                    'None'],
                                 shrink_factors=[[8, 4, 2, 1]]*3,
                                 transform_parameters=[(0.1,),
                                                       (0.1,),
                                                       (0.1, 3.0, 0.0)],
                                 use_histogram_matching=True,
                                 write_composite_transform=False),name='antsreg')

deform_workflow = Workflow(name="deform_workflow")
deform_workflow.base_dir = os.path.join(output_dir,'deform_workingdir')
deform_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'deform_crashdumps')}}
deform_workflow.connect([(getmusemri_baseline, reorient, [('musemri','in_file')]),
                         (reorient, antsreg, [('out_file','moving_image')])
                        ])


# Step 2: Concatenate transformations and bring baseline 1.5T MRI into MNI space

# We have to reverse the "forward transforms" output of antsreg Node so that affine and deformable warps are applied
# in the correct order
reverselist = Node(Function(input_names=['l'],output_names=['revlist'],function=reverse_list),
                   name='reverselist')

# We combine the affine+warp to study-specific template, for possible later use
compositetransform_to_template = Node(interface=ants.ApplyTransforms(reference_image=blsa_template,
                                                                     print_out_composite_warp_file=True,
                                                                     output_image='compositetransform_to_template.nii.gz'),
                                      name="compositetransform_to_template")

# We concatenate the transforms to template space with transforms to MNI space
merge_list = Node(Merge(3), name='merge_list')
merge_list.inputs.in1 = deform2mni
merge_list.inputs.in2 = affine2mni

# Apply the concatenated transform to baseline MRI to bring it into MNI space
#antswarp = Node(ants.WarpImageMultiTransform(reference_image=blsa_template_mni), name='antswarp')
applytransform_to_mni = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni),
                             name="applytransform_to_mni")
compositetransform_to_mni = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni,
                                                                print_out_composite_warp_file=True,
                                                                output_image='compositetransform_to_mni.nii.gz'),
                                      name="compositetransform_to_mni")

# Triplanar snapshots in MNI space
mni_qc = Node(interface=triplanar_snapshots(bgimgfile=blsa_template_mni,
                                            alpha=.4,
                                            cmap='viridis',
                                            x=81, y=93, z=77,
                                            vmin=100.0, vmax=600.0), name="mni_qc")

warp_workflow = Workflow(name='warp_workflow')
warp_workflow.base_dir = os.path.join(output_dir,'warp_workingdir')
warp_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'warp_crashdumps')}}
warp_workflow.connect([(reverselist, compositetransform_to_template,[('revlist','transforms')]),

                       (reverselist, merge_list, [('revlist','in3')]),
                       #(merge_list, antswarp, [('out','transformation_series')]),
                       (merge_list, applytransform_to_mni, [('out','transforms')]),
                       (merge_list, compositetransform_to_mni, [('out','transforms')]),

                       (applytransform_to_mni, mni_qc, [('output_image','imgfile')])
                      ])

# Save outputs of interest: reoriented baseline 1.5T MRI, transform to template, baseline 1.5T MRI in template space,
#                           baseline 1.5T MRI in MNI space
datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = 'output'
datasink.inputs.substitutions = [('_idvi_',sitePrefix+'_')]

# Connect input, step 1, and step 2
mri_baseline_workflow = Workflow(name="mri_baseline_workflow")
mri_baseline_workflow.base_dir = output_dir
mri_baseline_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'mri_baseline_crashdumps')}}
mri_baseline_workflow.connect([(infosource_baseline, deform_workflow, [('idvi','getmusemri_baseline.key')]),

                               (deform_workflow, warp_workflow, [('antsreg.forward_transforms','reverselist.l')]),
                               (deform_workflow, warp_workflow, [('reorient.out_file','compositetransform_to_template.input_image')]),
                               (deform_workflow, warp_workflow, [('reorient.out_file','applytransform_to_mni.input_image')]),
                               (deform_workflow, warp_workflow, [('reorient.out_file','compositetransform_to_mni.input_image')]),

                               (deform_workflow, datasink, [('reorient.out_file','baseline_space')]),
                               (deform_workflow, datasink, [('antsreg.forward_transforms','template_space')]),
                               (deform_workflow, datasink, [('antsreg.warped_image','template_space.@warped')]),
                               (warp_workflow, datasink, [('compositetransform_to_template.output_image','template_space.@compositetransform')]),
                               (warp_workflow, datasink, [('applytransform_to_mni.output_image','MNI_space')]),
                               (warp_workflow, datasink, [('compositetransform_to_mni.output_image','MNI_space.@compositetransform')]),

                               (warp_workflow, datasink, [('mni_qc.triplanar','QC')])
                              ])

mri_baseline_workflow.write_graph('mri_baseline.dot', graph2use='colored', simple_form=True)

# Run workflow
mri_baseline_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})

# -------------------------- FOLLOW-UP 1.5T MRIs -----------------------------#
# placeholder Node to enable iteration over scans
infosource = Node(interface=IdentityInterface(fields=['idvi']), name="infosource")
infosource.iterables = ('idvi', idvi_nonbaseline_list)

# get full path to MRI corresponding to idvi
getmusemri = Node(Function(input_names=['key','dict'],output_names=['musemri'],function=get_value),
                           name='getmusemri')
getmusemri.inputs.dict = musemri_dict

# get ID of baseline MRI corresponding to idvi
getID_baseline = Node(Function(input_names=['key','dict'],output_names=['baselineID'],function=get_value),
                      name='getID_baseline')
getID_baseline.inputs.dict = nonbaseline_to_baseline

# Step 3: Deformably register follow-up 1.5T MRI to baseline 1.5T MRI

# Get the affine matrix and warp between baseline MRI space and study-specific template space
templates = {'composite': os.path.join(output_dir,'output','template_space',
                                       sitePrefix+'_{idvi}','compositetransform_to_template.nii.gz'),
             'reorientedbaseline': os.path.join(output_dir,'output','baseline_space',
                                                sitePrefix+'_{idvi}',sitePrefix+'_*_reoriented.nii')}
selectfiles = Node(nio.SelectFiles(templates), name="selectfiles")

# Reorient
reorient = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorient")

# Compute within-subject registration onto baseline 1.5T MRI
antslongreg = Node(ants.Registration(args='--float',
                                     collapse_output_transforms=True,
                                     initial_moving_transform_com=True,
                                     num_threads=8,
                                     output_inverse_warped_image=True,
                                     output_warped_image=True,
                                     smoothing_sigmas=[[3, 2, 1, 0]]*3,
                                     sigma_units=['vox']*3,
                                     transforms=['Rigid', 'Affine', 'SyN'],
                                     terminal_output='file',
                                     winsorize_lower_quantile=0.005,
                                     winsorize_upper_quantile=0.995,
                                     convergence_threshold=[1e-06],
                                     convergence_window_size=[10],
                                     metric=['MI', 'MI', 'CC'],
                                     metric_weight=[1.0]*3,
                                     number_of_iterations=[[1000, 500, 250, 100],
                                                           [1000, 500, 250, 100],
                                                           [100, 70, 50, 20]],
                                     radius_or_number_of_bins=[32, 32, 4],
                                     sampling_percentage=[0.25, 0.25, 1],
                                     sampling_strategy=['Regular',
                                                        'Regular',
                                                        'None'],
                                     shrink_factors=[[8, 4, 2, 1]]*3,
                                     transform_parameters=[(0.1,),
                                                           (0.1,),
                                                           (0.1, 3.0, 0.0)],
                                     use_histogram_matching=True,
                                     write_composite_transform=False),name='antslongreg')

# Step 4: Concatenate transforms and apply to follow-up 1.5T MRI to bring it into template space and into MNI space
reverselist = Node(Function(input_names=['l'],output_names=['revlist'],function=reverse_list),
                   name='reverselist')

compositetransform_to_baseline = Node(interface=ants.ApplyTransforms(print_out_composite_warp_file=True,
                                                                     output_image='compositetransform_to_baseline.nii.gz'),
                                      name="compositetransform_to_baseline")

merge_list = Node(Merge(2), name='merge_list')

applytransform_to_template = Node(interface=ants.ApplyTransforms(reference_image=blsa_template),
                                  name="applytransform_to_template")
compositetransform_to_template = Node(interface=ants.ApplyTransforms(reference_image=blsa_template,
                                                                     print_out_composite_warp_file=True,
                                                                     output_image='compositetransform_to_template.nii.gz'),
                                      name="compositetransform_to_template")

merge_list2 = Node(Merge(4), name='merge_list2')
merge_list2.inputs.in1 = deform2mni
merge_list2.inputs.in2 = affine2mni

applytransform_to_mni = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni),
                             name="applytransform_to_mni")
compositetransform_to_mni = Node(interface=ants.ApplyTransforms(reference_image=blsa_template_mni,
                                                                print_out_composite_warp_file=True,
                                                                output_image='compositetransform_to_mni.nii.gz'),
                                 name="compositetransform_to_mni")

# Triplanar snapshots in MNI space
mni_qc = Node(interface=triplanar_snapshots(bgimgfile=blsa_template_mni,
                                            alpha=.4,
                                            cmap='viridis',
                                            x=81, y=93, z=77,
                                            vmin=100.0, vmax=600.0), name="mni_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = 'output'
datasink.inputs.substitutions = [('_idvi_',sitePrefix+'_')]

longreg_workflow = Workflow(name="longreg_workflow")
longreg_workflow.base_dir = output_dir
longreg_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'longregcrashdumps')}}
longreg_workflow.connect([(infosource, getmusemri, [('idvi','key')]),
                          (getmusemri, reorient, [('musemri','in_file')]),
                          (reorient, antslongreg, [('out_file','moving_image')]),

                          (infosource, getID_baseline, [('idvi','key')]),
                          (getID_baseline, selectfiles, [('baselineID','idvi')]),
                          (selectfiles, antslongreg, [('reorientedbaseline','fixed_image')]),

                          # Compute composite transform to baseline
                          (antslongreg, reverselist, [('forward_transforms','l')]),
                          (reorient, compositetransform_to_baseline, [('out_file','input_image')]),
                          (selectfiles, compositetransform_to_baseline, [('reorientedbaseline','reference_image')]),
                          (reverselist, compositetransform_to_baseline, [('revlist','transforms')]),

                          # Bring longitudinal MRI onto BLSA template
                          (reorient, applytransform_to_template, [('out_file','input_image')]),
                          (selectfiles, merge_list, [('composite','in1')]),
                          (reverselist, merge_list, [('revlist','in2')]),
                          (merge_list, applytransform_to_template, [('out','transforms')]),

                          # Composite transform bringing longitudinal MRI onto BLSA template
                          (reorient, compositetransform_to_template, [('out_file','input_image')]),
                          (merge_list, compositetransform_to_template, [('out','transforms')]),

                          # Bring longitudinal MRI to MNI space
                          (reorient, applytransform_to_mni, [('out_file','input_image')]),
                          (selectfiles, merge_list2, [('composite','in3')]),
                          (reverselist, merge_list2, [('revlist','in4')]),
                          (merge_list2, applytransform_to_mni, [('out','transforms')]),

                          # Composite transform bringing longitudinal MRI to MNI space
                          (reorient, compositetransform_to_mni, [('out_file','input_image')]),
                          (merge_list2, compositetransform_to_mni, [('out','transforms')]),

                          (applytransform_to_mni, mni_qc, [('output_image','imgfile')]),

                          (antslongreg, datasink, [('warped_image','baseline_space')]),
                          (compositetransform_to_baseline, datasink, [('output_image','baseline_space.@compositetransform')]),
                          (applytransform_to_template, datasink, [('output_image','template_space')]),
                          (compositetransform_to_template, datasink, [('output_image','template_space.@compositetransform')]),
                          (applytransform_to_mni, datasink, [('output_image','MNI_space')]),
                          (compositetransform_to_mni, datasink, [('output_image','MNI_space.@compositetransform')]),
                          (mni_qc, datasink, [('triplanar','QC')])
                         ])

longreg_workflow.write_graph('longreg.dot', graph2use='colored', simple_form=True)

# There is no longitudinal registration to run for SPGRs
#longreg_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})


# get follow-up PET visits that are associated with the same MRI as the baseline
# for these visits, create an output directory that contains a link to the corresponding composite transform
idvi_remainder_list = sorted(list(set(idvi_list) - set(idvi_baseline_list + idvi_nonbaseline_list)))
isremainder = [x in idvi_remainder_list for x in idvi_list]
data_table_remainder = data_table[isremainder]

# Merge the conversion and data spreadsheets
data_table_remainder = pd.merge(data_table_remainder,data_table_baseline[['blsaid','baseblsavi']], how='inner', on=['blsaid'])

# Map each remainder to its baseline (add column to data_table_remainder)
subjID = data_table_remainder['blsaid'].values.tolist()
basevisNo = data_table_remainder['baseblsavi'].values.tolist()
baseblsavi_integer = [ math.floor(x) for x in basevisNo ]
baseblsavi_decimal = [ str(x).split('.')[1] for x in basevisNo ]
idvi_baseremainder_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,baseblsavi_integer,baseblsavi_decimal)]
remainder_to_baseline = dict(zip(idvi_remainder_list, idvi_baseremainder_list))

for idvi in idvi_remainder_list:
    # Composite transform to template space
    if not os.path.isdir(os.path.join(output_dir,'output','template_space',sitePrefix+'_'+idvi)):
        os.mkdir(os.path.join(output_dir,'output','template_space',sitePrefix+'_'+idvi))
    else:
        print(os.path.join(output_dir,'output','template_space',sitePrefix+'_'+idvi) + ' already exists!')

    compositetransform_file = os.path.join(output_dir,'output','template_space',sitePrefix+'_'+idvi,'compositetransform_to_template.nii.gz')
    if os.path.islink(compositetransform_file):
        os.remove(compositetransform_file)
    if not os.path.isfile(compositetransform_file):
        os.symlink(os.path.join(output_dir,'output','template_space',sitePrefix+'_'+remainder_to_baseline[idvi],'compositetransform_to_template.nii.gz'),
                   compositetransform_file)
    else:
        print(compositetransform_file + ' already exists!')

    # Composite transform to MNI space
    if not os.path.isdir(os.path.join(output_dir,'output','MNI_space',sitePrefix+'_'+idvi)):
        os.mkdir(os.path.join(output_dir,'output','MNI_space',sitePrefix+'_'+idvi))
    else:
        print(os.path.join(output_dir,'output','MNI_space',sitePrefix+'_'+idvi) + ' already exists!')

    compositetransform_file = os.path.join(output_dir,'output','MNI_space',sitePrefix+'_'+idvi,'compositetransform_to_mni.nii.gz')
    if os.path.islink(compositetransform_file):
        os.remove(compositetransform_file)
    if (not os.path.isfile(compositetransform_file)):
        os.symlink(os.path.join(output_dir,'output','MNI_space',sitePrefix+'_'+remainder_to_baseline[idvi],'compositetransform_to_mni.nii.gz'),
                   compositetransform_file)
    else:
        print(compositetransform_file + ' already exists!')
