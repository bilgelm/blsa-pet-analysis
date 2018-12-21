# running on nialpc-nilab10
# FIRST EXECUTE:
#   source activate petpipeline_05152018

# import packages
import math, os, sys, logging
import pandas as pd

# nipype
import nipype.interfaces.io as nio
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import Function, IdentityInterface
from av1451_wrapper import get_value, concat_dirs, PipelineWrapper # av1451_wrapper.py must be in current working directory (or otherwise discoverable by python)
from nipype import config, logging
from nipype_misc_funcs import ConcatenateSpreadsheets
config.enable_debug_mode()
logging.update_logging(config)

# number of parallel processes
n_procs = 24

# directory to store the workflow results
output_dir = '/output/av1451/'

# prefix for the data collection site
sitePrefix = 'BLSA'

# spreadsheet with the following columns: blsaid, blsavi, AV1451path_c3, AV1451timingpath'
organization_spreadsheet = '/input/PETstatus_09Nov2018.xlsx'

# columns required in the spreadsheet
required_cols = ['blsaid','blsavi','AV1451path_c3','AV1451timingpath','musemripath','musemriskullpath','muselabelpath']

# values to be treated as missing in the spreadsheet - do not include NA as a null value as it is a valid EMSID
NAN_VALUES = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan','']


# AV1451 processing parameters
# 80 to 100-min mean, used for SUVR computation
startTime_80to100min = 80
endTime_80to100min = 100

# number of slices to remove from each side of the PET image
unpad_size = 7

# Ignore mean AV1451 signal below this threshold when computing ROI averages
signal_threshold = 100

# PET scanner PSF FWHM (in mm), used for PVC
psf_fwhm_x = 2.5
psf_fwhm_y = 2.5
psf_fwhm_z = 2.5

# Smoothing parameter (in mm) used for SUVR
smooth_fwhm = 4

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


# Read in the organization spreadsheet
data_table = pd.read_excel(organization_spreadsheet, keep_default_na=False, na_values=NAN_VALUES)

for col in required_cols:
    if not col in data_table.columns:
        sys.exit('Required column '+col+' is not present in the data organization spreadsheet '+organization_spreadsheet+'!')

data_table = data_table[data_table.excludeAV1451 != 'y']

# Find all visits with AV1451-PET (excluding incomplete/interrupted AV1451 scans) and MUSE labels
data_table = data_table[required_cols].dropna(axis=0, how='any')

musemri_list = data_table['musemripath'].values.tolist()
musemriskull_list = data_table['musemriskullpath'].values.tolist()
muselabel_list = data_table['muselabelpath'].values.tolist()
av1451_list = data_table['AV1451path_c3'].values.tolist()
av1451timing_list = data_table['AV1451timingpath'].values.tolist()

subjID = data_table['blsaid'].values.tolist()
visNo = data_table['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
blsavi_decimal = [ str(x).split('.')[1] for x in visNo ]

idvi_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,blsavi_integer,blsavi_decimal)]

av1451_dict = dict(zip(idvi_list, av1451_list))
av1451timing_dict = dict(zip(idvi_list, av1451timing_list))
musemri_dict = dict(zip(idvi_list, musemri_list))
musemriskull_dict = dict(zip(idvi_list, musemriskull_list))
muselabel_dict = dict(zip(idvi_list, muselabel_list))


# 1. INPUTS
#
# We set up the nipype Nodes that will act as the inputs to our Workflows. The `infosource` Node allows for iterating over scan IDs.

# placeholder Node to enable iteration over scans
infosource = Node(interface=IdentityInterface(fields=['idvi']), name="infosource")
infosource.iterables = ('idvi', idvi_list)

# get full path to AV1451 scan corresponding to idvi from spreadsheet
getav1451 = Node(Function(input_names=['key','dict'],output_names=['av1451'],function=get_value), name='getav1451')
getav1451.inputs.dict = av1451_dict

# get full path to the txt file listing the duration of each AV1451 time frame
#  - number of rows must be the same as the number of AV1451 time frames, with each row listing the time in minutes
getav1451timing = Node(Function(input_names=['key','dict'],output_names=['av1451timing'],function=get_value),
                       name='getav1451timing')
getav1451timing.inputs.dict = av1451timing_dict

# get full path to MRI with skull corresponding to idvi from spreadsheet,
#  in same space as MUSE labels
getmusemri = Node(Function(input_names=['key','dict'],
                                output_names=['musemri'],
                                function=get_value),
                       name='getmusemri')
getmusemri.inputs.dict = musemri_dict

# get full path to MRI with skull corresponding to idvi from spreadsheet,
#  in same space as MUSE labels
getmusemriskull = Node(Function(input_names=['key','dict'],
                                output_names=['musemriskull'],
                                function=get_value),
                       name='getmusemriskull')
getmusemriskull.inputs.dict = musemriskull_dict

# get full path to MUSE label image corresponding to idvi from spreadsheet,
#  in same space as MRI
getmuselabel = Node(Function(input_names=['key','dict'],
                             output_names=['muselabel'],
                             function=get_value),
                    name='getmuselabel')
getmuselabel.inputs.dict = muselabel_dict

# get the composite transform that corresponds to the MRI "concurrent" with PET
compositetransform_dir = '/output/mri/output/MNI_space'

# Get the affine matrix and warp between MRI space and study-specific template space
templates = {'composite': os.path.join(compositetransform_dir,
                                       sitePrefix+'_{idvi}','compositetransform_to_mni.nii.gz')}
selectfiles = Node(nio.SelectFiles(templates), name="selectfiles")

SUVR_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_SUVR'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='SUVR_xlsx')
SUVR_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_pvc_SUVR'),
                         joinsource='infosource', joinfield=['sheetlist'],
                         synchronize=True, unique=True, name='SUVR_pvc_xlsx')
SUVR_Q3_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_Q3_SUVR'),
                        joinsource='infosource', joinfield=['sheetlist'],
                        synchronize=True, unique=True, name='SUVR_Q3_xlsx')
SUVR_Q3_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_Q3_pvc_SUVR'),
                            joinsource='infosource', joinfield=['sheetlist'],
                            synchronize=True, unique=True, name='SUVR_Q3_pvc_xlsx')

# 2. RUN
outputdir_gen = Node(Function(input_names=['dir1','dir2'],
                              output_names=['outputdir'],
                              function=concat_dirs), name="outputdir_gen")
outputdir_gen.inputs.dir1 = output_dir
av1451_wrapper = Node(interface=PipelineWrapper(t_start_SUVR=startTime_80to100min,
                                                t_end_SUVR=endTime_80to100min,
                                                unpad_size=unpad_size,
                                                signal_threshold=signal_threshold,
                                                psf_fwhm_x=psf_fwhm_x,
                                                psf_fwhm_y=psf_fwhm_y,
                                                psf_fwhm_z=psf_fwhm_z,
                                                smooth_fwhm=smooth_fwhm,
                                                n_procs=2), # no_pvc=False
                      name="av1451_wrapper")

av1451_workflow = Workflow(name="av1451_workflow")
av1451_workflow.base_dir = output_dir
av1451_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'av1451_crashdumps')}}
av1451_workflow.connect([# PET time frame realignment
                         (infosource, getav1451, [('idvi','key')]),
                         (infosource, getav1451timing, [('idvi','key')]),
                         (infosource, getmusemri, [('idvi','key')]),
                         (infosource, getmusemriskull, [('idvi','key')]),
                         (infosource, getmuselabel, [('idvi','key')]),
                         (infosource, selectfiles, [('idvi','idvi')]),

                         (infosource, outputdir_gen, [('idvi','dir2')]),
                         (outputdir_gen, av1451_wrapper, [('outputdir','outputdir')]),

                         (getav1451, av1451_wrapper, [('av1451','pet4D')]),
                         (getav1451timing, av1451_wrapper, [('av1451timing','pettiming')]),
                         (getmusemri, av1451_wrapper, [('musemri','mri')]),
                         (getmusemriskull, av1451_wrapper, [('musemriskull','mriskull')]),
                         (getmuselabel, av1451_wrapper, [('muselabel','label')]),
                         (selectfiles, av1451_wrapper, [('composite','mnitransform')]),

                         (av1451_wrapper, SUVR_xlsx, [('suvr_xlsx', 'sheetlist')]),
                         (av1451_wrapper, SUVR_pvc_xlsx, [('suvr_pvc_xlsx', 'sheetlist')]),
                         (av1451_wrapper, SUVR_Q3_xlsx, [('suvr_Q3_xlsx', 'sheetlist')]),
                         (av1451_wrapper, SUVR_Q3_pvc_xlsx, [('suvr_Q3_pvc_xlsx', 'sheetlist')])
                        ])

result = av1451_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
