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
from pib_wrapper import get_value, concat_dirs, PipelineWrapper # pib_wrapper.py must be in current working directory (or otherwise discoverable by python)
from nipype import config, logging
from nipype_misc_funcs import ConcatenateSpreadsheets
config.enable_debug_mode()
logging.update_logging(config)

# number of parallel processes
n_procs = 24

# directory to store the workflow results
output_dir = '/output/pib/'

# prefix for the data collection site
sitePrefix = 'BLSA'

# spreadsheet with the following columns: blsaid, blsavi, PIBpath,PIBtimingpath
organization_spreadsheet = '/input/PETstatus_09Nov2018.xlsx'

# columns required in the spreadsheet
required_cols = ['blsaid','blsavi','PIBpath','PIBtimingpath','PIBscanner','musemripath','muselabelpath']

# values to be treated as missing in the spreadsheet - do not include NA as a null value as it is a valid EMSID
NAN_VALUES = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan','']


# PiB processing parameters
startTime = 0.75 # ignore frames prior to this time

# 2-min mean, used for time frame realignment
endTime_2min = 2

# 20-min mean, used for MRI coregistration
endTime_20min = 20

# 5-min mean, used for early amyloid image computation
endTime_EA = 2.5

# 50 to 70-min mean, used for SUVR computation
startTime_50to70min = 50
endTime_50to70min = 70

# Used for DVR computation
endTime_DVR = 70

# PET scanner PSF FWHM (in mm), used for PVC
psf_fwhm_x = 6.7
psf_fwhm_y = 6.7
psf_fwhm_z = 6.7

# Smoothing parameter (in mm) used for SUVR
smooth_fwhm = 4.25

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


# Read in the organization spreadsheet and extract information
data_table = pd.read_excel(organization_spreadsheet, keep_default_na=False, na_values=NAN_VALUES)

for col in required_cols:
    if not col in data_table.columns:
        sys.exit('Required column ' + col + \
                 ' is not present in the data organization spreadsheet ' + \
                 organization_spreadsheet + '!')

data_table = data_table[data_table.excludePIB != 'y']
data_table = data_table[data_table.PIBscanner=='GE Advance']

# Find all visits with PiB-PET (excluding incomplete/interrupted PiB scans) and MUSE labels
data_table = data_table[required_cols].dropna(axis=0, how='any')

#mri_list = pib_data['mripath'].values.tolist()
musemri_list = data_table['musemripath'].values.tolist()
muselabel_list = data_table['muselabelpath'].values.tolist()
pib_list = data_table['PIBpath'].values.tolist()
pibtiming_list = data_table['PIBtimingpath'].values.tolist()

subjID = data_table['blsaid'].values.tolist()
visNo = data_table['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
blsavi_decimal = [ str(x).split('.')[1] for x in visNo ]

idvi_list = ["%04d_%02d-%s" % idvi for idvi in zip(subjID,blsavi_integer,blsavi_decimal)]

# Form dictionaries, with IDs as keys and paths to images as values
pib_dict = dict(zip(idvi_list, pib_list))
pibtiming_dict = dict(zip(idvi_list, pibtiming_list))
musemri_dict = dict(zip(idvi_list, musemri_list))
muselabel_dict = dict(zip(idvi_list, muselabel_list))


# ## 1. INPUTS
#
# We set up the nipype Nodes that will act as the inputs to our Workflows. The `infosource` Node allows for iterating over scan IDs.

# placeholder Node to enable iteration over scans
infosource = Node(interface=IdentityInterface(fields=['idvi']),
                  name="infosource")
infosource.iterables = ('idvi', idvi_list) # id 139 is BLSA_1547_19-0, which gives an error in processing

# get full path to PiB scan corresponding to idvi from spreadsheet
getpib = Node(Function(input_names=['key','dict'],
                       output_names=['pib'],
                       function=get_value),
              name='getpib')
getpib.inputs.dict = pib_dict

# get full path to the txt file listing the duration of each PiB time frame
#  number of rows must be the same as the number of PiB time frames,
#  with each row listing the time in minutes
getpibtiming = Node(Function(input_names=['key','dict'],
                             output_names=['pibtiming'],
                             function=get_value),
                    name='getpibtiming')
getpibtiming.inputs.dict = pibtiming_dict

# get full path to MRI corresponding to idvi from spreadsheet,
#  in same space as MUSE labels
getmusemri = Node(Function(input_names=['key','dict'],
                           output_names=['musemri'],
                           function=get_value),
                  name='getmusemri')
getmusemri.inputs.dict = musemri_dict

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


DVR_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_DVR'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='DVR_xlsx')
R1_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_R1'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='R1_xlsx')
R1_lrsc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_R1_lrsc'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='R1_lrsc_xlsx')
EA_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_EA'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='EA_xlsx')
EA_wb_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_EA_wb'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='EA_wb_xlsx')

DVR_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_DVR_pvc'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='DVR_pvc_xlsx')
R1_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_R1_pvc'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='R1_pvc_xlsx')
R1_lrsc_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_R1_lrsc_pvc'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='R1_lrsc_pvc_xlsx')
EA_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_EA_pvc'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='EA_pvc_xlsx')
EA_pvc_wb_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_EA_pvc_wb'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='EA_pvc_wb_xlsx')

SUVR_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_SUVR'),
                     joinsource='infosource', joinfield=['sheetlist'],
                     synchronize=True, unique=True, name='SUVR_xlsx')
SUVR_pvc_xlsx = JoinNode(interface=ConcatenateSpreadsheets(outputname='ROI_pvc_SUVR'),
                         joinsource='infosource', joinfield=['sheetlist'],
                         synchronize=True, unique=True, name='SUVR_pvc_xlsx')

# 2. RUN
outputdir_gen = Node(Function(input_names=['dir1','dir2'],
                              output_names=['outputdir'],
                              function=concat_dirs), name="outputdir_gen")
outputdir_gen.inputs.dir1 = output_dir

pib_wrapper = Node(interface=PipelineWrapper(t_start=startTime,
                                             t_end_realign=endTime_2min,
                                             t_end_coreg=endTime_20min,
                                             t_end_EA=endTime_EA,
                                             t_end_kinetic_model=endTime_DVR,
                                             t_start_SUVR=startTime_50to70min,
                                             t_end_SUVR=endTime_50to70min,
                                             psf_fwhm_x=psf_fwhm_x,
                                             psf_fwhm_y=psf_fwhm_y,
                                             psf_fwhm_z=psf_fwhm_z,
                                             smooth_fwhm=smooth_fwhm,
                                             n_procs=2), # no_pvc=False
                   name="pib_wrapper")

pib_workflow = Workflow(name="pib_workflow")
pib_workflow.base_dir = output_dir
pib_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'pib_crashdumps')}}
pib_workflow.connect([# PET time frame realignment
                         (infosource, getpib, [('idvi','key')]),
                         (infosource, getpibtiming, [('idvi','key')]),
                         (infosource, getmusemri, [('idvi','key')]),
                         (infosource, getmuselabel, [('idvi','key')]),
                         (infosource, selectfiles, [('idvi','idvi')]),

                         (infosource, outputdir_gen, [('idvi','dir2')]),
                         (outputdir_gen, pib_wrapper, [('outputdir','outputdir')]),

                         (getpib, pib_wrapper, [('pib','pet4D')]),
                         (getpibtiming, pib_wrapper, [('pibtiming','pettiming')]),
                         (getmusemri, pib_wrapper, [('musemri','mri')]),
                         (getmuselabel, pib_wrapper, [('muselabel','label')]),
                         (selectfiles, pib_wrapper, [('composite','mnitransform')]),

                         (pib_wrapper, DVR_xlsx, [('dvr_xlsx', 'sheetlist')]),
                         (pib_wrapper, R1_xlsx, [('r1_xlsx', 'sheetlist')]),
                         (pib_wrapper, R1_lrsc_xlsx, [('r1_lrsc_xlsx', 'sheetlist')]),
                         (pib_wrapper, EA_xlsx, [('ea_xlsx', 'sheetlist')]),
                         (pib_wrapper, EA_wb_xlsx, [('ea_wb_xlsx', 'sheetlist')]),

                         (pib_wrapper, DVR_pvc_xlsx, [('dvr_pvc_xlsx', 'sheetlist')]),
                         (pib_wrapper, R1_pvc_xlsx, [('r1_pvc_xlsx', 'sheetlist')]),
                         (pib_wrapper, R1_lrsc_pvc_xlsx, [('r1_lrsc_pvc_xlsx', 'sheetlist')]),
                         (pib_wrapper, EA_pvc_xlsx, [('ea_pvc_xlsx', 'sheetlist')]),
                         (pib_wrapper, EA_pvc_wb_xlsx, [('ea_pvc_wb_xlsx', 'sheetlist')]),

                         (pib_wrapper, SUVR_xlsx, [('suvr_xlsx', 'sheetlist')]),
                         (pib_wrapper, SUVR_pvc_xlsx, [('suvr_pvc_xlsx', 'sheetlist')]),
                        ])

result = pib_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
