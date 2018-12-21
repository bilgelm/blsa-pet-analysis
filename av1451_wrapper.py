from nipype.interfaces.base import TraitedSpec, File, Directory, traits, CommandLineInputSpec, CommandLine
from glob import glob
import os

def get_value(key, dict):
    return dict[key]

def concat_dirs(dir1, dir2):
    import os
    return os.path.join(dir1, 'BLSA_'+dir2)

class PipelineWrapperInputSpec(CommandLineInputSpec):
    pet4D = File(desc="path to 4D PET image", exists=True, mandatory=True, position=0, argstr="%s")
    pettiming = File(desc="path to csv file describing PET timing info", exists=True, mandatory=True, position=1, argstr="%s")
    mriskull = File(desc="path to preprocessed MRI image with skull", exists=True, mandatory=True, position=2, argstr="%s")
    mri = File(desc="path to preprocessed MRI image without skull", exists=True, mandatory=True, position=3, argstr="%s")
    label = File(desc="path to anatomical label image (in MRI space)", exists=True, mandatory=True, position=4, argstr="%s")
    mnitransform = File(desc="path to composite (deformable) transform that takes MRI to MNI space", exists=True, mandatory=True, position=5, argstr="%s")
    outputdir = Directory(desc="output directory", exists=False, mandatory=True, position=6, argstr="%s")

    t_start_SUVR = traits.Float(desc="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)",
                                mandatory=False, argstr="--t_start_SUVR %f")
    t_end_SUVR = traits.Float(desc="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)",
                              mandatory=False, argstr="--t_end_SUVR %f")
    unpad_size = traits.Int(desc="number of slices to remove from each side of the PET image",
                            mandatory=False, argstr="--unpad_size %d")
    signal_threshold = traits.Float(desc="ignore signal below this threshold when computing ROI averages",
                                    mandatory=False, argstr="--signal_threshold %f")
    no_pvc = traits.Bool(desc="if True, do not perform partial volume correction (default is to perform partial volume correction)",
                         mandatory=False, argstr="--no_pvc")
    psf_fwhm_x = traits.Float(desc="PET scanner PSF FWHM along x (in mm)",
                              mandatory=False, argstr="--psf_fwhm_x %f")
    psf_fwhm_y = traits.Float(desc="PET scanner PSF FWHM along y (in mm)",
                              mandatory=False, argstr="--psf_fwhm_y %f")
    psf_fwhm_z = traits.Float(desc="PET scanner PSF FWHM along z (in mm)",
                              mandatory=False, argstr="--psf_fwhm_z %f")
    smooth_fwhm = traits.Float(desc="FWHM of Gaussian smoothing filter (in mm)",
                               mandatory=False, argstr="--smooth_fwhm %f")
    n_procs = traits.Int(desc="number of parallel processes", mandatory=False,
                         argstr="--n_procs %d")

class PipelineWrapperOutputSpec(TraitedSpec):
    suvr_xlsx = File(desc="SUVR spreadsheet", exists=True)
    suvr_pvc_xlsx = File(desc="SUVR spreadsheet (with PVC)", exists=True)
    suvr_Q3_xlsx = File(desc="SUVR Q3 spreadsheet", exists=True)
    suvr_Q3_pvc_xlsx = File(desc="SUVR Q3 spreadsheet (with PVC)", exists=True)

class PipelineWrapper(CommandLine):
    _cmd = 'python /code/av1451.py'
    input_spec = PipelineWrapperInputSpec
    output_spec = PipelineWrapperOutputSpec

    def _list_outputs(self):
        # one output should be the full path to the SUVR spreadsheet
        outputs = self._outputs().get()

        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'av1451_workflow','SUVR_workflow','ROImeans','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR xlsx found!')

        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'av1451_workflow','SUVR_pvc_workflow','ROImeans','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_pvc_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR PVC xlsx found!')

        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'av1451_workflow','SUVR_workflow','ROI_Q3','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_Q3_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR Q3 xlsx found!')

        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'av1451_workflow','SUVR_pvc_workflow','ROI_Q3','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_Q3_pvc_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR Q3 PVC xlsx found!')

        return outputs
