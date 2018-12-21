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
    mri = File(desc="path to preprocessed MRI image without skull", exists=True, mandatory=True, position=3, argstr="%s")
    label = File(desc="path to anatomical label image (in MRI space)", exists=True, mandatory=True, position=4, argstr="%s")
    mnitransform = File(desc="path to composite (deformable) transform that takes MRI to MNI space", exists=True, mandatory=True, position=5, argstr="%s")
    outputdir = Directory(desc="output directory", exists=False, mandatory=True, position=6, argstr="%s")

    t_start = traits.Float(desc="Frames prior to t_start (in min) will be excluded from analysis",
                           mandatory=False, argstr="--t_start %f")
    t_end_realign = traits.Float(desc="Time frame alignment will use the average of frames prior to t_end_realign (in min) as the target",
                                 mandatory=False, argstr="--t_end_realign %f")
    t_end_coreg = traits.Float(desc="MRI coregistration will use the average of the frames prior to t_end_coreg (in min) as the source",
                               mandatory=False, argstr="--t_end_coreg %f")
    t_end_EA = traits.Float(desc="early amyloid image will be computed as the average of frames prior to t_end_EA (in min)",
                            mandatory=False, argstr="--t_end_EA %f")
    t_end_kinetic_model = traits.Float(desc="Parametric images will be computed as the average of frames between t_start and t_end_kinetic_model (in min)",
                                       mandatory=False, argstr="--t_end_kinetic_model %f")
    t_start_SUVR = traits.Float(desc="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)",
                                mandatory=False, argstr="--t_start_SUVR %f")
    t_end_SUVR = traits.Float(desc="SUVR image will be computed as the average of frames between t_start_SUVR and t_end_SUVR (in min)",
                              mandatory=False, argstr="--t_end_SUVR %f")
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
    dvr_xlsx = File(desc="DVR spreadsheet", exists=True)
    dvr_pvc_xlsx = File(desc="DVR spreadsheet (with PVC)", exists=True)

    r1_xlsx = File(desc="R1 spreadsheet", exists=True)
    r1_pvc_xlsx = File(desc="R1 spreadsheet (with PVC)", exists=True)

    r1_lrsc_xlsx = File(desc="R1 LRSC spreadsheet", exists=True)
    r1_lrsc_pvc_xlsx = File(desc="R1 LRSC spreadsheet (with PVC)", exists=True)

    suvr_xlsx = File(desc="SUVR spreadsheet", exists=True)
    suvr_pvc_xlsx = File(desc="SUVR spreadsheet (with PVC)", exists=True)

    ea_xlsx = File(desc="EA spreadsheet", exists=True)
    ea_pvc_xlsx = File(desc="EA spreadsheet (with PVC)", exists=True)

    ea_wb_xlsx = File(desc="EA wb spreadsheet", exists=True)
    ea_pvc_wb_xlsx = File(desc="EA wb spreadsheet (with PVC)", exists=True)

class PipelineWrapper(CommandLine):
    _cmd = 'python /code/pib.py'
    input_spec = PipelineWrapperInputSpec
    output_spec = PipelineWrapperOutputSpec

    def _list_outputs(self):
        # one output should be the full path to the SUVR spreadsheet
        outputs = self._outputs().get()

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_workflow','ROImeans_dvr','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['dvr_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No DVR xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_workflow','ROImeans_r1_wlr','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['r1_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No R1 xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_workflow','ROImeans_r1_lrsc','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['r1_lrsc_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No R1 LRSC xlsx found!')


        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','EA_workflow','ROImeans_EA','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['ea_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No EA xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','EA_workflow','ROImeans_EA_wb','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['ea_wb_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No EA WB xlsx found!')



        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_pvc_workflow','ROImeans_dvr_pvc','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['dvr_pvc_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No DVR PVC xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_pvc_workflow','ROImeans_r1_wlr_pvc','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['r1_pvc_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No R1 PVC xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','kinetic_model_pvc_workflow','ROImeans_r1_lrsc_pvc','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['r1_lrsc_pvc_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No R1 LRSC PVC xlsx found!')


        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','EA_pvc_workflow','ROImeans_EA_pvc','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['ea_pvc_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No EA PVC xlsx found!')

        xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','EA_pvc_workflow','ROImeans_EA_pvc_wb','*.xlsx'))
        if xlsx_list and len(xlsx_list)==1:
            outputs['ea_pvc_wb_xlsx'] = xlsx_list[0]
        else:
            raise ValueError('No EA PVC WB xlsx found!')




        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','SUVR_workflow','ROImeans','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR xlsx found!')

        suvr_xlsx_list = glob(os.path.join(self.inputs.outputdir,'pib_workflow','SUVR_pvc_workflow','ROImeans_pvc','*.xlsx'))
        if suvr_xlsx_list and len(suvr_xlsx_list)==1:
            outputs['suvr_pvc_xlsx'] = suvr_xlsx_list[0]
        else:
            raise ValueError('No SUVR PVC xlsx found!')

        return outputs
