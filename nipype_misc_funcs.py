import os
import numpy as np
import nibabel as nib
from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec
from nipype.utils.filemanip import split_filename
from kineticmodel import SRTM_Zhou2003

def to_div_string(val):
    return ' -div ' + str(val)


def get_base_filename(pth):
    from nipype.utils.filemanip import split_filename
    _, base, _ = split_filename(pth)
    return base + '_'

def get_value(key, dict):
    return dict[key]

def to_div_string(val):
    return ' -div ' + str(val)

def reverse_list(l):
    return l[::-1]


class Pad4DImageInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='Dynamic image to be padded', mandatory=True)
    padsize = traits.Int(desc='Each time frame will be padded on each size by padsize', mandatory=True)

class Pad4DImageOutputSpec(TraitedSpec):
    paddedImgFile = File(exists=True, desc='Padded dynamic image')

class Pad4DImage(BaseInterface):
    """
    Pad each timeframe on each of the 6 sides (top, bottom, left, right, front, back) with the nearest slice
    """

    input_spec = Pad4DImageInputSpec
    output_spec = Pad4DImageOutputSpec

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        padsize = self.inputs.padsize
        _, base, _ = split_filename(timeSeriesImgFile)

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()
        [rows,cols,slices,comps] = img_dat.shape

        padded_dat = np.zeros((rows+2*padsize,cols+2*padsize,slices+2*padsize,comps))

        for l in range(comps):
            padded_dat[padsize:(rows+padsize),padsize:(cols+padsize),padsize:(slices+padsize),l] = img_dat[:,:,:,l]
            for x in range(padsize):
                padded_dat[x,:,:,l] = padded_dat[padsize,:,:,l]
                padded_dat[-(x+1),:,:,l] = padded_dat[-(padsize+1),:,:,l]
            for y in range(padsize):
                padded_dat[:,y,:,l] = padded_dat[:,padsize,:,l]
                padded_dat[:,-(y+1),:,l] = padded_dat[:,-(padsize+1),:,l]
            for z in range(padsize):
                padded_dat[:,:,z,l] = padded_dat[:,:,padsize,l]
                padded_dat[:,:,-(z+1),l] = padded_dat[:,:,-(padsize+1),l]

        # Save results
        paddedImg = nib.Nifti1Image(padded_dat, img.affine)
        paddedImgFile = base+'_padded.nii'
        nib.save(paddedImg,paddedImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['paddedImgFile'] = os.path.abspath(base+'_padded.nii')

        return outputs

class Unpad4DImageInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='Dynamic image unpad', mandatory=True)
    padsize = traits.Int(desc='Each time frame will be unpadded on each size by padsize', mandatory=True)

class Unpad4DImageOutputSpec(TraitedSpec):
    unpaddedImgFile = File(exists=True, desc='Unpadded dynamic image')

class Unpad4DImage(BaseInterface):
    """
    Pad each timeframe on each of the 6 sides (top, bottom, left, right, front, back) with the nearest slice
    """

    input_spec = Unpad4DImageInputSpec
    output_spec = Unpad4DImageOutputSpec

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        padsize = self.inputs.padsize
        _, base, _ = split_filename(timeSeriesImgFile)

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()
        [rows,cols,slices,comps] = img_dat.shape

        unpadded_dat = np.zeros((rows-2*padsize,cols-2*padsize,slices-2*padsize,comps))

        for l in range(comps):
            unpadded_dat[:,:,:,l] = img_dat[padsize:(rows-padsize),padsize:(cols-padsize),padsize:(slices-padsize),l]

        # Save results
        unpaddedImg = nib.Nifti1Image(unpadded_dat, img.affine)
        unpaddedImgFile = base+'_unpadded.nii.gz'
        nib.save(unpaddedImg,unpaddedImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['unpaddedImgFile'] = os.path.abspath(base+'_unpadded.nii.gz')

        return outputs



class CombineROIsInputSpec(BaseInterfaceInputSpec):
    labelImgFile = File(exists=True, desc='Label image file containing ROIs to be combined', mandatory=True)
    ROI_groupings = traits.List(traits.List(traits.Int(), minlen=1),
                                minlen=1, desc='list of lists of integers',
                                mandatory=True)
    #ROI_names = traits.List(traits.String(), minlen=1, desc='list of strings',
    #                        mandatory=False)

class CombineROIsOutputSpec(TraitedSpec):
    roi4DMaskFile = File(exists=True, desc='4D image volume, each corresponding to a combined ROI')
    #ROI_names = traits.List(traits.String(), desc='names of combined ROIs')

class CombineROIs(BaseInterface):
    """
    Combine multiple ROIs and write resulting mask to image.
    If multiple ROI combinations are provided, the result will be a 4D mask image,
    with each 3D image representing a separate ROI combination mask.

    """

    input_spec = CombineROIsInputSpec
    output_spec = CombineROIsOutputSpec

    def _run_interface(self, runtime):
        labelImgFile = self.inputs.labelImgFile
        ROI_groupings = self.inputs.ROI_groupings

        _, base, _ = split_filename(labelImgFile)

        labelimage = nib.load(labelImgFile)
        labelimage_dat = labelimage.get_data()

        ROI4Dmask_shape = list(labelimage_dat.shape)
        ROI4Dmask_shape.append(len(ROI_groupings))
        ROI4Dmask_dat = np.zeros(ROI4Dmask_shape)

        for n, ROI_grouping in enumerate(ROI_groupings):
            ROI_mask = labelimage_dat==ROI_grouping[0]
            if len(ROI_grouping)>1:
                for ROI in ROI_grouping[1:]:
                    ROI_mask = ROI_mask | (labelimage_dat==ROI)
            ROI4Dmask_dat[:,:,:,n] = ROI_mask

        # Save 4D mask
        ROI4Dmask = nib.Nifti1Image(np.squeeze(ROI4Dmask_dat), labelimage.affine, labelimage.header)
        ROI4Dmaskfile = base+'_'+'{:d}'.format(len(ROI_groupings))+'combinedROIs.nii.gz'
        nib.save(ROI4Dmask,ROI4Dmaskfile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.labelImgFile
        _, base, _ = split_filename(fname)

        outputs['roi4DMaskFile'] = os.path.abspath(base+'_'+'{:d}'.format(len(self.inputs.ROI_groupings))+'combinedROIs.nii.gz')

        #if isdefined(self.inputs.ROI_names):
        #    outputs['ROI_names'] = self.inputs.ROI_names
        #else:
        #    outputs['ROI_names'] = ['combinedROI_' + str(i) for i in list(range(len(self.inputs.ROI_groupings)))]

        return outputs




class ROI_stats_to_spreadsheetInputSpec(BaseInterfaceInputSpec):
    imgFile = File(exists=True, desc='Image list', mandatory=True)
    labelImgFile = File(exists=True, desc='Label image list', mandatory=True)
    ROI_list = traits.List(traits.Int(), minlen=1,
                           desc='list of ROI indices for which stats will be computed (should match the label indices in the label image)',
                           mandatory=True)
    ROI_names = traits.List(traits.String(), minlen=1,
                            desc='list of equal size to ROI_list that lists the corresponding ROI names',
                            mandatory=True)
    additionalROIs = traits.List(traits.List(traits.Int()), desc='list of lists of integers')
    additionalROI_names = traits.List(traits.String(),
                                      desc='names corresponding to additional ROIs')
    stat = traits.Enum('mean','Q1','median','Q3','min','max',
                       desc='one of: mean, Q1, median, Q3, min, max',
                       mandatory=True)

class ROI_stats_to_spreadsheetOutputSpec(TraitedSpec):
    xlsxFile = File(exists=True, desc='xlsx file')

class ROI_stats_to_spreadsheet(BaseInterface):
    """
    Compute ROI statistics and write to spreadsheet

    """

    input_spec = ROI_stats_to_spreadsheetInputSpec
    output_spec = ROI_stats_to_spreadsheetOutputSpec

    def _run_interface(self, runtime):
        import xlsxwriter

        imgFile = self.inputs.imgFile
        labelImgFile = self.inputs.labelImgFile
        ROI_list = self.inputs.ROI_list
        ROI_names = self.inputs.ROI_names
        additionalROIs = self.inputs.additionalROIs
        additionalROI_names = self.inputs.additionalROI_names
        stat = self.inputs.stat

        _, base, _ = split_filename(self.inputs.imgFile)
        xlsxfile = os.path.abspath(base+'_ROI_stats_'+stat+'.xlsx')

        assert(len(ROI_list)==len(ROI_names))
        assert(len(additionalROIs)==len(additionalROI_names))

        # Excel worksheet
        workbook = xlsxwriter.Workbook(xlsxfile)
        worksheet = workbook.add_worksheet('ROI means')

        row = 0
        col = 0
        worksheet.write(row,col,'image path')
        col += 1
        worksheet.write(row,col,'label image path')
        for ROI in ROI_list + additionalROIs:
            col += 1
            worksheet.write(row,col,str(ROI))

        row = 1
        col = 0
        worksheet.write(row,col,'image path')
        col += 1
        worksheet.write(row,col,'label image path')
        for ROI_name in ROI_names + additionalROI_names:
            col += 1
            worksheet.write(row,col,ROI_name)

        row = 2

        image = nib.load(imgFile)
        image_dat = image.get_data()

        labelimage = nib.load(labelImgFile)
        labelimage_dat = labelimage.get_data()

        #ROI_list = np.unique(labelimage_dat)
        bn = os.path.basename(imgFile)
        col = 0
        worksheet.write(row,col,imgFile)
        col += 1
        worksheet.write(row,col,labelImgFile)

        for ROI in ROI_list:
            ROI_mask = labelimage_dat==ROI
            if ROI_mask.sum()>0:
                if stat=="mean":
                    ROI_stat = image_dat[ROI_mask].mean()
                elif stat=="Q1":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 25)
                elif stat=="median":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 50)
                elif stat=="Q3":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 75)
                elif stat=="min":
                    ROI_stat = np.min(image_dat[ROI_mask])
                elif stat=="max":
                    ROI_stat = np.max(image_dat[ROI_mask])
                else:
                    ROI_stat=''
                if np.isnan(ROI_stat):
                    ROI_stat = ''
            else:
                ROI_stat = ''
            col += 1
            worksheet.write(row,col,ROI_stat)


        for compositeROI in additionalROIs:
            ROI_mask = labelimage_dat==compositeROI[0]
            if len(compositeROI)>1:
                for compositeROImember in compositeROI[1:]:
                    ROI_mask = ROI_mask | (labelimage_dat==compositeROImember)
            if ROI_mask.sum()>0:
                if stat=="mean":
                    ROI_stat = image_dat[ROI_mask].mean()
                elif stat=="Q1":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 25)
                elif stat=="median":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 50)
                elif stat=="Q3":
                    ROI_stat = np.percentile(image_dat[ROI_mask], 75)
                elif stat=="min":
                    ROI_stat = np.min(image_dat[ROI_mask])
                elif stat=="max":
                    ROI_stat = np.max(image_dat[ROI_mask])
                else:
                    ROI_stat=''
                if np.isnan(ROI_stat):
                    ROI_stat = ''
            else:
                ROI_stat = ''
            col += 1
            worksheet.write(row,col,ROI_stat)

        workbook.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        _, base, _ = split_filename(self.inputs.imgFile)

        outputs['xlsxFile'] = os.path.abspath(base+'_ROI_stats_'+self.inputs.stat+'.xlsx')

        return outputs


'''
class ROI_timestats_to_spreadsheet(BaseInterface):
    # compute ROI stats for each ROI over time for one scan

    input_spec = ROI_timestats_to_spreadsheetInputSpec
    output_spec = ROI_timestats_to_spreadsheetOutputSpec

    def _run_interface(self, runtime):
        import xlsxwriter

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        return outputs
'''


class ConcatenateSpreadsheetsInputSpec(BaseInterfaceInputSpec):
    sheetlist = traits.List(File, minlen=2, mandatory=True,
                            desc='list of spreadsheets to concatenate')
    outputname = traits.String(mandatory=True, desc='name of output spreadsheet without extension')

class ConcatenateSpreadsheetsOutputSpec(TraitedSpec):
    concatenatedlist = File(exists=True, desc='concatenated spreadsheet')

class ConcatenateSpreadsheets(BaseInterface):
    input_spec = ConcatenateSpreadsheetsInputSpec
    output_spec = ConcatenateSpreadsheetsOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        sheetlist = self.inputs.sheetlist
        outputname = self.inputs.outputname

        # read in first spreadsheet
        firstsheet = pd.read_excel(sheetlist[0])

        for sheetname in sheetlist[1:]:
            print("************")
            print(sheetname)
            sheet = pd.read_excel(sheetname)
            # check that column names are equal
            if not np.array_equal(firstsheet.columns.values, sheet.columns.values):
                raise ValueError('Column headers differ')

            # check that first rows are equal
            if not firstsheet.loc[0,:].equals(sheet.loc[0,:]):
                raise ValueError('First rows differ')

            # append to firstsheet
            firstsheet = firstsheet.append(sheet.loc[1,:], ignore_index=True)

        # sort by 'image path'
        # firstsheet.sort_values(by=['image path'], inplace=True) # this doesn't work because of two header rows

        # save spreadsheet
        writer = pd.ExcelWriter(os.path.abspath(self.inputs.outputname+'.xlsx'))
        firstsheet.to_excel(writer, 'Sheet1', index=False)
        writer.save()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['concatenatedlist'] = os.path.abspath(self.inputs.outputname+'.xlsx')

        return outputs
