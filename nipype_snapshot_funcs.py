import os
from math import pi
import numpy as np
import pandas as pd
import nibabel as nib
import temporalimage

import matplotlib
matplotlib.use('Agg') # Generate images without having a window appear
import matplotlib.pyplot as plt
from dipy.viz import regtools
from nilearn.plotting import plot_anat, cm
from nilearn.masking import apply_mask

from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec, isdefined
from nipype.utils.filemanip import split_filename

class realign_snapshotsInputSpec(BaseInterfaceInputSpec):
    petrealignedfile = File(exists=True, desc='Realigned 4D PET file', mandatory=True)
    realignParamsFile = File(exists=True, desc='Realignment parameters text file', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    splitTime = traits.Float(desc='minute beyond which time frames were realigned, inclusive', mandatory=True)

class realign_snapshotsOutputSpec(TraitedSpec):
    realign_param_plot = File(exists=True, desc='Realignment parameter plot')
    realigned_img_snap = File(exists=True, desc='Realigned time frame snapshot')

class realign_snapshots(BaseInterface):
    input_spec = realign_snapshotsInputSpec
    output_spec = realign_snapshotsOutputSpec

    def _run_interface(self, runtime):
        petrealignedfile = self.inputs.petrealignedfile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        splitTime = self.inputs.splitTime
        realignParamsFile = self.inputs.realignParamsFile

        petrealigned = temporalimage.load(petrealignedfile, frameTimingCsvFile)
        t = petrealigned.get_midTime()[petrealigned.get_frameStart()>=splitTime]

        # Time realignment parameters
        rp = pd.read_csv(realignParamsFile, delim_whitespace=True, header=None).as_matrix()
        translation = rp[:,:3]
        rotation = rp[:,3:] * 180 / pi

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
        axes[0].plot(t,translation[:,0],label='x')
        axes[0].plot(t,translation[:,1],label='y')
        axes[0].plot(t,translation[:,2],label='z')
        axes[0].legend(loc=0)
        axes[0].set_title('Translation over time')
        axes[0].set_xlabel('Time (min)', fontsize=16)
        axes[0].set_ylabel('Translation (mm)', fontsize=16)

        axes[1].plot(t,rotation[:,0],label='x')
        axes[1].plot(t,rotation[:,1],label='y')
        axes[1].plot(t,rotation[:,2],label='z')
        axes[1].legend(loc=0)
        axes[1].set_title('Rotation over time')
        axes[1].set_xlabel('Time (min)', fontsize=16)
        axes[1].set_ylabel('Rotation (degrees)', fontsize=16)

        fig.tight_layout()

        _, base, _ = split_filename(realignParamsFile)
        fig.savefig(base+'_plot.png', format='png')


        # visualize time frames of the realigned scan
        I = petrealigned.get_data()
        imdim = I.shape
        vmin, vmax = np.percentile(I,[1,99])
        voxsize = petrealigned.header.get_zooms()

        # Right hemisphere is on the right hand side
        nx = int(np.ceil(np.sqrt(imdim[3])))
        fig, axes = plt.subplots(nrows=nx, ncols=nx, figsize=(16,16))
        x = y = 0
        for tt in range(nx ** 2):
            if tt < imdim[3]:
                axes[x,y].imshow(np.fliplr(I[:,:,imdim[2]//2,tt]).T, aspect=voxsize[1]/voxsize[0], cmap='hot', vmin=vmin, vmax=vmax)
                axes[x,y].set_title('#'+str(tt)+': '+str(petrealigned.get_frameStart()[tt])+'-'+str(petrealigned.get_frameEnd()[tt])+' min')
            axes[x,y].set_axis_off()
            y += 1
            if y>=np.ceil(np.sqrt(imdim[3])):
                y = 0
                x += 1

        fig.tight_layout()

        _, base, _ = split_filename(petrealignedfile)
        fig.savefig(base+'_snap.png', format='png')

        return runtime

    def _list_outputs(self):
        petrealignedfile = self.inputs.petrealignedfile
        realignParamsFile = self.inputs.realignParamsFile

        outputs = self._outputs().get()

        _, base, _ = split_filename(realignParamsFile)
        outputs['realign_param_plot'] = os.path.abspath(base+'_plot.png')

        _, base, _ = split_filename(petrealignedfile)
        outputs['realigned_img_snap'] = os.path.abspath(base+'_snap.png')

        return outputs


class coreg_snapshotsInputSpec(BaseInterfaceInputSpec):
    mriregfile = File(exists=True, desc='MRI image registered to PET', mandatory=True)
    petavgfile = File(exists=True, desc='PET average image', mandatory=True)

class coreg_snapshotsOutputSpec(TraitedSpec):
    coreg_edges = File(exists=True, desc='PET with coregistered MRI edges')
    coreg_overlay_sagittal = File(exists=True, desc='Overlay of PET and coregistered MRI, sagittal')
    coreg_overlay_coronal = File(exists=True, desc='Overlay of PET and coregistered MRI, coronal')
    coreg_overlay_axial = File(exists=True, desc='Overlay of PET and coregistered MRI, axial')

class coreg_snapshots(BaseInterface):
    input_spec = coreg_snapshotsInputSpec
    output_spec = coreg_snapshotsOutputSpec

    def _run_interface(self, runtime):
        petavgfile = self.inputs.petavgfile
        mriregfile = self.inputs.mriregfile

        mrireg = nib.load(mriregfile)
        petavg = nib.load(petavgfile)

        _, base, _ = split_filename(petavgfile)

        # Visualize the overlaid PiB 20-min average and the coregistered MRI
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 0, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_sagittal.png')
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 1, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_coronal.png')
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 2, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_axial.png')

        fig = plt.figure(figsize=(15,5))
        display = plot_anat(petavgfile,figure=fig)
        display.add_edges(mriregfile)
        display.title('MRI edges on PET')
        fig.savefig(base+'_coreg_edges.png', format='png')

        return runtime

    def _list_outputs(self):
        petavgfile = self.inputs.petavgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(petavgfile)
        outputs['coreg_edges'] = os.path.abspath(base+'_coreg_edges.png')
        outputs['coreg_overlay_sagittal'] = os.path.abspath(base+'_coreg_overlay_sagittal.png')
        outputs['coreg_overlay_coronal'] = os.path.abspath(base+'_coreg_overlay_coronal.png')
        outputs['coreg_overlay_axial'] = os.path.abspath(base+'_coreg_overlay_axial.png')

        return outputs



class labels_snapshotsInputSpec(BaseInterfaceInputSpec):
    labelfile = File(exists=True, desc='4D label image', mandatory=True)
    labelnames = traits.List(traits.String(), minlen=1,
                             desc='list of equal size to the fourth dimension of the label image',
                             mandatory=True)

class labels_snapshotsOutputSpec(TraitedSpec):
    label_snap = File(exists=True, desc='Label image snapshot')

class labels_snapshots(BaseInterface):
    input_spec = labels_snapshotsInputSpec
    output_spec = labels_snapshotsOutputSpec

    def _run_interface(self, runtime):
        labelfile = self.inputs.labelfile
        labelnames = self.inputs.labelnames

        label = nib.load(labelfile)
        I = label.get_data()
        imdim = I.shape
        voxsize = label.header.get_zooms()

        # Right hemisphere is on the right hand side
        fig, axes = plt.subplots(imdim[3],3,figsize=(10,60))
        for tt in range(imdim[3]):

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(0,1)))
            axes[tt,0].imshow(np.fliplr(I[:,:,sli,tt]).T, aspect=voxsize[1]/voxsize[0])
            axes[tt,0].set_axis_off()

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(0,2)))
            axes[tt,1].imshow(np.fliplr(I[:,sli,:,tt]).T, aspect=voxsize[2]/voxsize[0])
            axes[tt,1].set_axis_off()
            axes[tt,1].set_title(labelnames[tt])

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(1,2)))
            axes[tt,2].imshow(np.fliplr(I[sli,:,:,tt]).T, aspect=voxsize[2]/voxsize[1])
            axes[tt,2].set_axis_off()
        fig.tight_layout()

        _, base, _ = split_filename(labelfile)
        fig.savefig(base+'_snap.png', format='png')

        return runtime

    def _list_outputs(self):
        labelfile = self.inputs.labelfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(labelfile)
        outputs['label_snap'] = os.path.abspath(base+'_snap.png')

        return outputs


class refReg_snapshotsInputSpec(BaseInterfaceInputSpec):
    petavgfile = File(exists=True, desc='PET average image', mandatory=True)
    petrealignedfile = File(exists=True, desc='Realigned 4D PET file', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    maskfile = File(exists=True, desc='Mask file', mandatory=True)

class refReg_snapshotsOutputSpec(TraitedSpec):
    maskOverlay_axial = File(exists=True, desc='Mask overlaid on PET, axial')
    maskOverlay_coronal = File(exists=True, desc='Mask overlaid on PET, coronal')
    maskOverlay_sagittal = File(exists=True, desc='Mask overlaid on PET, sagittal')
    mask_TAC = File(exists=True, desc='Reference region time activity curve')

class refReg_snapshots(BaseInterface):
    input_spec = refReg_snapshotsInputSpec
    output_spec = refReg_snapshotsOutputSpec

    def _run_interface(self, runtime):
        petavgfile = self.inputs.petavgfile
        petrealignedfile = self.inputs.petrealignedfile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        maskfile = self.inputs.maskfile

        petrealigned = temporalimage.load(petrealignedfile, frameTimingCsvFile)
        t = petrealigned.get_midTime()

        _, base, _ = split_filename(petavgfile)
        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="z", cut_coords=10)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - axial')
        fig.savefig(base+'_maskOverlay_axial.png', format='png')

        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="y", cut_coords=10, annotate=False)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - coronal')
        fig.savefig(base+'_maskOverlay_coronal.png', format='png')

        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="x", cut_coords=10, annotate=False)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - sagittal')
        fig.savefig(base+'_maskOverlay_sagittal.png', format='png')

        # Reference region Time Activity Curve (TAC)
        masked_data = apply_mask(petrealignedfile, maskfile)
        ref_TAC = np.mean(masked_data,axis=1)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(t,ref_TAC)
        ax.set_title('Reference region Time Activity Curve')
        ax.set_xlabel('Time (min)', fontsize=16)
        ax.set_ylabel('Activity', fontsize=16)
        fig.tight_layout()

        _, base, _ = split_filename(petrealignedfile)
        fig.savefig(base+'_mask_TAC.png', format='png')

        return runtime

    def _list_outputs(self):
        petrealignedfile = self.inputs.petrealignedfile
        petavgfile = self.inputs.petavgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(petavgfile)
        outputs['maskOverlay_axial'] = os.path.abspath(base+'_maskOverlay_axial.png')
        outputs['maskOverlay_coronal'] = os.path.abspath(base+'_maskOverlay_coronal.png')
        outputs['maskOverlay_sagittal'] = os.path.abspath(base+'_maskOverlay_sagittal.png')

        _, base, _ = split_filename(petrealignedfile)
        outputs['mask_TAC'] = os.path.abspath(base+'_mask_TAC.png')

        return outputs


class triplanar_snapshotsInputSpec(BaseInterfaceInputSpec):
    imgfile = File(exists=True, desc='Image file', mandatory=True)
    bgimgfile = File(exists=True, desc='Background image file', mandatory=False)
    vmin = traits.Float(desc='vmin', mandatory=False)
    vmax = traits.Float(desc='vmax', mandatory=False)
    cmap = traits.String(desc='cmap', mandatory=False)
    alpha = traits.Range(low=0.0, high=1.0, desc='alpha', mandatory=False) # higher alpha means more bg visibility
    x = traits.Range(low=0, desc='x cut', mandatory=False)
    y = traits.Range(low=0, desc='y cut', mandatory=False)
    z = traits.Range(low=0, desc='z cut', mandatory=False)

class triplanar_snapshotsOutputSpec(TraitedSpec):
    triplanar = File(exists=True, desc='Triplanar snapshot')

class triplanar_snapshots(BaseInterface):
    input_spec = triplanar_snapshotsInputSpec
    output_spec = triplanar_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg') # Generate images without having a window appear
        import matplotlib.pyplot as plt

        imgfile = self.inputs.imgfile
        bgimgfile = self.inputs.bgimgfile
        vmin = self.inputs.vmin
        vmax = self.inputs.vmax
        cmap = self.inputs.cmap
        alpha = self.inputs.alpha
        x = self.inputs.x
        y = self.inputs.y
        z = self.inputs.z

        if not isdefined(cmap):
            cmap = 'jet'
        if not isdefined(alpha):
            if not isdefined(bgimgfile):
                alpha = 0
            else:
                alpha = 0.5

        img = nib.load(imgfile)
        I = img.get_data()
        imdim = I.shape
        voxsize = img.header.get_zooms()

        if not isdefined(vmin):
            vmin = np.percentile(np.abs(I),5)
        if not isdefined(vmax):
            vmax = np.percentile(np.abs(I),100)
        if not isdefined(x):
            x = imdim[0]//2
        if not isdefined(y):
            y = imdim[1]//2
        if not isdefined(z):
            z = imdim[2]//2

        if isdefined(bgimgfile):
            bgimg = nib.load(bgimgfile)
            bgI = bgimg.get_data()
            bgimdim = bgI.shape
            bgvoxsize = bgimg.header.get_zooms()
            assert(imdim==bgimdim)
            assert(voxsize==bgvoxsize)

            # trim to remove 0 voxels
            trimmask = bgI>0
            tmp = np.argwhere(trimmask)
            (xstart, ystart, zstart), (xstop, ystop, zstop) = tmp.min(0), tmp.max(0) + 1
            bgI = bgI[xstart:xstop, ystart:ystop, zstart:zstop]
            I = I[xstart:xstop, ystart:ystop, zstart:zstop]
            imdim = I.shape
            bgimdim = bgI.shape
            x = x - xstart
            y = y - ystart
            z = z - zstart

            mask = bgI==0
            bgI = np.ma.array(bgI, mask=mask)
            I = np.ma.array(I, mask=mask)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,5), facecolor='black')
        axes[0].imshow(np.fliplr(I[:,:,z]).T,
                       aspect=voxsize[1]/voxsize[0],
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        axes[1].imshow(np.fliplr(I[:,y,:]).T,
                       aspect=voxsize[2]/voxsize[0],
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        im = axes[2].imshow(np.fliplr(I[x,:,:]).T,
                            aspect=voxsize[2]/voxsize[1],
                            cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        if isdefined(bgimgfile):
            axes[0].imshow(np.fliplr(bgI[:,:,z]).T,
                           aspect=voxsize[1]/voxsize[0],
                           cmap='gray',alpha=alpha)
            axes[1].imshow(np.fliplr(bgI[:,y,:]).T,
                           aspect=voxsize[2]/voxsize[0],
                           cmap='gray',alpha=alpha)
            axes[2].imshow(np.fliplr(bgI[x,:,:]).T,
                           aspect=voxsize[2]/voxsize[1],
                           cmap='gray',alpha=alpha)

        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()

        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=20) # colorbar legend font size

        # colorbar ticks' and labels' color
        cbar.ax.yaxis.set_tick_params(color='w')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')

        # smooth colorbar without lines in it
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")

        _, base, _ = split_filename(imgfile)
        fig.savefig(base+'_snap.png', format='png', facecolor=fig.get_facecolor())

        return runtime

    def _list_outputs(self):
        imgfile = self.inputs.imgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(imgfile)
        outputs['triplanar'] = os.path.abspath(base+'_snap.png')

        return outputs



class mosaicInputSpec(BaseInterfaceInputSpec):
    imgfile = File(exists=True, desc='Image file', mandatory=True)
    bgimgfile = File(exists=True, desc='Background image file', mandatory=False)
    vmin = traits.Float(desc='vmin')
    vmax = traits.Float(desc='vmax')
    cmap = traits.String(desc='cmap', mandatory=False)
    alpha = traits.Range(low=0.0, high=1.0, desc='alpha', mandatory=False) # higher alpha means more bg visibility

class mosaicOutputSpec(TraitedSpec):
    mosaic = File(exists=True, desc='Mosaic snapshot')

class mosaic(BaseInterface):
    input_spec = mosaicInputSpec
    output_spec = mosaicOutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg') # Generate images without having a window appear
        import matplotlib.pyplot as plt

        imgfile = self.inputs.imgfile
        bgimgfile = self.inputs.bgimgfile
        vmin = self.inputs.vmin
        vmax = self.inputs.vmax
        cmap = self.inputs.cmap
        alpha = self.inputs.alpha

        if not isdefined(cmap):
            cmap = 'jet'
        if not isdefined(alpha):
            alpha = 0

        img = nib.load(imgfile)
        I = img.get_data()
        imdim = I.shape
        voxsize = img.header.get_zooms()

        if isdefined(bgimgfile):
            bgimg = nib.load(bgimgfile)
            bgI = bgimg.get_data()
            bgimdim = bgI.shape
            bgvoxsize = bgimg.header.get_zooms()
            assert(imdim==bgimdim)
            assert(voxsize==bgvoxsize)
            mask = bgI==0
            bgI = np.ma.array(bgI, mask=mask)
            I = np.ma.array(I, mask=mask)

        nx = int(np.ceil(np.sqrt(imdim[2])))
        fig, axes = plt.subplots(nrows=nx, ncols=nx, figsize=(nx*3,nx*3), facecolor='black')
        x = y = 0
        for z in range(nx ** 2):
            if z < imdim[2]:
                if isdefined(bgimgfile):
                    axes[x,y].imshow(np.fliplr(bgI[:,:,z]).T, aspect=voxsize[1]/voxsize[0], cmap='gray', alpha=alpha)
                im = axes[x,y].imshow(np.fliplr(I[:,:,z]).T, aspect=voxsize[1]/voxsize[0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)
            axes[x,y].set_axis_off()
            axes[x,y].set_adjustable('box-forced')
            y += 1
            if y>=np.ceil(np.sqrt(imdim[2])):
                y = 0
                x += 1

        cax = fig.add_axes([0.1, 0.03, 0.8, 0.03])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=30) # colorbar legend font size

        # colorbar ticks' and labels' color
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
        cbar.ax.xaxis.set_tick_params(color='w')

        # smooth colorbar without lines in it
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
        cbar.set_alpha(1)
        cbar.draw_all() # recent addition

        fig.tight_layout()

        _, base, _ = split_filename(imgfile)
        fig.savefig(base+'_mosaic.png', format='png', facecolor=fig.get_facecolor())

        return runtime

    def _list_outputs(self):
        imgfile = self.inputs.imgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(imgfile)
        outputs['mosaic'] = os.path.abspath(base+'_mosaic.png')

        return outputs
