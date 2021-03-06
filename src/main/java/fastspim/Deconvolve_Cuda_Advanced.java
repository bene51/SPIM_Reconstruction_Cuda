package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.plugin.PlugIn;

import java.io.File;

public class Deconvolve_Cuda_Advanced implements PlugIn {

	@Override
	public void run(String arg) {
		GenericDialogPlus gd = new GenericDialogPlus("Deconvolve Cuda (Advanced)");
		gd.addNumericField("number_of_views", 2, 0);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		int nViews = (int)gd.getNextNumber();

		gd = new GenericDialogPlus("Deconvolve Cuda (Advanced)");
		for(int i = 0; i < nViews; i++)
			gd.addFileField("data_view_" + i, "");
		for(int i = 0; i < nViews; i++)
			gd.addFileField("weights_view_" + i, "");
		for(int i = 0; i < nViews; i++)
			gd.addFileField("kernel_view_" + i, "");

		gd.addFileField("output_file", "");

		gd.addNumericField("width", 0, 0);
		gd.addNumericField("height", 0, 0);
		gd.addNumericField("depth", 0, 0);
		gd.addNumericField("kernel_width", 0, 0);
		gd.addNumericField("kernel_height", 0, 0);
		String[] choice = new String[] { "INDEPENDENT", "EFFICIENT_BAYESIAN", "OPTIMIZATION_1", "OPTIMIZATION_2" };
		gd.addChoice("Iteration_type", choice, "OPTIMIZATION_1");
		gd.addNumericField("iterations", 3, 0);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		String[] datafiles = new String[nViews];
		String[] weightfiles = new String[nViews];
		String[] kernelfiles = new String[nViews];
		for(int i = 0; i < nViews; i++)
			datafiles[i] = gd.getNextString();
		for(int i = 0; i < nViews; i++)
			weightfiles[i] = gd.getNextString();
		for(int i = 0; i < nViews; i++)
			kernelfiles[i] = gd.getNextString();
		String outputfile = gd.getNextString();
		int w = (int)gd.getNextNumber();
		int h = (int)gd.getNextNumber();
		int d = (int)gd.getNextNumber();
		int kernelW = (int)gd.getNextNumber();
		int kernelH = (int)gd.getNextNumber();
		int psfType = gd.getNextChoiceIndex();
		int iterations = (int)gd.getNextNumber();
		int bitDepth = 8;
		long whd = (long)w * (long)h * d;
		if(new File(datafiles[0]).length() > whd)
			bitDepth = 16;

		IJ.log("Starting plane-wise multi-view deconvolution");
		try {
			NativeSPIMReconstructionCuda.deconvolve(datafiles, outputfile, w, h, d, weightfiles, kernelfiles, kernelH, kernelW, psfType, nViews, iterations, bitDepth);
		} catch(Exception e) {
			IJ.handleException(e);
		}
		IJ.log("done");
	}
}
