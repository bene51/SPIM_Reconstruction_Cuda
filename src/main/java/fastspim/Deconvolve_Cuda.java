package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.plugin.PlugIn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

public class Deconvolve_Cuda implements PlugIn {

	@Override
	public void run(String arg) {
		final GenericDialogPlus gd = new GenericDialogPlus("Deconvolve Cuda");
		gd.addDirectoryField("SPIM_Folder", "");
		gd.addDirectoryField("PSF_Folder", "");
		String[] choice = new String[] { "INDEPENDENT", "EFFICIENT_BAYESIAN", "OPTIMIZATION_1", "OPTIMIZATION_2" };
		gd.addChoice("Iteration_type", choice, "OPTIMIZATION_1");
		gd.addNumericField("iterations", 10, 0);

		gd.showDialog();
		if(gd.wasCanceled())
			return;

		String spimfolderString = gd.getNextString();
		String psffolderString = gd.getNextString();
		int psfType = gd.getNextChoiceIndex();
		int iterations = (int)gd.getNextNumber();
		deconvolve(spimfolderString, psffolderString, psfType, iterations);
	}

	void deconvolve(String spimfolderString, String psffolderString, int psfType, int iterations) {
		File spimfolder = new File(spimfolderString);
		File psffolder = new File(psffolderString);
		File outputfolder = new File(spimfolder, "output");
		File weightsfolder = new File(outputfolder, "masks");
		String[] fileNames = weightsfolder.list(new FilenameFilter() {
			@Override
			public boolean accept(File arg0, String arg1) {
				return arg1.endsWith(".raw");
			}
		});

		int nViews = fileNames.length;


		String[] datafiles = new String[nViews];
		String[] weightfiles = new String[nViews];
		String[] kernelfiles = new String[nViews];
		for(int i = 0; i < nViews; i++) {
			datafiles[i] = new File(outputfolder, fileNames[i]).getAbsolutePath();
			weightfiles[i] = new File(weightsfolder, fileNames[i]).getAbsolutePath();
			kernelfiles[i] = new File(psffolder, fileNames[i]).getAbsolutePath();
		}
		String outputfile = new File(spimfolder, "deconvolved.raw").getAbsolutePath();
		int[] dims = null;
		int[] psfDims = null;

		try {
			dims = readDimensions(new File(outputfolder, "dims.txt"));
			psfDims = readDimensions(new File(psffolder, "dims.txt"));
		} catch(Exception e) {
			IJ.handleException(e);
			return;
		}
		int w = dims[0];
		int h = dims[1];
		int d = dims[2];
		int kernelW = psfDims[0];
		int kernelH = psfDims[1];

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

	private final int[] readDimensions(File file) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(file));
		ArrayList<String> lines = new ArrayList<String>(3);
		String line = null;
		while((line = in.readLine()) != null)
			lines.add(line);
		in.close();

		int[] ret = new int[lines.size()];
		for(int i = 0; i < ret.length; i++)
			ret[i] = Integer.parseInt(lines.get(i));

		return ret;
	}
}
