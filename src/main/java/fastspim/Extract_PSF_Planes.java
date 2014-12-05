package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class Extract_PSF_Planes implements PlugIn {

	public static final int X_AXIS = 0;
	public static final int Y_AXIS = 1;
	public static final int Z_AXIS = 2;

	@Override
	public void run(String arg) {
		GenericDialogPlus gd = new GenericDialogPlus("Extract PSF Planes");
		gd.addDirectoryField("SPIM_directory", "");
		gd.addStringField("Pattern of SPIM files", "");
		gd.addStringField("Angles to process", "");
		String[] axes = new String[] { "x-axis", "y-axis", "z-axis" };
		gd.addChoice("Reslice result", axes, axes[1]);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		File spimdir = new File(gd.getNextString());
		String pattern = gd.getNextString();
		String angles = gd.getNextString();
		int rotAxis = gd.getNextChoiceIndex();

		try {
			extractPSFPlanes(spimdir, pattern, angles, rotAxis);
		} catch(Exception e) {
			IJ.handleException(e);
		}
	}

	static void saveAsRaw(FloatProcessor ip, File file) throws IOException {
		int w = ip.getWidth();
		int h = ip.getHeight();
		int wh = w * h;

		ByteBuffer bbuf = ByteBuffer.allocateDirect(4 * wh).order(ByteOrder.LITTLE_ENDIAN);
		bbuf.asFloatBuffer().put((float[])ip.getPixels());

		FileOutputStream fos = new FileOutputStream(file);
		FileChannel channel = fos.getChannel();
		while(bbuf.hasRemaining())
			channel.write(bbuf);

		fos.close();
	}

	public static FloatProcessor extractMiddlePlane(ImagePlus imp, int rotAxis) {
		switch(rotAxis) {
		case X_AXIS: return extractMiddlePlaneX(imp);
		case Y_AXIS: return extractMiddlePlaneY(imp);
		case Z_AXIS: return extractMiddlePlaneZ(imp);
		}
		throw new RuntimeException("Invalid rotation axis: " + rotAxis);
	}

	public static FloatProcessor extractMiddlePlaneY(ImagePlus imp) {
		int d = imp.getStackSize();
		FloatProcessor ip = new FloatProcessor(imp.getWidth(), d);
		int y = imp.getHeight() / 2 + 1;
		for(int z = 0; z < d; z++)
			for(int x = 0; x < imp.getWidth(); x++)
				ip.setf(x, z, imp.getStack().getProcessor(d - z).getf(x, y));
		return ip;
	}

	public static FloatProcessor extractMiddlePlaneX(ImagePlus imp) {
		int d = imp.getStackSize();
		FloatProcessor ip = new FloatProcessor(d, imp.getHeight());
		int x = imp.getWidth() / 2 + 1;
		for(int y = 0; y < imp.getHeight(); y++)
			for(int z = 0; z < d; z++)
				ip.setf(z, y, imp.getStack().getProcessor(z + 1).getf(x, y));
		return ip;
	}

	public static FloatProcessor extractMiddlePlaneZ(ImagePlus imp) {
		int z = imp.getStackSize() / 2 + 2;
		return imp.getStack().getProcessor(z).convertToFloatProcessor();
	}

	public static void extractPSFPlanes(File spimdir, String pattern, String angles, int rotAxis) throws IOException {
		String dir = spimdir.getAbsolutePath().replaceAll("\\\\", "/");
		File psfdir = new File(spimdir, "psfs");
		if(!psfdir.exists())
			psfdir.mkdirs();
		fiji.plugin.Multi_View_Deconvolution.makeAllPSFSameSize = true;
		String macro =
				"run(\"Multi-view deconvolution\",\n" +
				"\"spim_data_directory=[" + dir + "] \" + \n" +
				"\"pattern_of_spim=[" + pattern + "] \" + \n" +
				"\"timepoints_to_process=1 \" + \n" +
				"\"angles=[" + angles + "] \" + \n" +
				"\"imglib_container_(input=[Array container (images smaller ~2048x2048x450 px)] \" + \n" +
				"\"imglib_container_(processing)=[Array container (images smaller ~2048x2048x450 px)] \" + \n" +
				"\"registration=[Individual registration of channel 0] \" + \n" +
				"\"crop_output_image_offset_x=0 \" + \n" +
				"\"crop_output_image_offset_y=0 \" + \n" +
				"\"crop_output_image_offset_z=0 \" + \n" +
				"\"crop_output_image_size_x=1 \" + \n" +
				"\"crop_output_image_size_y=1 \" + \n" +
				"\"crop_output_image_size_z=1 \" + \n" +
				"\"type_of_iteration=[Efficient Bayesian - Optimization I (fast, precise)] \" + \n" +
				"\"osem_acceleration=[1 (balanced)] \" + \n" +
				"\"number_of_iterations=0 \" + \n" +
				"\"tikhonov_parameter=0.0060 \" + \n" +
				"\"compute=[Entire image at once] \" + \n" +
				"\"compute_on=[CPU (Java)] \" + \n" +
				"\"psf_estimation=[Extract from beads] \" + \n" +
				"\"psf_display=[Show individual PSF's] \" + \n" +
				"\"load_input_images_sequentially \" + \n" +
				"\"fused_image_output=[Display only]\"); \n" +
				"close(); \n";
		IJ.runMacro(macro);
		int[] ids = WindowManager.getIDList();
		int w = -1, h = -1;
		for(int id : ids) {
			ImagePlus imp = WindowManager.getImage(id);
			if(!imp.getTitle().startsWith("PSF"))
				continue;
			FloatProcessor psfPlane = extractMiddlePlane(imp, rotAxis);
			if(w != -1) {
				if(w != psfPlane.getWidth() || h != psfPlane.getHeight())
					throw new RuntimeException("PSFs have different sizes");
			} else {
				w = psfPlane.getWidth();
				h = psfPlane.getHeight();
			}
			File out = new File(psfdir, imp.getTitle().substring(8) + ".raw");
			// IJ.save(psfPlane, out.getAbsolutePath());
			saveAsRaw(psfPlane, out);
			imp.close();
		}
		IJ.saveString(w + "\n" + h, new File(psfdir, "dims.txt").getAbsolutePath());
	}
}
