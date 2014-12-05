package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class Extract_PSF_Planes implements PlugIn {

	@Override
	public void run(String arg) {
		GenericDialogPlus gd = new GenericDialogPlus("Extract PSF Planes");
		gd.addDirectoryField("SPIM_directory", "");
		gd.addStringField("pattern_of_spim", "");
		gd.addStringField("angles", "");
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		File spimdir = new File(gd.getNextString());
		String pattern = gd.getNextString();
		String angles = gd.getNextString();

		try {
			extractPSFPlanes(spimdir, pattern, angles);
		} catch(Exception e) {
			IJ.handleException(e);
		}
	}

	public static void saveAsRaw(ImagePlus imp, File file) throws IOException {
		int w = imp.getWidth();
		int h = imp.getHeight();
		int d = imp.getStackSize();
		int wh = w * h;

		DataOutputStream out = new DataOutputStream(
				new BufferedOutputStream(
					new FileOutputStream(file)));

		for(int z = 0; z < d; z++) {
			ImageProcessor ip = imp.getStack().getProcessor(z + 1);
			for(int i = 0; i < wh; i++) {
				float v = ip.getf(i);
				out.writeFloat(v);
			}
		}
		out.close();
	}

	// TODO this just works as desired if the y axis is the rotation axis;
	public static ImageProcessor extractMiddlePlane(ImagePlus imp) {
		int d = imp.getStackSize();
		ImageProcessor ip = new FloatProcessor(imp.getWidth(), d);
		int y = imp.getHeight() / 2 + 1;
		for(int z = 0; z < d; z++)
			for(int x = 0; x < imp.getWidth(); x++)
				ip.setf(x, z, imp.getStack().getProcessor(d - z).getf(x, y));
		return ip;
	}

	// TODO check with spaces in path name
	// TODO make sure Intel byte order is used.
	// TODO make sure all PSFs have the same size.
	public static void extractPSFPlanes(File spimdir, String pattern, String angles) throws IOException {
		String dir = spimdir.getAbsolutePath().replaceAll("\\\\", "/");
		File psfdir = new File(spimdir, "psfs");
		if(!psfdir.exists())
			psfdir.mkdirs();
		String macro =
				"run(\"Multi-view deconvolution\",\n" +
				"\"spim_data_directory=" + dir + " \" + \n" +
				"\"pattern_of_spim=" + pattern + " \" + \n" +
				"\"timepoints_to_process=1 \" + \n" +
				"\"angles=" + angles + " \" + \n" +
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
			ImagePlus psfPlane = new ImagePlus("", extractMiddlePlane(imp));
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
