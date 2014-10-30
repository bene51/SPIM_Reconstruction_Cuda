package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Transform_Cuda implements PlugIn {

	// TODO check whether output directory exists.
	@Override
	public void run(String arg) {
		GenericDialogPlus gd = new GenericDialogPlus("Transform_Cuda");
		gd.addDirectoryField("SPIM_directory", "");
		// gd.addCheckbox("use_cuda", true);
		gd.addNumericField("offset_x", 0, 0);
		gd.addNumericField("offset_y", 0, 0);
		gd.addNumericField("offset_z", 0, 0);
		gd.addNumericField("size_x", 0, 0);
		gd.addNumericField("size_y", 0, 0);
		gd.addNumericField("size_z", 0, 0);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		File spimdir = new File(gd.getNextString());
		int[] offset = new int[3];
		int[] size = new int[3];
		offset[0] = (int)gd.getNextNumber();
		offset[1] = (int)gd.getNextNumber();
		offset[2] = (int)gd.getNextNumber();
		size[0] = (int)gd.getNextNumber();
		size[1] = (int)gd.getNextNumber();
		size[2] = (int)gd.getNextNumber();
		boolean useCuda = true;
		boolean rotateX = true;

		try {
			transform(spimdir, offset, size, rotateX, useCuda);
		} catch(Exception e) {
			IJ.handleException(e);
		}
	}

	public static void transform(File spimdir, int[] offset, int[] size, boolean rotateX, boolean useCuda) throws IOException {
		File registrationdir = new File(spimdir, "registration");
		String[] names = registrationdir.list(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {
				return name.endsWith(".dim");
			}
		});
		for(int i = 0; i < names.length; i++)
			names[i] = names[i].substring(0, names[i].length() - 4);

		transform(spimdir, names, offset, size, rotateX, useCuda);
	}

	private static void writeDims(File outfile, int w, int h, int d) {
		try {
			PrintStream out = new PrintStream(new FileOutputStream(outfile));
			out.println(w);
			out.println(h);
			out.println(d);
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void transform(File spimdir, String[] names, int[] offset, int[] size, boolean rotateX, boolean useCuda) throws IOException {
		File registrationdir = new File(spimdir, "registration");
		File outdir = new File(spimdir, "output");
		if(!outdir.exists())
			outdir.mkdirs();
		int[][] dims = new int[names.length][];
		float[][] inverseMatrices = new float[names.length][12];
		float[] zspacing = new float[names.length];
		float[] max = new float[] { Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY };
		float[] min = new float[] { Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY };
		float[] res = new float[3];
		for(int i = 0; i < names.length; i++) {
			System.out.println("analyzing " + names[i]);
			dims[i] = readDims(new File(registrationdir, names[i] + ".dim"));
			zspacing[i] = readTransformation(new File(registrationdir, names[i] + ".registration"), inverseMatrices[i]);

			int w = dims[i][0], h = dims[i][1], d = dims[i][2];
			apply(inverseMatrices[i], 0, 0, 0, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], w, 0, 0, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], w, h, 0, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], 0, h, 0, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], 0, 0, d, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], w, 0, d, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], w, h, d, res); min(res, min); max(res, max);
			apply(inverseMatrices[i], 0, h, d, res); min(res, min); max(res, max);
		}
		System.out.println("min: " + Arrays.toString(min));
		System.out.println("max: " + Arrays.toString(max));
		/*
		int tw = (int)(max[0] - min[0] + 0.5);
		int th = (int)(max[1] - min[1] + 0.5);
		int td = (int)(max[2] - min[2] + 0.5);
		*/
		int tw, th, td;

		if(size[0] == 0)
			tw = (int)Math.ceil(max[0] - min[0]) + 1;
		else
			tw = size[0];

		if(size[1] == 0)
			th = (int)Math.ceil(max[1] - min[1]) + 1;
		else
			th = size[1];

		if(size[2] == 0)
			td = (int)Math.ceil(max[2] - min[2]) + 1;
		else
			td = size[2];

		min[0] += offset[0];
		min[1] += offset[1];
		min[2] += offset[2];

		if(rotateX)
			writeDims(new File(outdir, "dims.txt"), tw, td, th);
		else
			writeDims(new File(outdir, "dims.txt"), tw, th, td);

		boolean createTransformedMasks = true;
		int border = 30;
		File maskdir = new File(outdir, "masks");
		if(!maskdir.exists())
			maskdir.mkdir();

		for(int i = 0; i < names.length; i++) {
			invert(inverseMatrices[i]);
			inverseMatrices[i][3]  += (min[0] * inverseMatrices[i][0] + min[1] * inverseMatrices[i][1] + min[2] * inverseMatrices[i][2]);
			inverseMatrices[i][7]  += (min[0] * inverseMatrices[i][4] + min[1] * inverseMatrices[i][5] + min[2] * inverseMatrices[i][6]);
			inverseMatrices[i][11] += (min[0] * inverseMatrices[i][8] + min[1] * inverseMatrices[i][9] + min[2] * inverseMatrices[i][10]);

			int targetD = td;
			int targetH = th;
			int targetW = tw;
			if(rotateX) {
				float[] rotx = new float[] {
						1, 0, 0, 0,
						0, 0, -1, targetD - 1,
						0, 1, 0, 0,
				};
				invert(rotx);
				inverseMatrices[i] = mul(inverseMatrices[i], rotx);
				targetH = td;
				targetD = th;
			}

			String outname = names[i];
			if(!outname.endsWith("raw"))
				outname += ".raw";

			File maskfile = new File(maskdir, outname);
			transform(
					new File(spimdir, names[i]),
					new File(outdir, outname),
					inverseMatrices[i], targetW, targetH, targetD,
					createTransformedMasks,
					maskfile,
					border,
					zspacing[i],
					useCuda);
		}
	}

	public static void transform(File infile, File outfile, float[] inverseMatrix, int targetW, int targetH, int targetD, boolean createTransformedMask, File maskfile, int border, float zspacing, boolean useCuda) {
		long start = System.currentTimeMillis();
		ImagePlus imp = IJ.openImage(infile.getAbsolutePath());
		long end = System.currentTimeMillis();
		System.out.println("opening took " + (end - start) + " ms");
		if(useCuda) {
			if(imp.getType() != ImagePlus.GRAY16)
				throw new RuntimeException("Only 16-bit grayscale images supported for CUDA");
			short[][] data = new short[imp.getStackSize()][];
			for(int z = 0; z < data.length; z++)
				data[z] = (short[])imp.getStack().getPixels(z + 1);
			NativeSPIMReconstructionCuda.transform(
					data,
					imp.getWidth(),
					imp.getHeight(),
					data.length,
					inverseMatrix,
					targetW,
					targetH,
					targetD,
					outfile.getAbsolutePath(),
					createTransformedMask,
					border,
					zspacing,
					maskfile.getAbsolutePath());
		} else {
			ImagePlus xformed = transform(imp, inverseMatrix, targetW, targetH, targetD);
			IJ.save(xformed, outfile.getAbsolutePath() + ".tif");
		}
	}

	public static ImagePlus transform(ImagePlus in, final float[] inv, final int w, final int h, final int d) {
		final ImageProcessor[] inProcessors = new ImageProcessor[in.getStackSize()];
		for(int z = 0; z < inProcessors.length; z++)
			inProcessors[z] = in.getStack().getProcessor(z + 1);

		final int wIn = in.getWidth();
		final int hIn = in.getHeight();
		final int dIn = in.getStackSize();

		final ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		ImageStack outStack = new ImageStack(w, h);
		for(int z = 0; z < d; z++)
			outStack.addSlice(inProcessors[0].createProcessor(w, h));

		for(int iz = 0; iz < d; iz++) {
			final int z = iz;
			final ImageProcessor ip = outStack.getProcessor(z + 1);
			exec.submit(new Runnable() {
				@Override
				public void run() {
					try {
						float[] result = new float[3];
						for(int y = 0, xy = 0; y < h; y++) {
							for(int x = 0; x < w; x++, xy++) {
								apply(inv, x, y, z, result);
								if(x == 213 && y == 36 && z == 833)
									System.out.println("cpu: (" + x + ", " + y + ", " + z + ") -> (" +
											result[0] + ", " + result[1] + ", " + result[2] + ")");
								if(result[0] < 0 || result[1] < 0 || result[2] < 0 ||
										result[0] >= wIn - 1 || result[1] >= hIn - 1 || result[2] >= dIn - 1) {
									ip.setf(xy, 0);
									continue;
								}
								int lx = (int)result[0];
								int ly = (int)result[1];
								int lz = (int)result[2];
								float xR = 1 + lx - result[0];
								float yR = 1 + ly - result[1];
								float zR = 1 + lz - result[2];

								float v000 =     inProcessors[lz].getf(lx,     ly);
								float v001 = inProcessors[lz + 1].getf(lx,     ly);
								float v010 =     inProcessors[lz].getf(lx,     ly + 1);
								float v011 = inProcessors[lz + 1].getf(lx,     ly + 1);
								float v100 =     inProcessors[lz].getf(lx + 1, ly);
								float v101 = inProcessors[lz + 1].getf(lx + 1, ly);
								float v110 =     inProcessors[lz].getf(lx + 1, ly + 1);
								float v111 = inProcessors[lz + 1].getf(lx + 1, ly + 1);

								float ret = xR * (yR * (zR * v000 + (1 - zR) * v001)
										+ (1 - yR) * (zR * v010 + (1 - zR) * v011))
										+ (1 - xR) * (yR * (zR * v100 + (1 - zR) * v101)
										+ (1 - yR) * (zR * v110 + (1 - zR) * v111));

								if(x == 213 && y == 36 && z == 833) {
									System.out.println("\tlower: (" + lx + ", " + ly + ", " + lz + ")");
									System.out.println("\tresult: " + ret);
								}
								ip.setf(xy, ret);
							}
						}
					} catch(Exception e) {
						e.printStackTrace();
					}
				}
			});
		}
		exec.shutdown();
		try {
			exec.awaitTermination(1, TimeUnit.DAYS);
		} catch(Exception e) {
			e.printStackTrace();
		}
		return new ImagePlus(in.getTitle() + "-transformed", outStack);
	}

	private static void invert3x3(float[] mat) {
		double sub00 = mat[5] * mat[10] - mat[6] * mat[9];
		double sub01 = mat[4] * mat[10] - mat[6] * mat[8];
		double sub02 = mat[4] * mat[9]  - mat[5] * mat[8];
		double sub10 = mat[1] * mat[10] - mat[2] * mat[9];
		double sub11 = mat[0] * mat[10] - mat[2] * mat[8];
		double sub12 = mat[0] * mat[9]  - mat[1] * mat[8];
		double sub20 = mat[1] * mat[6]  - mat[2] * mat[5];
		double sub21 = mat[0] * mat[6]  - mat[2] * mat[4];
		double sub22 = mat[0] * mat[5]  - mat[1] * mat[4];
		double det = mat[0] * sub00 - mat[1] * sub01 + mat[2] * sub02;

		mat[0]  =  (float)(sub00 / det);
		mat[1]  = -(float)(sub10 / det);
		mat[2]  =  (float)(sub20 / det);
		mat[4]  = -(float)(sub01 / det);
		mat[5]  =  (float)(sub11 / det);
		mat[6]  = -(float)(sub21 / det);
		mat[8]  =  (float)(sub02 / det);
		mat[9]  = -(float)(sub12 / det);
		mat[10] =  (float)(sub22 / det);
	}

	private static void invert(float[] mat) {
		float dx = -mat[3];
		float dy = -mat[7];
		float dz = -mat[11];
		invert3x3(mat);

		mat[3]  = mat[0] * dx + mat[1] * dy + mat[2]  * dz;
		mat[7]  = mat[4] * dx + mat[5] * dy + mat[6]  * dz;
		mat[11] = mat[8] * dx + mat[9] * dy + mat[10] * dz;
	}


	private static void apply(float[] mat, float x, float y, float z, float[] result) {
		result[0] = mat[0] * x + mat[1] * y + mat[2]  * z + mat[3];
		result[1] = mat[4] * x + mat[5] * y + mat[6]  * z + mat[7];
		result[2] = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];
	}

	private static float[] mul(float[] m1, float[] m2) {
		float[] res = new float[12];
		res[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8];
		res[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9];
		res[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10];
		res[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3];

		res[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8];
		res[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9];
		res[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10];
		res[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7];

		res[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8];
		res[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9];
		res[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10];
		res[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11];

		return res;
	}

	private static void min(float[] x, float[] min)	{
		for(int i = 0; i < x.length; i++)
			if(x[i] < min[i])
				min[i] = x[i];
	}

	private static void max(float[] x, float[] max)	{
		for(int i = 0; i < x.length; i++)
			if(x[i] > max[i])
				max[i] = x[i];
	}

	private static float readTransformation(File path, float[] ret) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(path));
		float[] m = ret;
		for(int i = 0; i < 12; i++) {
			String line = in.readLine();
			String[] toks = line.split(":");
			m[i] = (float)Double.parseDouble(toks[1].trim());
		}
		String line = null;
		double dz = 1;
		while((line = in.readLine()) != null) {
			if(line.startsWith("z-scaling")) {
				String[] toks = line.split(":");
				dz = Double.parseDouble(toks[1]);
			}
		}
		in.close();
		m[2]  *= dz;
		m[6]  *= dz;
		m[10] *= dz;
		return (float)dz;
	}

	private static int[] readDims(File path) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(path));
		int[] d = new int[3];
		for(int i = 0; i < 3; i++) {
			String line = in.readLine();
			String[] tok = line.split(": ");
			d[i] = Integer.parseInt(tok[1]);
		}
		in.close();
		return d;
	}
}
