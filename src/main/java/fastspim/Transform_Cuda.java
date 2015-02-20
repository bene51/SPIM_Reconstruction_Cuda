package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileInfo;
import ij.io.FileOpener;
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

	public static final int NO_RESLICE = 0;
	public static final int FROM_TOP   = 1;
	public static final int FROM_RIGHT = 2;

	@Override
	public void run(String arg) {
		int nCudaDevices = NativeSPIMReconstructionCuda.getNumCudaDevices();
		String[] devices = new String[nCudaDevices];
		for(int i = 0; i < nCudaDevices; i++)
			devices[i] = NativeSPIMReconstructionCuda.getCudaDeviceName(i);

		GenericDialogPlus gd = new GenericDialogPlus("Transform Cuda");
		gd.addDirectoryField("SPIM_directory", "");
		gd.addNumericField("offset_x", 0, 0);
		gd.addNumericField("offset_y", 0, 0);
		gd.addNumericField("offset_z", 0, 0);
		gd.addNumericField("size_x", 0, 0);
		gd.addNumericField("size_y", 0, 0);
		gd.addNumericField("size_z", 0, 0);
		gd.addNumericField("spacing_x", 1, 4);
		gd.addNumericField("spacing_y", 1, 4);
		gd.addNumericField("spacing_z", 1, 4);
		gd.addCheckbox("Create_weights", true);
		String[] resliceChoice = new String[3];
		resliceChoice[NO_RESLICE] = "No";
		resliceChoice[FROM_TOP] = "From top";
		resliceChoice[FROM_RIGHT] = "From left";
		gd.addChoice("Reslice result", resliceChoice, resliceChoice[FROM_TOP]);
		gd.addChoice("Device", devices, devices[0]);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		File spimdir = new File(gd.getNextString());
		int[] offset = new int[3];
		int[] size = new int[3];
		float[] spacing = new float[3];
		offset[0] = (int)gd.getNextNumber();
		offset[1] = (int)gd.getNextNumber();
		offset[2] = (int)gd.getNextNumber();
		size[0] = (int)gd.getNextNumber();
		size[1] = (int)gd.getNextNumber();
		size[2] = (int)gd.getNextNumber();
		spacing[0] = (float)gd.getNextNumber();
		spacing[1] = (float)gd.getNextNumber();
		spacing[2] = (float)gd.getNextNumber();

		boolean createWeights = gd.getNextBoolean();
		int reslice = gd.getNextChoiceIndex();
		int device = gd.getNextChoiceIndex();

		boolean useCuda = true;

		try {
			NativeSPIMReconstructionCuda.setCudaDevice(device);
			transform(spimdir, offset, size, spacing, reslice, createWeights, useCuda);
		} catch(Exception e) {
			IJ.handleException(e);
		}
	}

	public static ImagePlus openAndTurnBack(File file, int[] dims, int reslice) {
		FileInfo fi = new FileInfo();
		fi.fileFormat = FileInfo.RAW;
		fi.fileName = file.getName();
		fi.directory = file.getParent();
		fi.width = dims[0];
		fi.height = dims[1];
		fi.offset = 0;

		fi.nImages = dims[2];
		fi.gapBetweenImages = 0;
		fi.intelByteOrder = true;
		fi.whiteIsZero = false;
		fi.fileType = FileInfo.GRAY16_UNSIGNED;

		FileOpener fo = new FileOpener(fi);
		ImagePlus imp = fo.open(false);
		turnBack(imp, reslice);
		return imp;
	}

	public static void turnBack(ImagePlus imp, int reslice) {
		if(reslice == NO_RESLICE)
			return;
		ImageProcessor[] ips = new ImageProcessor[imp.getStackSize()];
		for(int z = 0; z < ips.length; z++)
			ips[z] = imp.getStack().getProcessor(z + 1);
		int wOld = imp.getWidth();
		int hOld = imp.getHeight();
		int dOld = imp.getStackSize();
		if(reslice == FROM_TOP) {
			int wNew = wOld;
			int hNew = dOld;
			int dNew = hOld;
			ImagePlus ret = IJ.createImage(imp.getTitle(), wNew, hNew, dNew, imp.getBitDepth());
			for(int z = 0; z < dNew; z++) {
				ImageProcessor ip = ret.getStack().getProcessor(z + 1);
				int yOld = hOld - z - 1;
				for(int y = 0, idx = 0; y < hNew; y++) {
					int zOld = y;
					for(int x = 0; x < wNew; x++, idx++) {
						try {
							ip.setf(idx, ips[zOld].getf(x, yOld));
						} catch(Exception e) {
							System.out.println("x = " + x + " y = " + y + " z = " + z);
							throw new RuntimeException(e);
						}
					}
				}
			}
			imp.setStack(ret.getStack());
		} else if(reslice == FROM_RIGHT) {
			int wNew = dOld;
			int hNew = hOld;
			int dNew = wOld;
			ImagePlus ret = IJ.createImage(imp.getTitle(), wNew, hNew, dNew, imp.getBitDepth());
			for(int z = 0; z < dNew; z++) {
				ImageProcessor ip = ret.getStack().getProcessor(z + 1);
				int xOld = z;
				for(int y = 0, idx = 0; y < hNew; y++) {
					for(int x = 0; x < wNew; x++, idx++) {
						int zOld = dOld - x - 1;
						ip.setf(idx, ips[zOld].getf(xOld, y));
					}
				}
			}
		}
	}

	public static void transform(File spimdir, int[] offset, int[] size, float[] pw, int reslice, boolean createWeights, boolean useCuda) throws IOException {
		File registrationdir = new File(spimdir, "registration");
		String[] names = registrationdir.list(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {
				return name.endsWith(".dim");
			}
		});
		for(int i = 0; i < names.length; i++)
			names[i] = names[i].substring(0, names[i].length() - 4);

		transform(spimdir, names, offset, size, pw, reslice, createWeights, useCuda);
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

	public static void transform(File spimdir, String[] names, int[] offset, int[] size, float[] pw, int reslice, boolean createWeights, boolean useCuda) throws IOException {
		File registrationdir = new File(spimdir, "registration");
		File outdir = new File(spimdir, "output");
		File[] images = new File[names.length];
		File[] dims = new File[names.length];
		File[] registrations = new File[names.length];

		for(int i = 0; i < names.length; i++) {
			images[i] = new File(spimdir, names[i]);
			dims[i] = new File(registrationdir, names[i] + ".dim");
			registrations[i] = new File(registrationdir, names[i] + ".registration");
		}
		transform(images, dims, registrations, offset, size, pw, reslice, createWeights, useCuda, outdir);
	}

	public static void transform(
			File[] images,
			File[] dims,
			File[] registrations,
			int[] offset,
			int[] size,
			float[] pw,
			int reslice,
			boolean createWeights,
			boolean useCuda,
			File outdir) throws IOException {

		int n = images.length;
		int[][] dimensions = new int[n][3];
		float[][] matrices = new float[n][12];
		float[] zspacings = new float[n];
		for(int i = 0; i < images.length; i++) {
			dimensions[i] = readDims(dims[i]);
			zspacings[i] = readTransformation(registrations[i], matrices[i]);
		}
		transform(images, dimensions, matrices, zspacings, offset, size, pw, reslice, createWeights, useCuda, outdir);
	}


	/**
	 * Calculates the transformation matrices, and also fills the <code>size</code> array
	 * if all entries are 0.
	 *
	 * @param dims
	 * @param matrices
	 * @param zspacing
	 * @param offset
	 * @param size
	 * @param rotateX
	 * @return
	 */
	public static  void createRealTransformationMatrices(
			int[][] dims,
			float[][] matrices,
			int[] offset,
			int[] size,
			float[] pw,
			int reslice) {

		int n = matrices.length;

		float[] max = new float[] { Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY };
		float[] min = new float[] { Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY };
		float[] res = new float[3];
		for(int i = 0; i < n; i++) {
			int w = dims[i][0], h = dims[i][1], d = dims[i][2];
			apply(matrices[i], 0, 0, 0, res); min(res, min); max(res, max);
			apply(matrices[i], w, 0, 0, res); min(res, min); max(res, max);
			apply(matrices[i], w, h, 0, res); min(res, min); max(res, max);
			apply(matrices[i], 0, h, 0, res); min(res, min); max(res, max);
			apply(matrices[i], 0, 0, d, res); min(res, min); max(res, max);
			apply(matrices[i], w, 0, d, res); min(res, min); max(res, max);
			apply(matrices[i], w, h, d, res); min(res, min); max(res, max);
			apply(matrices[i], 0, h, d, res); min(res, min); max(res, max);
		}
		System.out.println("min: " + Arrays.toString(min));
		System.out.println("max: " + Arrays.toString(max));

		if(size[0] == 0)
			size[0] = (int)Math.ceil(max[0] - min[0]) + 1;

		if(size[1] == 0)
			size[1] = (int)Math.ceil(max[1] - min[1]) + 1;

		if(size[2] == 0)
			size[2] = (int)Math.ceil(max[2] - min[2]) + 1;

		size[0] = Math.round(size[0] / pw[0]);
		size[1] = Math.round(size[1] / pw[1]);
		size[2] = Math.round(size[2] / pw[2]);

		min[0] += offset[0];
		min[1] += offset[1];
		min[2] += offset[2];

		float[] scaleMatrix = new float[] {pw[0], 0, 0, 0, 0, pw[1], 0, 0, 0, 0, pw[2], 0, 0, 0, 0, 1};

		if(reslice == FROM_TOP) {
			int tmp = size[2];
			size[2] = size[1];
			size[1] = tmp;
		} else if(reslice == FROM_RIGHT) {
			int tmp = size[2];
			size[2] = size[0];
			size[0] = tmp;
		}

		for(int i = 0; i < n; i++) {
			invert(matrices[i]);
			matrices[i][3]  += (min[0] * matrices[i][0] + min[1] * matrices[i][1] + min[2] * matrices[i][2]);
			matrices[i][7]  += (min[0] * matrices[i][4] + min[1] * matrices[i][5] + min[2] * matrices[i][6]);
			matrices[i][11] += (min[0] * matrices[i][8] + min[1] * matrices[i][9] + min[2] * matrices[i][10]);

			matrices[i] = mul(matrices[i], scaleMatrix);


			if(reslice == FROM_TOP) {  // rotate 90º around x axis
				float[] rotx = new float[] {
						1, 0, 0, 0,
						0, 0, -1, size[1] - 1,
						0, 1, 0, 0,
				};
				invert(rotx);
				matrices[i] = mul(matrices[i], rotx);
			}
			else if(reslice == FROM_RIGHT) {  // rotate 90ª around y axis
				float[] roty = new float[] {
						0, 0, 1, 0,
						0, 1, 0, 0,
						-1, 0, 0, size[2] - 1,
				};
				invert(roty);
				matrices[i] = mul(matrices[i], roty);
			}
		}
	}

	public static void transform(
			File[] images,
			int[][] dims,
			float[][] matrices,
			float[] zspacing,
			int[] offset,
			int[] size,
			float[] pw,
			int reslice,
			boolean createWeights,
			boolean useCuda,
			File outdir) {

		if(!outdir.exists())
			outdir.mkdirs();


		createRealTransformationMatrices(dims, matrices, offset, size, pw, reslice);

		writeDims(new File(outdir, "dims.txt"), size[0], size[1], size[2]);

		int border = 30;
		File maskdir = new File(outdir, "masks");
		if(createWeights && !maskdir.exists())
			maskdir.mkdir();

		for(int i = 0; i < images.length; i++) {
			String outname = images[i].getName();
			if(!outname.endsWith("raw"))
				outname += ".raw";

			File maskfile = new File(maskdir, outname);
			transform(
					images[i],
					new File(outdir, outname),
					matrices[i], size[0], size[1], size[2],
					createWeights,
					maskfile,
					border,
					zspacing[i],
					useCuda);
		}
	}

	public static void transform(
			File infile,
			File outfile,
			float[] inverseMatrix,
			int targetW,
			int targetH,
			int targetD,
			boolean createTransformedMask,
			File maskfile,
			int border,
			float zspacing,
			boolean useCuda) {

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
