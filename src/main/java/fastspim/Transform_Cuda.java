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
import java.util.ArrayList;
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

	public static void writeDims(File outfile, int w, int h, int d) {
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
		File[] beads = new File[names.length];

		for(int i = 0; i < names.length; i++) {
			images[i] = new File(spimdir, names[i]);
			dims[i] = new File(registrationdir, names[i] + ".dim");
			registrations[i] = new File(registrationdir, names[i] + ".registration");
			beads[i] = new File(registrationdir, names[i] + ".beads.txt");
		}
		transform(images, beads, dims, registrations, offset, size, pw, reslice, createWeights, useCuda, outdir);
	}

	public static void transform(
			File[] images,
			File[] beads,
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
		transform(images, beads, dimensions, matrices, zspacings, offset, size, pw, reslice, createWeights, useCuda, outdir);
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
		createRealTransformationMatrices(dims, matrices, offset, size, pw, reslice, false);
	}

	public static  void createRealTransformationMatrices(
			int[][] dims,
			float[][] matrices,
			int[] offset,
			int[] size,
			float[] pw,
			int reslice,
			boolean dimensionsFromFirst) {

		int n = matrices.length;

		float[] max = new float[] { Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY };
		float[] min = new float[] { Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY };
		float[] res = new float[3];
		for(int i = 0; i < n; i++) {
			float w = dims[i][0], h = dims[i][1], d = dims[i][2];
			MatrixUtils.apply(matrices[i], 0, 0, 0, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], w, 0, 0, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], w, h, 0, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], 0, h, 0, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], 0, 0, d, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], w, 0, d, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], w, h, d, res); min(res, min); max(res, max);
			MatrixUtils.apply(matrices[i], 0, h, d, res); min(res, min); max(res, max);
			if(dimensionsFromFirst)
				break;
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
			// mat^(-1) * T_min * spacing * turn,
			// applied to (x,y,z) of output image
			MatrixUtils.invert(matrices[i]);
			matrices[i][3]  += (min[0] * matrices[i][0] + min[1] * matrices[i][1] + min[2] * matrices[i][2]);
			matrices[i][7]  += (min[0] * matrices[i][4] + min[1] * matrices[i][5] + min[2] * matrices[i][6]);
			matrices[i][11] += (min[0] * matrices[i][8] + min[1] * matrices[i][9] + min[2] * matrices[i][10]);

			matrices[i] = MatrixUtils.mul(matrices[i], scaleMatrix);


			if(reslice == FROM_TOP) {  // rotate 90º around x axis
				float[] rotx = new float[] {
						1, 0, 0, 0,
						0, 0, -1, size[1] - 1,
						0, 1, 0, 0,
				};
				MatrixUtils.invert(rotx);
				matrices[i] = MatrixUtils.mul(matrices[i], rotx);
			}
			else if(reslice == FROM_RIGHT) {  // rotate 90ª around y axis
				float[] roty = new float[] {
						0, 0, 1, 0,
						0, 1, 0, 0,
						-1, 0, 0, size[2] - 1,
				};
				MatrixUtils.invert(roty);
				matrices[i] = MatrixUtils.mul(matrices[i], roty);
			}
		}
	}

	public static void transform(
			File[] images,
			File[] beads,
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

		ImagePlus[] imps = new ImagePlus[images.length];
		for(int i = 0; i < images.length; i++) {
			imps[i] = IJ.openImage(images[i].getAbsolutePath());
			imps[i].setTitle(images[i].getName());
		}
		transform(imps,
				beads,
				dims,
				matrices,
				zspacing,
				offset,
				size,
				pw,
				reslice,
				createWeights,
				useCuda,
				outdir);
	}

	public static void transform(
			ImagePlus[] images,
			File[] beads,
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

		File registrationOut = new File(outdir, "registration");
		if(beads != null)
			registrationOut.mkdir();

		for(int i = 0; i < images.length; i++) {
			String outname = images[i].getFileInfo().fileName;
			if(outname.equals("Untitled"))
				outname = images[i].getTitle();
			IJ.log("Transforming " + outname);
			if(!outname.endsWith("raw"))
				outname += ".raw";

			System.out.println("outname = " + outname);
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

			if(beads == null)
				continue;

			try {
				float[] fwd = new float[12];
				System.arraycopy(matrices[i], 0, fwd, 0, 12);
				MatrixUtils.invert(fwd);
				transform(beads[i], new File(registrationOut, beads[i].getName()), fwd);
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
		IJ.log("Done");
	}

	public static void transform(File beadsIn, File beadsOut, float[] mat) throws IOException {
		ArrayList<Bead> beads = readBeads(beadsIn);
		for(Bead b : beads)
			b.transform(mat);
		writeBeads(beadsOut, beads);
	}

	public static void transform(
			ImagePlus imp,
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

		if(useCuda) {
			int d = imp.getStackSize();//Math.min(400, imp.getStackSize());
			if(imp.getType() == ImagePlus.GRAY16) {
				short[][] data = new short[d][];
				for(int z = 0; z < d; z++)
					data[z] = (short[])imp.getStack().getPixels(z + 1);
				NativeSPIMReconstructionCuda.transform16(
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
			}
			else if(imp.getType() == ImagePlus.GRAY8) {
				byte[][] data = new byte[d][];
				for(int z = 0; z < d; z++)
					data[z] = (byte[])imp.getStack().getPixels(z + 1);
				NativeSPIMReconstructionCuda.transform8(
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
			}
			else
				throw new RuntimeException("Only 16-bit grayscale images supported for CUDA");
		} else {
			ImagePlus xformed = transform(imp, inverseMatrix, targetW, targetH, targetD);
			IJ.save(xformed, outfile.getAbsolutePath() + ".tif");
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
		transform(imp, outfile, inverseMatrix, targetW, targetH, targetD, createTransformedMask, maskfile, border, zspacing, useCuda);
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
								MatrixUtils.apply(inv, x, y, z, result);
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

	static ArrayList<Bead> readBeads(File path) throws IOException {
		BufferedReader buf = new BufferedReader(new FileReader(path));
		buf.readLine();
		String line;
		ArrayList<Bead> beads = new ArrayList<Bead>();
		while((line = buf.readLine()) != null)
			beads.add(Bead.parse(line, false));
		buf.close();
		return beads;
	}

	static void writeBeads(File path, ArrayList<Bead> beads) throws IOException {
		PrintStream out = new PrintStream(new FileOutputStream(path));
		out.println("ID	ViewID\tLx\tLy\tLz\tWx\tWy\tWz\tWeight\tDescCorr\tRansacCorr");
		for(Bead b : beads)
			out.println(b.toLine());
		out.close();
	}

	// ret = M * scaleMatrix(dz)
	public static float readTransformation(File path, float[] ret) throws IOException {
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

	public static int[] readDims(File path) throws IOException {
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

	// ID	ViewID	Lx	Ly	Lz	Wx	Wy	Wz	Weight	DescCorr	RansacCorr
	static class Bead {
		int id, viewId;
		float x, y, z;
		float weight;
		String DescCorr, RansacCorr;

		static Bead parse(String line, boolean onlyCorr) {
			String[] toks = line.split("\t");
			if(onlyCorr && !toks[10].contains(":"))
				return null;
			Bead bead = new Bead();
			bead.id     = Integer.parseInt(toks[0]);
			bead.viewId = Integer.parseInt(toks[1]);
			bead.x      = Float.parseFloat(toks[2]);
			bead.y      = Float.parseFloat(toks[3]);
			bead.z      = Float.parseFloat(toks[4]);
			bead.weight = Float.parseFloat(toks[8]);
			bead.DescCorr = toks[9];
			bead.RansacCorr = toks[10];
			return bead;
		}

		void transform(float[] mat) {
//			float zscaling = 6.15384f; // TODO
//			this.z /= zscaling;
			float tx = mat[0] * x + mat[1] * y + mat[2]  * z + mat[3];
			float ty = mat[4] * x + mat[5] * y + mat[6]  * z + mat[7];
			float tz = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];
			this.x = tx;
			this.y = ty;
			this.z = tz;
		}

		String toLine() {
			return id + "\t" + viewId + "\t" + x + "\t" + y + "\t" + z + "\t" +
					x + "\t" + y + "\t" + z + "\t" + weight + "\t" + DescCorr + "\t" + RansacCorr;
		}
	}
}
