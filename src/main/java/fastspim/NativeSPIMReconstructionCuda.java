package fastspim;


public class NativeSPIMReconstructionCuda {

	static {
		System.loadLibrary("NativeSPIMReconstructionCuda");
	}

	public static final int INDEPENDENT        = 0;
	public static final int EFFICIENT_BAYESIAN = 1;
	public static final int OPTIMIZATION_1     = 2;
	public static final int OPTIMIZATION_2     = 3;

	private static DataProvider provider = null;

	public static void setDataProvider(DataProvider p) {
		provider = p;
	}

	public static void clearDataProvider() {
		provider = null;
	}

	public synchronized static native int getNumCudaDevices();

	public synchronized static native String getCudaDeviceName(int dev);

	public synchronized static native void setCudaDevice(int dev);

	public synchronized static native void transform(
			short[][] data,
			int w,
			int h,
			int d,
			float[] inverseMatrix,
			int targetW,
			int targetH,
			int targetD,
			String outfile,
			boolean createTransformedMask,
			int border,
			float zspacing,
			String maskfile);

	public synchronized static native void deconvolve(
			String[] inputfiles,
			String outputfile,
			int w,
			int h,
			int d,
			String[] weightfiles,
			String[] kernelfiles,
			int kernelH,
			int kernelW,
			int psfType,
			int nViews,
			int iterations);

	public synchronized static native void deconvolve_quit();

	public synchronized static native void deconvolve_interactive(int iterations);

	public synchronized static native void deconvolve_init(
			int w,
			int h,
			int d,
			String[] weightfiles,
			String[] kernelfiles,
			int kernelH,
			int kernelW,
			int iterationType,
			int nViews);

	public static short[][] getNextPlane() {
		if(provider == null)
			return null;
		return provider.getNextPlane();
	}

	public static void returnNextPlane(short[] plane) {
		if(provider != null)
			provider.returnNextPlane(plane);
	}

	public static interface DataProvider {
		public short[][] getNextPlane();

		public void returnNextPlane(short[] plane);
	}
}
