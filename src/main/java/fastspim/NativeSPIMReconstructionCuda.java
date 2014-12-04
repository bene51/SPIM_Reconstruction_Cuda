package fastspim;


public class NativeSPIMReconstructionCuda {

	static {
		System.loadLibrary("NativeSPIMReconstructionCuda");
	}


	public static final int INDEPENDENT        = 0;
	public static final int EFFICIENT_BAYESIAN = 1;
	public static final int OPTIMIZATION_1     = 2;
	public static final int OPTIMIZATION_2     = 3;

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
}
