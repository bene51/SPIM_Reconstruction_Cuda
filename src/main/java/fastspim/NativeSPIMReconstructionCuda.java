package fastspim;


public class NativeSPIMReconstructionCuda {

	static {
		System.loadLibrary("NativeSPIMReconstructionCuda");
	}

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
			int nViews,
			int iterations);
}
