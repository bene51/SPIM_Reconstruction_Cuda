package fastspim;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

public class MatrixUtils {

	/**
	 * Calculate the concatenated tranformation that's needed to transform data that's been cropped at
	 * x,y,z and saves the resulting matrix back to <code>outfile</code>.
	 * @param infile
	 * @param outfile
	 * @param x
	 * @param y
	 * @param z
	 * @throws IOException
	 */
	static void transform(String infile, String outfile, float x, float y, float z) throws IOException {
		float[] mat = new float[12];
		float zspacing = Transform_Cuda.readTransformation(new File(infile), mat);
		mat[2]  /= zspacing;
		mat[6]  /= zspacing;
		mat[10] /= zspacing;
		float[] res = new float[3];
		System.out.println(Arrays.toString(mat));
		MatrixUtils.apply(mat, x, y, z * zspacing, res);
		mat[3]  = res[0];
		mat[7]  = res[1];
		mat[11] = res[2];
		System.out.println(Arrays.toString(mat));

		ArrayList<String>lines = new ArrayList<String>();
		BufferedReader buf = new BufferedReader(new FileReader(infile));
		for(int i = 0; i < 12; i++)
			buf.readLine();
		String line;
		while((line = buf.readLine()) != null) {
			lines.add(line);
		}
		buf.close();

		PrintStream out = new PrintStream(new FileOutputStream(outfile));
		out.println("m00: " + mat[0]);
		out.println("m01: " + mat[1]);
		out.println("m02: " + mat[2]);
		out.println("m03: " + mat[3]);
		out.println("m10: " + mat[4]);
		out.println("m11: " + mat[5]);
		out.println("m12: " + mat[6]);
		out.println("m13: " + mat[7]);
		out.println("m20: " + mat[8]);
		out.println("m21: " + mat[9]);
		out.println("m22: " + mat[10]);
		out.println("m23: " + mat[11]);
		for(String l : lines)
			out.println(l);
		out.close();
	}

	public static void main(String[] args) throws IOException {
		String infile  = "E:\\SPIM5_Deconvolution\\t00000_s000_v0.tif.registration";
		String outfile = "E:\\SPIM5_Deconvolution\\dataset1\\registration\\t00000_s000_v0.tif.registration";
		transform(infile, outfile, 650, 0, 169);

		infile  = "E:\\SPIM5_Deconvolution\\t00000_s000_v6.tif.registration";
		outfile = "E:\\SPIM5_Deconvolution\\dataset1\\registration\\t00000_s000_v6.tif.registration";
		transform(infile, outfile, 650, 0, 0);
	}

	public static void invert3x3(float[] mat) {
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

	public static void invert(float[] mat) {
		float dx = -mat[3];
		float dy = -mat[7];
		float dz = -mat[11];
		invert3x3(mat);

		mat[3]  = mat[0] * dx + mat[1] * dy + mat[2]  * dz;
		mat[7]  = mat[4] * dx + mat[5] * dy + mat[6]  * dz;
		mat[11] = mat[8] * dx + mat[9] * dy + mat[10] * dz;
	}


	public static void apply(float[] mat, float x, float y, float z, float[] result) {
		result[0] = mat[0] * x + mat[1] * y + mat[2]  * z + mat[3];
		result[1] = mat[4] * x + mat[5] * y + mat[6]  * z + mat[7];
		result[2] = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];
	}

	public static float[] fromTranslation(float dx, float dy, float dz, float[] result) {
		if(result == null)
			result = new float[12];
		result[0] = result[5] = result[10] = 1;
		result[1] = result[2] = result[4] = result[6] = result[8] = result[9] = 0;
		result[3] = dx;
		result[7] = dy;
		result[11] = dz;
		return result;
	}

	public static float[] mul(float[] m1, float[] m2) {
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
}
