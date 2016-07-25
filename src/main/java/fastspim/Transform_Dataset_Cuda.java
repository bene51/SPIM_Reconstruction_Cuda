package fastspim;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.plugin.PlugIn;

import java.io.File;
import java.util.Arrays;
import java.util.Map;

import mpicbg.spim.data.SpimDataException;
import mpicbg.spim.data.registration.ViewRegistration;
import mpicbg.spim.data.sequence.Angle;
import mpicbg.spim.data.sequence.Channel;
import mpicbg.spim.data.sequence.Illumination;
import mpicbg.spim.data.sequence.ImgLoader;
import mpicbg.spim.data.sequence.SequenceDescription;
import mpicbg.spim.data.sequence.TimePoint;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import mpicbg.spim.data.sequence.VoxelDimensions;
import net.imglib2.Dimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.imageplus.ImagePlusImgFactory;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import spim.fiji.spimdata.SpimData2;
import spim.fiji.spimdata.XmlIoSpimData2;
import spim.fiji.spimdata.imgloaders.AbstractImgFactoryImgLoader;

public class Transform_Dataset_Cuda implements PlugIn {

	@Override
	public void run(String arg) {
		int nCudaDevices = NativeSPIMReconstructionCuda.getNumCudaDevices();
		String[] devices = new String[nCudaDevices];
		for(int i = 0; i < nCudaDevices; i++)
			devices[i] = NativeSPIMReconstructionCuda.getCudaDeviceName(i);

		GenericDialogPlus gd = new GenericDialogPlus("Transform Cuda");
		gd.addDirectoryField("Dataset xml file", "");
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
		resliceChoice[Transform_Cuda.NO_RESLICE] = "No";
		resliceChoice[Transform_Cuda.FROM_TOP] = "From top";
		resliceChoice[Transform_Cuda.FROM_RIGHT] = "From left";
		gd.addChoice("Reslice result", resliceChoice, resliceChoice[Transform_Cuda.FROM_TOP]);
		gd.addChoice("Device", devices, devices[0]);
		gd.showDialog();
		if(gd.wasCanceled())
			return;

		String dataset = gd.getNextString();
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

		try {
			NativeSPIMReconstructionCuda.setCudaDevice(device);
			transform(dataset, offset, size, spacing, reslice, createWeights);
		} catch(Exception e) {
			IJ.handleException(e);
		}
	}

	public void transform(
			String xml,
			int[] offset,
			int[] size,
			float[] spacing,
			int reslice,
			boolean createWeights) throws SpimDataException	{

		// ask for everything but the channels

		final XmlIoSpimData2 io = new XmlIoSpimData2("");
		final SpimData2 data = io.load(xml);

		File outdir = new File(data.getBasePath(), "output");

		SequenceDescription seqDesc = data.getSequenceDescription();
		Map<ViewId, ViewDescription> viewDescriptions = seqDesc.getViewDescriptions();

		int n = viewDescriptions.size();
		ViewId[] viewIds = new ViewId[n];
		ViewDescription[] viewDescs = new ViewDescription[n];
		Dimensions[] dims = new Dimensions[n];
		VoxelDimensions[] voxelDims = new VoxelDimensions[n];
		AffineTransform3D[] matrices = new AffineTransform3D[n];
		float[] zspacing = new float[n];

		// List<BoundingBox> boundingBoxes = data.getBoundingBoxes().getBoundingBoxes();
		// String pattern = data.getSequenceDescription().getImgLoader().
		int i = 0;
		for(Map.Entry<ViewId, ViewDescription> e : viewDescriptions.entrySet()) {
			ViewId viewId = e.getKey();
			ViewDescription viewDesc = e.getValue();

			viewIds[i] = viewId;
			viewDescs[i] = viewDesc;
			ViewSetup vSetup = viewDesc.getViewSetup();
			if(vSetup.hasSize())
				dims[i] = vSetup.getSize();
			else {
				dims[i] = data.getSequenceDescription().getImgLoader().getSetupImgLoader(vSetup.getId()).getImageSize(viewDesc.getTimePointId());
				vSetup.setSize(dims[i]);
			}
			if(vSetup.hasVoxelSize())
				voxelDims[i] = vSetup.getVoxelSize();
			else {
				voxelDims[i] = data.getSequenceDescription().getImgLoader().getSetupImgLoader(vSetup.getId()).getVoxelSize(viewDesc.getTimePointId());
				vSetup.setVoxelSize(voxelDims[i]);
			}
			for(int j = 0; j < voxelDims[i].numDimensions(); j++)
				IJ.log("voxelDims[" + j + "] = " + voxelDims[i].dimension(j));
			ViewRegistration vRegistration = data.getViewRegistrations().getViewRegistration(viewId);
			matrices[i] = vRegistration.getModel();
			zspacing[i] = (float)voxelDims[i].dimension(2);
			IJ.log("zspacing[" + i + "] = " + zspacing[i]);
			i++;
		}

		if(!outdir.exists())
			outdir.mkdirs();

		float[][] mats = createRealTransformationMatrices(dims, matrices, offset, size, spacing, reslice);

		Transform_Cuda.writeDims(new File(outdir, "dims.txt"), size[0], size[1], size[2]);

		int border = 30;
		File maskdir = new File(outdir, "masks");
		if(createWeights && !maskdir.exists())
			maskdir.mkdir();

		for(i = 0; i < n; i++) {
			String t = viewDescs[i].getTimePoint().getName();
			String a = viewDescs[i].getViewSetup().getAngle().getName();
			String c = viewDescs[i].getViewSetup().getChannel().getName();
			String il = viewDescs[i].getViewSetup().getIllumination().getName();
			String outname = String.format("t%s_a%s_c%s_i%s.raw", t, a, c, il);
			IJ.log("Transforming " + outname);
			IJ.log("matrix[i] = " + Arrays.toString(mats[i]));

			File maskfile = new File(maskdir, outname);

			Transform_Cuda.transform(
					open(data, viewIds[i]),
					new File(outdir, outname),
					mats[i], size[0], size[1], size[2],
					createWeights,
					maskfile,
					border,
					zspacing[i],
					true);
		}
	}

	public float[][] createRealTransformationMatrices(
			Dimensions[] dims,
			AffineTransform3D[] matrices,
			int[] offset,
			int[] size,
			float[] spacing,
			int reslice) {

		int n = dims.length;
		int[][] dimensions = new int[n][3];
		float[][] mats = new float[n][12];
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < 3; j++)
				dimensions[i][j] = (int)dims[i].dimension(j);

			for(int j = 0; j < 12; j++)
				mats[i][j] = (float)matrices[i].get(j / 4, j % 4);
		}
		Transform_Cuda.createRealTransformationMatrices(dimensions, mats, offset, size, spacing, reslice);
		return mats;
	}

	public static String name(final ViewDescription vd) {
		final Angle angle = vd.getViewSetup().getAngle();
		final Channel channel = vd.getViewSetup().getChannel();
		final Illumination illumination = vd.getViewSetup().getIllumination();
		final TimePoint tp = vd.getTimePoint();

		return "angle: " + angle.getName() + " channel: " + channel.getName() + " illum: " + illumination.getName() + " timepoint: " + tp.getName();
	}

	public RandomAccessibleInterval< UnsignedShortType > open(
			final SpimData2 spimData,
			final ViewId viewId)
	{
		final ImgLoader imgLoader = spimData.getSequenceDescription().getImgLoader();
		System.out.println(spimData.getSequenceDescription().getViewDescription(viewId).getViewSetup().getName());
		final ImgFactory< ? extends NativeType< ? > > factory;
		final AbstractImgFactoryImgLoader il;

		// load as ImagePlus directly if possible
		if ( AbstractImgFactoryImgLoader.class.isInstance( imgLoader ) )
		{
			il = (AbstractImgFactoryImgLoader)imgLoader;
			factory = il.getImgFactory();
			il.setImgFactory( new ImagePlusImgFactory< FloatType >());
		}
		else
		{
			il = null;
			factory = null;
		}

		@SuppressWarnings( "unchecked" )
		RandomAccessibleInterval< UnsignedShortType > img =
			( RandomAccessibleInterval< UnsignedShortType > ) spimData.getSequenceDescription().getImgLoader()
					.getSetupImgLoader( viewId.getViewSetupId() )
					.getImage( viewId.getTimePointId());
		if ( factory != null && il != null )
			il.setImgFactory( factory );

		return img;
	}
}
