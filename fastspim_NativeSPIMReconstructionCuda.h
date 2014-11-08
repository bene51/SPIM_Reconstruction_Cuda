/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class fastspim_NativeSPIMReconstructionCuda */

#ifndef _Included_fastspim_NativeSPIMReconstructionCuda
#define _Included_fastspim_NativeSPIMReconstructionCuda
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     fastspim_NativeSPIMReconstructionCuda
 * Method:    transform
 * Signature: ([[SIII[FIIILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_transform(
		JNIEnv *env,
		jclass,
		jobjectArray data,
		jint w,
		jint h,
		jint d,
		jfloatArray invMatrix,
		jint targetW,
		jint targetH,
		jint targetD,
		jstring outfile,
		jboolean createTransformedMasks,
		jint border,
		jfloat zspacing,
		jstring maskfile);


JNIEXPORT void JNICALL Java_fastspim_NativeSPIMReconstructionCuda_deconvolve(
		JNIEnv *env,
		jclass,
		jobjectArray inputfiles,
		jstring outputfile,
		jint dataW,
		jint dataH,
		jint dataD,
		jobjectArray weightfiles,
		jobjectArray kernelfiles,
		jint kernelH,
		jint kernelW,
		jint psfType,
		jint nViews,
		jint iterations);
#ifdef __cplusplus
}
#endif
#endif
