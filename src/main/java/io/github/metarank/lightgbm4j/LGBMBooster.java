package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.*;

import java.io.*;

import static com.microsoft.ml.lightgbm.lightgbmlib.*;

public class LGBMBooster {
    private int iterations;
    private SWIGTYPE_p_p_void handle;

    private static final long MODEL_SAVE_BUFFER_SIZE = 10 * 1024 * 1024L;
    private static final long EVAL_RESULTS_BUFFER_SIZE = 1024;


    private static volatile boolean nativeLoaded = false;
    static {
        try {
            LGBMBooster.loadNative();
        } catch (IOException e) {
            System.out.println("Cannot load native library for your platform");
        }
    }

    /**
     * Called from tests.
     * @return true if JNI libraries were loaded successfully.
     */
    public static boolean isNativeLoaded() {
        return nativeLoaded;
    }

    /**
     * Loads all corresponsing native libraries for current platform. Called from the class initializer,
     * so usually there is no need to call it directly.
     * @throws IOException
     */
    public synchronized static void loadNative() throws IOException {
        if (!nativeLoaded) {
            String os = System.getProperty("os.name");
            if (os.startsWith("Linux") || os.startsWith("LINUX")) {
                loadNative("linux/x86_64/lib_lightgbm.so", "lightgbm");
                loadNative("linux/x86_64/lib_lightgbm_swig.so", "lightgbm_swig");
                nativeLoaded = true;
            } else if (os.startsWith("Mac")) {
                loadNative("osx/x86_64/lib_lightgbm.dylib", "lightgbm");
                loadNative("osx/x86_64/lib_lightgbm_swig.dylib", "lightgbm_swig");
                nativeLoaded = true;
            } else if (os.startsWith("Windows")) {
                loadNative("windows/x86_64/lib_lightgbm.dll", "lightgbm");
                loadNative("windows/x86_64/lib_lightgbm_swig.dll", "lightgbm_swig");
                nativeLoaded = true;
            } else {
                System.out.println("Only Linux/Windows/Mac on x86_64 are supported");
            }
        }
    }

    private static void loadNative(String path, String name) throws IOException {
        System.out.println("Loading native lib " + path);
        String libPath = extractResource(path, name).getPath();
        File libFile = new File(libPath);
        System.out.println("Extracted file: exists=" + libFile.exists() + " path="+ libFile);
        try {
            System.load(libPath);
        } catch (UnsatisfiedLinkError err) {
            System.out.println("Cannot load library: " + err + " cause: " + err.getMessage());
        }
    }

    private static File extractResource(String path, String name) throws IOException {
        File tempFile = File.createTempFile(name + "_", ".bin" );
        System.out.println("Loading native lib " + tempFile);
        InputStream libStream = LGBMBooster.class.getClassLoader().getResourceAsStream(path);
        OutputStream fileStream = new FileOutputStream(tempFile);
        copyStream(libStream, fileStream);
        libStream.close();
        fileStream.close();
        return tempFile;
    }

    private static void copyStream(InputStream source, OutputStream target) throws IOException {
        byte[] buf = new byte[8192];
        int length;
        int bytesCopied = 0;
        while ((length = source.read(buf)) > 0) {
            target.write(buf, 0, length);
            bytesCopied += length;
        }
        System.out.println("Copied " + bytesCopied + " bytes");
    }

    /**
     * Constructor is private because you need to have a JNI handle for native LightGBM instance.
     * @param iterations
     * @param handle
     */
    LGBMBooster(int iterations, SWIGTYPE_p_p_void handle) {
        this.iterations = iterations;
        this.handle = handle;
    }

    /**
     * Load an existing booster from model file.
     * @param file Filename of model
     * @return Booster instance.
     * @throws LGBMException
     */
    public static LGBMBooster createFromModelfile(String file) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        SWIGTYPE_p_int outIterations = new_intp();
        int result = LGBM_BoosterCreateFromModelfile(file, outIterations, handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int iterations = intp_value(outIterations);
            delete_intp(outIterations);
            return new LGBMBooster(iterations, handle);
        }
    }

    /**
     * Load an existing booster from string.
     * @param model Model string
     * @return Booster instance.
     * @throws LGBMException
     */
    public static LGBMBooster loadModelFromString(String model) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        SWIGTYPE_p_int outIterations = new_intp();
        int result = LGBM_BoosterLoadModelFromString(model, outIterations, handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int iterations = intp_value(outIterations);
            delete_intp(outIterations);
            return new LGBMBooster(iterations, handle);
        }
    }

    /**
     * Deallocate all native memory for the LightGBM model.
     * @throws LGBMException
     */
    public void close() throws LGBMException {
        int result = LGBM_BoosterFree(voidpp_value(handle));
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Make prediction for a new float[] dataset.
     * @param input input matrix, as a 1D array. Size should be rows * cols.
     * @param rows number of rows
     * @param cols number of cols
     * @param isRowMajor is the 1d encoding a row-major?
     * @return array of predictions
     * @throws LGBMException
     */
    public double[] predictForMat(float[] input, int rows, int cols, boolean isRowMajor) throws LGBMException {
        SWIGTYPE_p_float dataBuffer = new_floatArray(input.length);
        for (int i = 0; i < input.length; i++) {
            floatArray_setitem(dataBuffer, i, input[i]);
        }
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        SWIGTYPE_p_double outBuffer = new_doubleArray(2L * rows);
        int result = LGBM_BoosterPredictForMat(
                voidpp_value(handle),
                float_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT32,
                rows,
                cols,
                isRowMajor ? 1 : 0,
                C_API_PREDICT_NORMAL,
                0,
                iterations,
                "",
                outLength,
                outBuffer);
        if (result < 0) {
            delete_floatArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            long length = int64_tp_value(outLength);
            double[] values = new double[(int)length];
            for (int i = 0; i < length; i++) {
                values[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_floatArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            return values;
        }
    }
    /**
     * Make prediction for a new double[] dataset.
     * @param input input matrix, as a 1D array. Size should be rows * cols.
     * @param rows number of rows
     * @param cols number of cols
     * @param isRowMajor is the 1d encoding a row-major?
     * @return array of predictions
     * @throws LGBMException
     */

    public double[] predictForMat(double[] input, int rows, int cols, boolean isRowMajor) throws LGBMException {
        SWIGTYPE_p_double dataBuffer = new_doubleArray(input.length);
        for (int i = 0; i < input.length; i++) {
            doubleArray_setitem(dataBuffer, i, input[i]);
        }
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        SWIGTYPE_p_double outBuffer = new_doubleArray(2L * rows);
        int result = LGBM_BoosterPredictForMat(
                voidpp_value(handle),
                double_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT64,
                rows,
                cols,
                isRowMajor ? 1 : 0,
                C_API_PREDICT_NORMAL,
                0,
                iterations,
                "",
                outLength,
                outBuffer);
        if (result < 0) {
            delete_doubleArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            long length = int64_tp_value(outLength);
            double[] values = new double[(int)length];
            for (int i = 0; i < length; i++) {
                values[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_doubleArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            return values;
        }
    }

    /**
     * Create a new boosting learner.
     * @param dataset a LGBMDataset with the training data.
     * @param parameters Parameters in format ‘key1=value1 key2=value2’
     * @return
     * @throws LGBMException
     */
    public static LGBMBooster create(LGBMDataset dataset, String parameters) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        int result = LGBM_BoosterCreate(dataset.handle, parameters, handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMBooster(0, handle);
        }
    }

    /**
     * Update the model for one iteration.
     * @return true if there are no more splits possible, so training is finished.
     * @throws LGBMException
     */
    public boolean updateOneIter() throws LGBMException {
        SWIGTYPE_p_int isFinishedP = new_intp();
        int result = LGBM_BoosterUpdateOneIter(voidpp_value(handle), isFinishedP);
        iterations++;
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int isFinished = intp_value(isFinishedP);
            delete_intp(isFinishedP);
            return isFinished == 1;
        }
    }

    public enum FeatureImportanceType {
        SPLIT,
        GAIN
    }


    /**
     * Save model to string.
     * @param startIteration Start index of the iteration that should be saved
     * @param numIteration Index of the iteration that should be saved, 0 and negative means save all
     * @param featureImportance Type of feature importance, can be FeatureImportanceType.SPLIT or FeatureImportanceType.GAIN
     * @return
     */
    public String saveModelToString(int startIteration, int numIteration, FeatureImportanceType featureImportance)  {
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        String result = LGBM_BoosterSaveModelToStringSWIG(
                voidpp_value(handle),
                startIteration,
                numIteration,
                importanceType(featureImportance),
                MODEL_SAVE_BUFFER_SIZE,
                outLength
        );
        return result;
    }

    /**
     * Get names of features.
     * @return a list of feature names.
     */
    public String[] getFeatureNames() {
        SWIGTYPE_p_void buffer = LGBM_BoosterGetFeatureNamesSWIG(voidpp_value(handle));
        String[] result = StringArrayHandle_get_strings(buffer);
        StringArrayHandle_free(buffer);
        return result;
    }

    /**
     * Add new validation data to booster.
     * @param dataset dataset to validate
     * @throws LGBMException
     */
    public void addValidData(LGBMDataset dataset) throws LGBMException {
        int result = LGBM_BoosterAddValidData(voidpp_value(handle), dataset.handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Get evaluation for training data and validation data.
     * @param dataIndex Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
     * @return
     * @throws LGBMException
     */
    public double[] getEval(int dataIndex) throws LGBMException {
        SWIGTYPE_p_int outLength = new_int32_tp();
        SWIGTYPE_p_double outBuffer = new_doubleArray(EVAL_RESULTS_BUFFER_SIZE);
        int result = LGBM_BoosterGetEval(voidpp_value(handle), dataIndex, outLength, outBuffer);
        if (result < 0) {
            delete_intp(outLength);
            delete_doubleArray(outBuffer);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            double[] evals = new double[intp_value(outLength)];
            for (int i=0; i < evals.length; i++) {
                evals[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_intp(outLength);
            delete_doubleArray(outBuffer);
            return evals;
        }
    }

    /**
     * Get names of evaluation datasets.
     * @return array of eval dataset names.
     * @throws LGBMException
     */
    public String[] getEvalNames() throws LGBMException {
        SWIGTYPE_p_void namesP = LGBM_BoosterGetEvalNamesSWIG(voidpp_value(handle));
        String[] names = StringArrayHandle_get_strings(namesP);
        StringArrayHandle_free(namesP);
        return names;
    }

    /**
     * Get model feature importance.
     * @param numIteration Number of iterations for which feature importance is calculated, 0 or less means use all
     * @param importanceType GAIN or SPLIT
     * @return Result array with feature importance
     * @throws LGBMException
     */
    public double[] featureImportance(int numIteration, FeatureImportanceType importanceType) throws LGBMException {
        int numFeatures = getNumFeature();
        SWIGTYPE_p_double outBuffer = new_doubleArray(numFeatures);
        int result = LGBM_BoosterFeatureImportance(
                voidpp_value(handle),
                numIteration,
                importanceType(importanceType),
                outBuffer
        );
        if (result < 0) {
            delete_doubleArray(outBuffer);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            double[] importance = new double[numFeatures];
            for (int i=0; i < numFeatures; i++) {
                importance[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_doubleArray(outBuffer);
            return importance;
        }
    }

    /**
     * Get number of features.
     * @return number of features
     * @throws LGBMException
     */
    public int getNumFeature() throws LGBMException {
        SWIGTYPE_p_int outNum = new_int32_tp();
        int result = LGBM_BoosterGetNumFeature(voidpp_value(handle), outNum);
        if (result < 0) {
            delete_intp(outNum);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int num = intp_value(outNum);
            delete_intp(outNum);
            return num;
        }
    }

    /**
     * Make prediction for a new double[] row dataset. This method re-uses the internal predictor structure from previous calls
     * and is optimized for single row invocation.
     * @param data input vector
     * @return score
     * @throws LGBMException
     */
    public double predictForMatSingleRow(double[] data) throws LGBMException {
        SWIGTYPE_p_double dataBuffer = new_doubleArray(data.length);
        for (int i = 0; i < data.length; i++) {
            doubleArray_setitem(dataBuffer, i, data[i]);
        }
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        SWIGTYPE_p_double outBuffer = new_doubleArray(1);

        int result = LGBM_BoosterPredictForMatSingleRow(
                voidpp_value(handle),
                double_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT64,
                data.length,
                1,
                C_API_PREDICT_NORMAL,
                0,
                iterations,
                "",
                outLength,
                outBuffer
                );
        if (result < 0) {
            delete_doubleArray(dataBuffer);
            delete_doubleArray(outBuffer);
            delete_int64_tp(outLength);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            long length = int64_tp_value(outLength);
            double[] values = new double[(int)length];
            for (int i = 0; i < length; i++) {
                values[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_doubleArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            return values[0];
        }
    }
    /**
     * Make prediction for a new float[] row dataset. This method re-uses the internal predictor structure from previous calls
     * and is optimized for single row invocation.
     * @param data input vector
     * @return score
     * @throws LGBMException
     */
    public double predictForMatSingleRow(float[] data) throws LGBMException {
        SWIGTYPE_p_float dataBuffer = new_floatArray(data.length);
        for (int i = 0; i < data.length; i++) {
            floatArray_setitem(dataBuffer, i, data[i]);
        }
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        SWIGTYPE_p_double outBuffer = new_doubleArray(1);

        int result = LGBM_BoosterPredictForMatSingleRow(
                voidpp_value(handle),
                float_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT32,
                data.length,
                1,
                C_API_PREDICT_NORMAL,
                0,
                iterations,
                "",
                outLength,
                outBuffer
        );
        if (result < 0) {
            delete_floatArray(dataBuffer);
            delete_doubleArray(outBuffer);
            delete_int64_tp(outLength);
            throw new LGBMException(LGBM_GetLastError());
        } else {
            long length = int64_tp_value(outLength);
            double[] values = new double[(int)length];
            for (int i = 0; i < length; i++) {
                values[i] = doubleArray_getitem(outBuffer, i);
            }
            delete_floatArray(dataBuffer);
            delete_int64_tp(outLength);
            delete_doubleArray(outBuffer);
            return values[0];
        }
    }

    private int importanceType(FeatureImportanceType tpe) {
        int importanceType = C_API_FEATURE_IMPORTANCE_GAIN;
        switch (tpe) {
            case GAIN:
                importanceType = C_API_FEATURE_IMPORTANCE_GAIN;
                break;
            case SPLIT:
                importanceType = C_API_FEATURE_IMPORTANCE_SPLIT;
                break;
        }
        return importanceType;
    }
}
