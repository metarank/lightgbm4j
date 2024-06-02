package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

import static com.microsoft.ml.lightgbm.lightgbmlib.*;

public class LGBMBooster implements AutoCloseable {
    private int iterations;
    private SWIGTYPE_p_p_void handle;

    private static final long MODEL_SAVE_BUFFER_SIZE = 10 * 1024 * 1024L;
    private static final long EVAL_RESULTS_BUFFER_SIZE = 1024;

    private static final Logger logger = LoggerFactory.getLogger(LGBMBooster.class);
    private static volatile boolean nativeLoaded = false;

    private volatile boolean isClosed = false;

    static {
        try {
            LGBMBooster.loadNative();
        } catch (IOException e) {
            logger.info("Cannot load native library for your platform");
        }
    }

    /**
     * Called from tests.
     *
     * @return true if JNI libraries were loaded successfully.
     */
    public static boolean isNativeLoaded() {
        return nativeLoaded;
    }

    /**
     * Loads all corresponsing native libraries for current platform. Called from the class initializer,
     * so usually there is no need to call it directly.
     *
     * @throws IOException
     */
    public synchronized static void loadNative() throws IOException {
        if (!nativeLoaded) {
            String os = System.getProperty("os.name");
            String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
            if (os.startsWith("Linux") || os.startsWith("LINUX")) {
                try {
                    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
                        loadNative("lightgbm4j/linux/x86_64/", "lib_lightgbm.so");
                        loadNative("lightgbm4j/linux/x86_64/", "lib_lightgbm_swig.so");
                        nativeLoaded = true;
                    } else if (arch.startsWith("aarch64") || arch.startsWith("arm64")) {
                        loadNative("lightgbm4j/linux/aarch64/", "lib_lightgbm.so");
                        loadNative("lightgbm4j/linux/aarch64/", "lib_lightgbm_swig.so");
                        nativeLoaded = true;
                    }
                } catch (UnsatisfiedLinkError err) {
                    String message = err.getMessage();
                    if (message.contains("libgomp")) {
                        logger.warn("\n\n\n");
                        logger.warn("****************************************************");
                        logger.warn("Your Linux system probably has no 'libgomp' library installed!");
                        logger.warn("Please double-check the lightgbm4j install instructions:");
                        logger.warn("- https://github.com/metarank/lightgbm4j/");
                        logger.warn("- or just install the libgomp with your package manager");
                        logger.warn("****************************************************");
                        logger.warn("\n\n\n");
                    }
                }
            } else if (os.startsWith("Mac")) {
                try {
                    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
                        loadNative("lightgbm4j/osx/x86_64/", "lib_lightgbm.dylib");
                        loadNative("lightgbm4j/osx/x86_64/", "lib_lightgbm_swig.dylib");
                        nativeLoaded = true;
                    } else if (arch.startsWith("aarch64") || arch.startsWith("arm64")) {
                        loadNative("lightgbm4j/osx/aarch64/", "lib_lightgbm.dylib");
                        loadNative("lightgbm4j/osx/aarch64/", "lib_lightgbm_swig.dylib");
                        nativeLoaded = true;
                    } else {
                        logger.warn("arch " + arch + " is not supported");
                        throw new UnsatisfiedLinkError("no native lightgbm library found for your OS "+os);
                    }
                } catch (UnsatisfiedLinkError err) {
                    String message = err.getMessage();
                    if (message.contains("libomp.dylib")) {
                        logger.warn("\n\n\n");
                        logger.warn("****************************************************");
                        logger.warn("Your MacOS system probably has no 'libomp' library installed!");
                        logger.warn("Please double-check the lightgbm4j install instructions:");
                        logger.warn("- https://github.com/metarank/lightgbm4j/");
                        logger.warn("- or just do 'brew install libomp'");
                        logger.warn("****************************************************");
                        logger.warn("\n\n\n");

                    }
                    throw err;
                }
            } else if (os.startsWith("Windows")) {
                loadNative("lightgbm4j/windows/x86_64/", "lib_lightgbm.dll");
                loadNative("lightgbm4j/windows/x86_64/", "lib_lightgbm_swig.dll");
                nativeLoaded = true;
            } else {
                logger.error("Only Linux@x86_64, Windows@x86_64, Mac@x86_64 and Mac@aarch are supported");
            }
        }
    }

    private static void loadNative(String path, String name) throws IOException, UnsatisfiedLinkError {
        String nativePathOverride = System.getenv("LIGHTGBM_NATIVE_LIB_PATH");
        if (nativePathOverride != null) {
            if (!nativePathOverride.endsWith("/")) {
                nativePathOverride = nativePathOverride + "/";
            }
            String libFile = nativePathOverride + name;
            logger.info("LIGHTGBM_NATIVE_LIB_PATH is set: loading " + libFile);
            try {
                System.load(libFile);
            } catch (UnsatisfiedLinkError err) {
                logger.error("Cannot load library:" + err.getMessage(), err);
                throw err;
            }
        } else {
            logger.info("Loading native lib from resource " + path + "/" + name);
            String tmp = System.getProperty("java.io.tmpdir");
            File libFile = new File(tmp + File.separator + name);
            if (libFile.exists()) {
                logger.info(libFile + " already exists");
                extractResource(path + name, name, libFile);
            } else {
                extractResource(path + name, name, libFile);
            }
            logger.info("Extracted file: exists=" + libFile.exists() + " path=" + libFile);
            try {
                System.load(libFile.toString());
            } catch (UnsatisfiedLinkError err) {
                logger.error("Cannot load library:" + err.getMessage(), err);
                throw err;
            }
        }
    }

    private static void extractResource(String path, String name, File dest) throws IOException {
        logger.info("Extracting native lib " + dest);
        InputStream libStream = LGBMBooster.class.getClassLoader().getResourceAsStream(path);
        ByteArrayOutputStream libByteStream = new ByteArrayOutputStream();
        copyStream(libStream, libByteStream);
        libStream.close();

        InputStream md5Stream = LGBMBooster.class.getClassLoader().getResourceAsStream(path + ".md5");
        ByteArrayOutputStream md5ByteStream = new ByteArrayOutputStream();
        copyStream(md5Stream, md5ByteStream);
        md5Stream.close();
        String expectedDigest = md5ByteStream.toString();
        try {
            byte[] digest = MessageDigest.getInstance("MD5").digest(libByteStream.toByteArray());
            String checksum = new BigInteger(1, digest).toString(16);
            if (!checksum.equals(expectedDigest)) {
                logger.warn("\n\n\n");
                logger.warn("****************************************************");
                logger.warn("Hash mismatch between expected and real LightGBM native library in classpath!");
                logger.warn("Your JVM classpath has "+name+" with md5="+checksum+" and we expect "+expectedDigest);
                logger.warn("This usually means that you have another LightGBM wrapper in classpath");
                logger.warn("- MMLSpark/SynapseML is the main suspect");
                logger.warn("****************************************************");
                logger.warn("\n\n\n");
                //throw new IOException("hash mismatch");
            }
            ByteArrayInputStream source = new ByteArrayInputStream(libByteStream.toByteArray());
            OutputStream fileStream = new FileOutputStream(dest);
            copyStream(source, fileStream);
            source.close();
            fileStream.close();
        } catch (NoSuchAlgorithmException ex) {
            throw new IOException("md5 algorithm not supported, cannot check digest");
        }
    }

    private static void copyStream(InputStream source, OutputStream target) throws IOException {
        byte[] buf = new byte[8192];
        int length;
        int bytesCopied = 0;
        while ((length = source.read(buf)) > 0) {
            target.write(buf, 0, length);
            bytesCopied += length;
        }
        logger.info("Copied " + bytesCopied + " bytes");
    }

    /**
     * Constructor is private because you need to have a JNI handle for native LightGBM instance.
     *
     * @param iterations
     * @param handle
     */
    LGBMBooster(int iterations, SWIGTYPE_p_p_void handle) {
        this.iterations = iterations;
        this.handle = handle;
    }

    /**
     * Load an existing booster from model file.
     *
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
     *
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
     *
     * @throws LGBMException
     */
    @Override
    public void close() throws LGBMException {
        if (!isClosed) {
            isClosed = true;
            int result = LGBM_BoosterFree(voidpp_value(handle));
            if (result < 0) {
                throw new LGBMException(LGBM_GetLastError());
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Make prediction for a new float[] dataset.
     *
     * @param input          input matrix, as a 1D array. Size should be rows * cols.
     * @param rows           number of rows
     * @param cols           number of cols
     * @param isRowMajor     is the 1d encoding a row-major?
     * @param predictionType the prediction type
     * @param parameter      prediction options
     * @return array of predictions
     * @throws LGBMException
     */
    public double[] predictForMat(float[] input, int rows, int cols, boolean isRowMajor, PredictionType predictionType, String parameter) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_float dataBuffer = new_floatArray(input.length);
            for (int i = 0; i < input.length; i++) {
                floatArray_setitem(dataBuffer, i, input[i]);
            }
            SWIGTYPE_p_long_long outLength = new_int64_tp();
            long outSize = outBufferSize(rows, cols, predictionType);
            SWIGTYPE_p_double outBuffer = new_doubleArray(outSize);
            int result = LGBM_BoosterPredictForMat(
                    voidpp_value(handle),
                    float_to_voidp_ptr(dataBuffer),
                    C_API_DTYPE_FLOAT32,
                    rows,
                    cols,
                    isRowMajor ? 1 : 0,
                    predictionType.getType(),
                    0,
                    iterations,
                    parameter,
                    outLength,
                    outBuffer);
            if (result < 0) {
                delete_floatArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                long length = int64_tp_value(outLength);
                double[] values = new double[(int) length];
                for (int i = 0; i < length; i++) {
                    values[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_floatArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                return values;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    public double[] predictForMat(float[] input, int rows, int cols, boolean isRowMajor, PredictionType predictionType) throws LGBMException {
        return predictForMat(input, rows, cols, isRowMajor, predictionType, "");
    }

    /**
     * Make prediction for a new double[] dataset.
     *
     * @param input          input matrix, as a 1D array. Size should be rows * cols.
     * @param rows           number of rows
     * @param cols           number of cols
     * @param isRowMajor     is the 1 d encoding a row-major?
     * @param predictionType the prediction type
     * @param parameter      prediction options
     * @return array of predictions
     * @throws LGBMException
     */

    public double[] predictForMat(double[] input, int rows, int cols, boolean isRowMajor, PredictionType predictionType, String parameter) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_double dataBuffer = new_doubleArray(input.length);
            for (int i = 0; i < input.length; i++) {
                doubleArray_setitem(dataBuffer, i, input[i]);
            }
            SWIGTYPE_p_long_long outLength = new_int64_tp();
            long outSize = outBufferSize(rows, cols, predictionType);
            SWIGTYPE_p_double outBuffer = new_doubleArray(outSize);
            int result = LGBM_BoosterPredictForMat(
                    voidpp_value(handle),
                    double_to_voidp_ptr(dataBuffer),
                    C_API_DTYPE_FLOAT64,
                    rows,
                    cols,
                    isRowMajor ? 1 : 0,
                    predictionType.getType(),
                    0,
                    iterations,
                    parameter,
                    outLength,
                    outBuffer);
            if (result < 0) {
                delete_doubleArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                long length = int64_tp_value(outLength);
                double[] values = new double[(int) length];
                for (int i = 0; i < length; i++) {
                    values[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_doubleArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                return values;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }
    public double[] predictForMat(double[] input, int rows, int cols, boolean isRowMajor, PredictionType predictionType) throws LGBMException {
        return predictForMat(input, rows, cols, isRowMajor, predictionType, "");
    }

    /**
     * Create a new boosting learner.
     *
     * @param dataset    a LGBMDataset with the training data.
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
     *
     * @return true if there are no more splits possible, so training is finished.
     * @throws LGBMException
     */
    public boolean updateOneIter() throws LGBMException {
        if (!isClosed) {
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
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    public enum FeatureImportanceType {
        SPLIT,
        GAIN
    }


    /**
     * Save model to string.
     *
     * @param startIteration    Start index of the iteration that should be saved
     * @param numIteration      Index of the iteration that should be saved, 0 and negative means save all
     * @param featureImportance Type of feature importance, can be FeatureImportanceType.SPLIT or FeatureImportanceType.GAIN
     * @return
     */
    public String saveModelToString(int startIteration, int numIteration, FeatureImportanceType featureImportance) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_long_long outLength = new_int64_tp();
            String result = LGBM_BoosterSaveModelToStringSWIG(
                    voidpp_value(handle),
                    startIteration,
                    numIteration,
                    importanceType(featureImportance),
                    MODEL_SAVE_BUFFER_SIZE,
                    outLength
            );
            delete_int64_tp(outLength);
            return result;
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get names of features.
     *
     * @return a list of feature names.
     */
    public String[] getFeatureNames() throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_void buffer = LGBM_BoosterGetFeatureNamesSWIG(voidpp_value(handle));
            String[] result = StringArrayHandle_get_strings(buffer);
            StringArrayHandle_free(buffer);
            return result;
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Add new validation data to booster.
     *
     * @param dataset dataset to validate
     * @throws LGBMException
     */
    public void addValidData(LGBMDataset dataset) throws LGBMException {
        if (!isClosed) {
            int result = LGBM_BoosterAddValidData(voidpp_value(handle), dataset.handle);
            if (result < 0) {
                throw new LGBMException(LGBM_GetLastError());
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get evaluation for training data and validation data.
     *
     * @param dataIndex Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
     * @return
     * @throws LGBMException
     */
    public double[] getEval(int dataIndex) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_int outLength = new_int32_tp();
            SWIGTYPE_p_double outBuffer = new_doubleArray(EVAL_RESULTS_BUFFER_SIZE);
            int result = LGBM_BoosterGetEval(voidpp_value(handle), dataIndex, outLength, outBuffer);
            if (result < 0) {
                delete_intp(outLength);
                delete_doubleArray(outBuffer);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                double[] evals = new double[intp_value(outLength)];
                for (int i = 0; i < evals.length; i++) {
                    evals[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_intp(outLength);
                delete_doubleArray(outBuffer);
                return evals;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get names of evaluation datasets.
     *
     * @return array of eval dataset names.
     * @throws LGBMException
     */
    public String[] getEvalNames() throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_void namesP = LGBM_BoosterGetEvalNamesSWIG(voidpp_value(handle));
            String[] names = StringArrayHandle_get_strings(namesP);
            StringArrayHandle_free(namesP);
            return names;
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get model feature importance.
     *
     * @param numIteration   Number of iterations for which feature importance is calculated, 0 or less means use all
     * @param importanceType GAIN or SPLIT
     * @return Result array with feature importance
     * @throws LGBMException
     */
    public double[] featureImportance(int numIteration, FeatureImportanceType importanceType) throws LGBMException {
        if (!isClosed) {
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
                for (int i = 0; i < numFeatures; i++) {
                    importance[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_doubleArray(outBuffer);
                return importance;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get number of features.
     *
     * @return number of features
     * @throws LGBMException
     */
    public int getNumFeature() throws LGBMException {
        if (!isClosed) {
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
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Make prediction for a new double[] row dataset. This method re-uses the internal predictor structure from previous calls
     * and is optimized for single row invocation.
     *
     * @param data           input vector
     * @param predictionType the prediction type
     * @return score
     * @throws LGBMException
     */
    public double predictForMatSingleRow(double[] data, PredictionType predictionType) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_double dataBuffer = new_doubleArray(data.length);
            for (int i = 0; i < data.length; i++) {
                doubleArray_setitem(dataBuffer, i, data[i]);
            }
            SWIGTYPE_p_long_long outLength = new_int64_tp();
            long outBufferSize = outBufferSize(1, data.length, predictionType);
            SWIGTYPE_p_double outBuffer = new_doubleArray(outBufferSize);

            int result = LGBM_BoosterPredictForMatSingleRow(
                    voidpp_value(handle),
                    double_to_voidp_ptr(dataBuffer),
                    C_API_DTYPE_FLOAT64,
                    data.length,
                    1,
                    predictionType.getType(),
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
                double[] values = new double[(int) length];
                for (int i = 0; i < length; i++) {
                    values[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_doubleArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                return values[0];
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Make prediction for a new float[] row dataset. This method re-uses the internal predictor structure from previous calls
     * and is optimized for single row invocation.
     *
     * @param data           input vector
     * @param predictionType the prediction type
     * @return score
     * @throws LGBMException
     */
    public double predictForMatSingleRow(float[] data, PredictionType predictionType) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_float dataBuffer = new_floatArray(data.length);
            for (int i = 0; i < data.length; i++) {
                floatArray_setitem(dataBuffer, i, data[i]);
            }
            SWIGTYPE_p_long_long outLength = new_int64_tp();
            long outBufferSize = outBufferSize(1, data.length, predictionType);
            SWIGTYPE_p_double outBuffer = new_doubleArray(outBufferSize);

            int result = LGBM_BoosterPredictForMatSingleRow(
                    voidpp_value(handle),
                    float_to_voidp_ptr(dataBuffer),
                    C_API_DTYPE_FLOAT32,
                    data.length,
                    1,
                    predictionType.getType(),
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
                double[] values = new double[(int) length];
                for (int i = 0; i < length; i++) {
                    values[i] = doubleArray_getitem(outBuffer, i);
                }
                delete_floatArray(dataBuffer);
                delete_int64_tp(outLength);
                delete_doubleArray(outBuffer);
                return values[0];
            }
        } else {
            throw new LGBMException("Booster was already closed");
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

    /**
     * Get number of classes.
     * @return Number of classes
     * @throws LGBMException
     */
    public int getNumClasses() throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_int numHandle = new_int32_tp();
            int result = LGBM_BoosterGetNumClasses(voidpp_value(handle), numHandle);
            if (result < 0) {
                delete_intp(numHandle);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                int numClasses = intp_value(numHandle);
                delete_intp(numHandle);
                return numClasses;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get number of predictions for training data and validation data (this can be used to support customized evaluation functions).
     * @param dataIdx  Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
     * @return Number of predictions
     * @throws LGBMException
     */
    public long getNumPredict(int dataIdx) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_long_long numHandle = new_int64_tp();
            int result = LGBM_BoosterGetNumPredict(voidpp_value(handle), dataIdx, numHandle);
            if (result < 0) {
                delete_int64_tp(numHandle);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                long numClasses = int64_tp_value(numHandle);
                delete_int64_tp(numHandle);
                return numClasses;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Get prediction for training data and validation data.
     * @param dataIdx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
     * @return array with predictions, of size num_class * dataset.num_data
     * @throws LGBMException
     */
    public double[] getPredict(int dataIdx) throws LGBMException {
        if (!isClosed) {
            int allocatedSize = getNumClasses() * (int)getNumPredict(dataIdx);
            SWIGTYPE_p_double buffer = new_doubleArray(allocatedSize);
            SWIGTYPE_p_long_long size = new_int64_tp();
            int result = LGBM_BoosterGetPredict(voidpp_value(handle), dataIdx, size, buffer);
            if (result < 0) {
                delete_doubleArray(buffer);
                delete_int64_tp(size);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                double[] out = new double[(int)int64_tp_value(size)];
                for (int i=0; i<out.length; i++) {
                    out[i] = doubleArray_getitem(buffer, i);
                }
                delete_doubleArray(buffer);
                delete_int64_tp(size);
                return out;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }
    }

    /**
     * Update the model by specifying gradient and Hessian directly (this can be used to support customized loss functions).
     * The length of the arrays referenced by grad and hess must be equal to num_class * num_train_data, this is not
     * verified by the library, the caller must ensure this.
     *
     * @param grad The first order derivative (gradient) statistics
     * @param hess The second order derivative (Hessian) statistics
     * @return true means the update was successfully finished (cannot split anymore), false indicates failure
     * @throws LGBMException
     */
    public boolean updateOneIterCustom(float[] grad, float[] hess) throws LGBMException {
        if (!isClosed) {
            SWIGTYPE_p_float gradHandle = new_floatArray(grad.length);
            for (int i=0; i<grad.length; i++) {
                floatArray_setitem(gradHandle, i, grad[i]);
            }
            SWIGTYPE_p_float hessHandle = new_floatArray(hess.length);
            for (int i=0; i<hess.length; i++) {
                floatArray_setitem(hessHandle, i, hess[i]);
            }
            SWIGTYPE_p_int isFinishedHandle = new_intp();
            int result = LGBM_BoosterUpdateOneIterCustom(voidpp_value(handle), gradHandle, hessHandle, isFinishedHandle);
            if (result < 0) {
                delete_floatArray(gradHandle);
                delete_floatArray(hessHandle);
                delete_intp(isFinishedHandle);
                throw new LGBMException(LGBM_GetLastError());
            } else {
                int isFinished = intp_value(isFinishedHandle);
                delete_floatArray(gradHandle);
                delete_floatArray(hessHandle);
                delete_intp(isFinishedHandle);
                return isFinished == 1;
            }
        } else {
            throw new LGBMException("Booster was already closed");
        }

    }


    /**
     * Calculates the output buffer size for the different prediction types. See the notes at:
     * <a href="https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMat">predictForMat</a> &
     * <a href="https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRow">predictForMatSingleRow</a>
     * for more info.
     *
     * @param rows           the number of rows in the input data
     * @param cols           the number of columns in the input data
     * @param predictionType the type of prediction we are trying to achieve
     * @return number of elements in the output result (size)
     */
    private long outBufferSize(int rows, int cols, PredictionType predictionType) {
        long defaultSize = 2L * rows;
        if (PredictionType.C_API_PREDICT_CONTRIB.equals(predictionType))
            return defaultSize * (cols + 1);
        else if (PredictionType.C_API_PREDICT_LEAF_INDEX.equals(predictionType))
            return defaultSize * iterations;
        else // for C_API_PREDICT_NORMAL & C_API_PREDICT_RAW_SCORE
            return defaultSize;
    }

}
