package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.*;

import java.io.IOException;

import static com.microsoft.ml.lightgbm.lightgbmlib.*;

public class LGBMDataset implements AutoCloseable {
    public SWIGTYPE_p_void handle;
    static {
        try {
            LGBMBooster.loadNative();
        } catch (IOException e) {
            System.out.println("Cannot load native library for your platform");
        }
    }

    LGBMDataset(SWIGTYPE_p_void handle) {
        this.handle = handle;
    }

    /**
     * Load dataset from file (like LightGBM CLI version does).
     * @param fileName  The name of the file
     * @param parameters Additional parameters
     * @return
     * @throws LGBMException
     */
    public static LGBMDataset createFromFile(String fileName, String parameters, LGBMDataset reference) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        int result = LGBM_DatasetCreateFromFile(fileName, parameters, reference == null ? null : reference.handle, handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMDataset(voidpp_value(handle));
        }
    }

    /**
     * Create dataset from dense float[] matrix.
     * @param data input matrix
     * @param rows number of rows
     * @param cols number of cols
     * @param isRowMajor is a row-major encoding used?
     * @param parameters extra parameters
     * @return
     * @throws LGBMException
     */
    public static LGBMDataset createFromMat(float[] data, int rows, int cols, boolean isRowMajor, String parameters, LGBMDataset reference) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        SWIGTYPE_p_float dataBuffer = new_floatArray(data.length);
        for (int i = 0; i < data.length; i++) {
            floatArray_setitem(dataBuffer, i, data[i]);
        }

        int result = LGBM_DatasetCreateFromMat(
                float_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT32,
                rows,
                cols,
                isRowMajor ? 1 : 0,
                parameters,
                reference == null ? null : reference.handle,
                handle
        );
        delete_floatArray(dataBuffer);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMDataset(voidpp_value(handle));
        }
    }
    /**
     * Create dataset from dense double[] matrix.
     * @param data input matrix
     * @param rows number of rows
     * @param cols number of cols
     * @param isRowMajor is a row-major encoding used?
     * @param parameters extra parameters
     * @param reference to align bin mappers with other dataset
     * @return
     * @throws LGBMException
     */
    public static LGBMDataset createFromMat(double[] data, int rows, int cols, boolean isRowMajor, String parameters, LGBMDataset reference) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        SWIGTYPE_p_double dataBuffer = new_doubleArray(data.length);
        for (int i = 0; i < data.length; i++) {
            doubleArray_setitem(dataBuffer, i, data[i]);
        }
        int result = LGBM_DatasetCreateFromMat(
                double_to_voidp_ptr(dataBuffer),
                C_API_DTYPE_FLOAT64,
                rows,
                cols,
                isRowMajor ? 1 : 0,
                parameters,
                reference == null ? null : reference.handle,
                handle
        );
        delete_doubleArray(dataBuffer);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMDataset(voidpp_value(handle));
        }
    }

    /**
     * Get number of data points.
     * @return  number of data points
     * @throws LGBMException
     */
    public int getNumData() throws LGBMException {
        SWIGTYPE_p_int numDataP = new_intp();
        int result = LGBM_DatasetGetNumData(handle, numDataP);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int numData = intp_value(numDataP);
            delete_intp(numDataP);
            return numData;
        }
    }

    /**
     * Get number of features.
     * @return number of features
     * @throws LGBMException
     */
    public int getNumFeatures() throws LGBMException {
        SWIGTYPE_p_int numFeaturesP = new_intp();
        int result = LGBM_DatasetGetNumFeature(handle, numFeaturesP);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            int numFeatures = intp_value(numFeaturesP);
            delete_intp(numFeaturesP);
            return numFeatures;
        }
    }

    /**
     * Set feature names
     * @param featureNames a list of names.
     * @throws LGBMException
     */
    public void setFeatureNames(String[] featureNames) throws LGBMException {
        int result = LGBM_DatasetSetFeatureNames(handle, featureNames, featureNames.length);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Dumps dataset into a file for debugging.
     * @param fileName
     * @throws LGBMException
     */
    public void dumpText(String fileName) throws LGBMException {
        int result = LGBM_DatasetDumpText(handle, fileName);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Sets a double field. label and weight fields can only be float[]
     * @param fieldName
     * @param data
     * @throws LGBMException
     */
    public void setField(String fieldName, double[] data) throws LGBMException {
        if (fieldName.equals("label")) throw new LGBMException("label can only be float[]");
        if (fieldName.equals("weight")) throw new LGBMException("weight can only be float[]");
        SWIGTYPE_p_double dataBuffer = new_doubleArray(data.length);
        for (int i = 0; i < data.length; i++) {
            doubleArray_setitem(dataBuffer, i, data[i]);
        }

        int result = LGBM_DatasetSetField(handle, fieldName, double_to_voidp_ptr(dataBuffer), data.length, C_API_DTYPE_FLOAT64);
        delete_doubleArray(dataBuffer);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Sets an int field. It can only accept the group field.
     * @param fieldName
     * @param data
     * @throws LGBMException
     */
    public void setField(String fieldName, int[] data) throws LGBMException {
        if (!fieldName.equals("group")) throw new LGBMException("only group field can be int[]");
        SWIGTYPE_p_int dataBuffer = new_intArray(data.length);
        for (int i = 0; i < data.length; i++) {
            intArray_setitem(dataBuffer, i, data[i]);
        }

        int result = LGBM_DatasetSetField(handle, fieldName, int_to_voidp_ptr(dataBuffer), data.length, C_API_DTYPE_INT32);
        delete_intArray(dataBuffer);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Sets a double field. label and weight fields can only be float[]
     * @param fieldName
     * @param data
     * @throws LGBMException
     */
    public void setField(String fieldName, float[] data) throws LGBMException {
        SWIGTYPE_p_float dataBuffer = new_floatArray(data.length);
        for (int i = 0; i < data.length; i++) {
            floatArray_setitem(dataBuffer, i, data[i]);
        }

        int result = LGBM_DatasetSetField(handle, fieldName,float_to_voidp_ptr(dataBuffer), data.length, C_API_DTYPE_FLOAT32);
        delete_floatArray(dataBuffer);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }

    /**
     * Gets feature names from dataset, if dataset supports it.
     * @return list of feature names
     * @throws LGBMException
     */
    public String[] getFeatureNames() throws LGBMException {
        SWIGTYPE_p_void arrayHandle = LGBM_DatasetGetFeatureNamesSWIG(handle);
        String[] names = StringArrayHandle_get_strings(arrayHandle);
        StringArrayHandle_free(arrayHandle);
        return names;
    }

    /**
     * Deallocate all native memory for the LightGBM dataset.
     * @throws LGBMException
     */
    @Override
    public void close() throws LGBMException {
        int result = LGBM_DatasetFree(handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }
}
