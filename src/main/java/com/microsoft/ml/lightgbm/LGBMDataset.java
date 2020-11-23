package com.microsoft.ml.lightgbm;

import java.io.IOException;

import static com.microsoft.ml.lightgbm.lightgbmlib.*;

public class LGBMDataset {
    public SWIGTYPE_p_void handle;
    static {
        try {
            LGBMBooster.loadNative();
        } catch (IOException e) {
            System.out.println("Cannot load native library for your platform");
        }
    }

    public LGBMDataset(SWIGTYPE_p_void handle) {
        this.handle = handle;
    }

    public static LGBMDataset createFromFile(String fileName, String parameters) throws LGBMException {
        SWIGTYPE_p_p_void handle = new_voidpp();
        int result = LGBM_DatasetCreateFromFile(fileName, parameters, null, handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMDataset(voidpp_value(handle));
        }
    }

    public static LGBMDataset createFromMat(float[] data, int rows, int cols, boolean isRowMajor) throws LGBMException {
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
                "",
                null,
                handle
        );
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        } else {
            return new LGBMDataset(voidpp_value(handle));
        }
    }

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

    public void close() throws LGBMException {
        int result = LGBM_DatasetFree(handle);
        if (result < 0) {
            throw new LGBMException(LGBM_GetLastError());
        }
    }
}
