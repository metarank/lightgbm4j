package com.microsoft.ml.lightgbm;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LGBMBoosterTest {
    @Test
    void testLoad() {
        assertTrue(LGBMBooster.isNativeLoaded());
    }

    @Test
    public void testLoadModelFromStringFail() {
        Assertions.assertThrows(LGBMException.class, () -> {
            LGBMBooster.loadModelFromString("whatever");
        });
    }

    @Test
    public void testLoadModelFromModelfileFail() {
        Assertions.assertThrows(LGBMException.class, () -> {
            LGBMBooster.createFromModelfile("whatever");
        });
    }

    @Test
    public void testCreate() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
            LGBMBooster.create(ds, "");
        });
    }

    @Test
    public void testCreateFree() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
            LGBMBooster booster = LGBMBooster.create(ds, "");
            ds.close();
            booster.close();
        });
    }

    @Test
    public void testUpdateOneIter() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        ds.close();
        booster.close();
        assertTrue(finished);
    }

    @Test
    public void testSaveModelToString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
        ds.close();
        booster.close();
        assertFalse(model.isEmpty(), "model string should not be empty");
    }

    @Test
    public void testLoadSaveString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
        ds.close();
        booster.close();
        Assertions.assertDoesNotThrow( () -> {
            LGBMBooster loaded = LGBMBooster.loadModelFromString(model);
            loaded.close();
        });
    }

    @Test
    public void testGetFeatureNames() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String[] names = booster.getFeatureNames();
        assertTrue(names.length > 0, "feature names should be present");
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloat() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred = booster.predictForMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        assertTrue(pred.length > 0, "predicted values should not be empty");
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatDouble() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred = booster.predictForMat(new double[] {1.0, 1.0, 1.0, 1.0}, 2, 2, true);
        assertTrue(pred.length > 0, "predicted values should not be empty");
        ds.close();
        booster.close();
    }
}
