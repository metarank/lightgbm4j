package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.PredictionType;
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
            LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
            LGBMBooster.create(ds, "");
        });
    }

    @Test
    public void testCreateFree() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
            LGBMBooster booster = LGBMBooster.create(ds, "");
            ds.close();
            booster.close();
        });
    }

    @Test
    public void testUpdateOneIter() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        ds.close();
        booster.close();
        assertTrue(finished);
    }

    @Test
    public void testSaveModelToString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
        ds.close();
        booster.close();
        assertFalse(model.isEmpty(), "model string should not be empty");
    }

    @Test
    public void testLoadSaveString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
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
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String[] names = booster.getFeatureNames();
        assertTrue(names.length > 0, "feature names should be present");
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloat() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred = booster.predictForMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, PredictionType.C_API_PREDICT_NORMAL);
        assertTrue(pred.length > 0, "predicted values should not be empty");
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatDouble() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred = booster.predictForMat(new double[] {1.0, 1.0, 1.0, 1.0}, 2, 2, true, PredictionType.C_API_PREDICT_NORMAL);
        assertTrue(pred.length > 0, "predicted values should not be empty");
        ds.close();
        booster.close();
    }

    @Test
    public void testAddValidData() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
        LGBMBooster booster = LGBMBooster.create(ds, "");
        booster.addValidData(ds);
        ds.close();
        booster.close();
    }

    @Test
    public void testGetEval() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification");
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        boolean finished = booster.updateOneIter();
        double[] eval = booster.getEval(0);
        assertTrue(eval.length > 0);
        assertTrue(eval[0] > 0);
        dataset.close();
        booster.close();
    }

    @Test
    public void testGetEvalNames() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification");
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.addValidData(dataset);
        String[] eval = booster.getEvalNames();
        assertTrue(eval.length > 0);
        dataset.close();
        booster.close();
    }

    @Test
    public void testFeatureImportance() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification");
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        String[] names = booster.getFeatureNames();
        double[] importance = booster.featureImportance(0, LGBMBooster.FeatureImportanceType.GAIN);
        assertTrue(names.length > 0);
        assertTrue(importance.length > 0);
        dataset.close();
        booster.close();
    }

    @Test
    public void testPredictForMatSingleRow() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification");
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        double pred1 = booster.predictForMatSingleRow(new double[] {1,2,3,4,5,6,7,8,9}, PredictionType.C_API_PREDICT_NORMAL);
        assertTrue(pred1 > 0);
        double pred2 = booster.predictForMatSingleRow(new float[] {1,2,3,4,5,6,7,8,9}, PredictionType.C_API_PREDICT_NORMAL);
        assertTrue(pred2 > 0);
        dataset.close();
        booster.close();
    }

}
