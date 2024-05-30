package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.PredictionType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

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
        Assertions.assertDoesNotThrow(() -> {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
            LGBMBooster.create(ds, "");
        });
    }

    @Test
    public void testCreateFree() {
        Assertions.assertDoesNotThrow(() -> {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
            LGBMBooster booster = LGBMBooster.create(ds, "");
            ds.close();
            booster.close();
        });
    }

    @Test
    public void testUpdateOneIter() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        ds.close();
        booster.close();
        assertTrue(finished);
    }

    @Test
    public void testSaveModelToString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
        ds.close();
        booster.close();
        assertFalse(model.isEmpty(), "model string should not be empty");
    }

    @Test
    public void testLoadSaveString() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
        ds.close();
        booster.close();
        Assertions.assertDoesNotThrow(() -> {
            LGBMBooster loaded = LGBMBooster.loadModelFromString(model);
            loaded.close();
        });
    }

    @Test
    public void testGetFeatureNames() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        String[] names = booster.getFeatureNames();
        assertTrue(names.length > 0, "feature names should be present");
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloatNormalPrediction() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred;
        for (int i = 0; i < 10; i++) {
            // repeat multiple times to ensure it works
            // offline tests showed that one execution can work, while in multiple iterations it fails
            // valid for all types of predictions
            pred = booster.predictForMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, PredictionType.C_API_PREDICT_NORMAL);
            assertTrue(pred.length > 0, "predicted values should not be empty");
        }
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloatRawScorePrediction() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred;
        for (int i = 0; i < 10; i++) {
            pred = booster.predictForMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, PredictionType.C_API_PREDICT_RAW_SCORE);
            assertTrue(pred.length > 0, "predicted values should not be empty");
        }
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloatLeafIndexPrediction() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred;
        for (int i = 0; i < 10; i++) {
            pred = booster.predictForMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, PredictionType.C_API_PREDICT_LEAF_INDEX);
            assertTrue(pred.length > 0, "predicted values should not be empty");
        }
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatFloatContributionPrediction() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred;
        for (int i = 0; i < 10; i++) {
            pred = booster.predictForMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, PredictionType.C_API_PREDICT_CONTRIB);
            assertTrue(pred.length > 0, "predicted values should not be empty");
        }
        ds.close();
        booster.close();
    }

    @Test
    public void testPredictForMatDouble() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        boolean finished = booster.updateOneIter();
        double[] pred;
        for (int i = 0; i < 10; i++) {
            pred = booster.predictForMat(new double[]{1.0, 1.0, 1.0, 1.0}, 2, 2, true, PredictionType.C_API_PREDICT_NORMAL);
            assertTrue(pred.length > 0, "predicted values should not be empty");
        }
        ds.close();
        booster.close();
    }

    @Test
    public void testAddValidData() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        booster.addValidData(ds);
        ds.close();
        booster.close();
    }

    @Test
    public void testGetEval() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
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
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.addValidData(dataset);
        String[] eval = booster.getEvalNames();
        assertTrue(eval.length > 0);
        dataset.close();
        booster.close();
    }

    @Test
    public void testFeatureImportance() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
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
    public void testPredictForMatSingleRowNormalPrediction() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        for (int i = 0; i < 10; i++) {
            double pred1 = booster.predictForMatSingleRow(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_NORMAL);
            assertTrue(pred1 > 0);
            double pred2 = booster.predictForMatSingleRow(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_NORMAL);
            assertTrue(pred2 > 0);
        }
        dataset.close();
        booster.close();
    }

    @Test
    public void testPredictForMatSingleRowRawScorePrediction() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        for (int i = 0; i < 10; i++) {
            // we assert that the result is finite, as it get both negative and positive values
            double pred1 = booster.predictForMatSingleRow(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_RAW_SCORE);
            assertTrue(Double.isFinite(pred1));
            double pred2 = booster.predictForMatSingleRow(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_RAW_SCORE);
            assertTrue(Double.isFinite(pred2));
        }
        dataset.close();
        booster.close();
    }

    @Test
    public void testPredictForMatSingleRowLeafIndexPrediction() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        for (int i = 0; i < 10; i++) {
            // we assert that the result is finite, as it get both negative and positive values
            double pred1 = booster.predictForMatSingleRow(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_LEAF_INDEX);
            assertTrue(Double.isFinite(pred1));
            double pred2 = booster.predictForMatSingleRow(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_LEAF_INDEX);
            assertTrue(Double.isFinite(pred2));
        }
        dataset.close();
        booster.close();
    }

    @Test
    public void testPredictForMatSingleRowContributionPrediction() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        for (int i = 0; i < 10; i++) {
            // we assert that the result is finite, as it get both negative and positive values
            double pred1 = booster.predictForMatSingleRow(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_CONTRIB);
            assertTrue(Double.isFinite(pred1));
            double pred2 = booster.predictForMatSingleRow(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_CONTRIB);
            assertTrue(Double.isFinite(pred2));
        }
        dataset.close();
        booster.close();
    }

    @Test
    void testCreateByReference() throws LGBMException {
        LGBMDataset dataset1 = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMDataset dataset2 = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", dataset1);
        LGBMBooster booster = LGBMBooster.create(dataset1, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();

        booster.addValidData(dataset2);
        double[] train = booster.getEval(0);
        double[] test = booster.getEval(1);
        assertEquals(train[0], test[0], 0.001);
    }

    @Test void testGetNumClasses() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        assertEquals(booster.getNumClasses(), 1);
        dataset.close();
        booster.close();
    }

    @Test void testGetNumPredict() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        assertEquals(booster.getNumPredict(0), 116);
        dataset.close();
        booster.close();
    }

    @Test void testGetPredict() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        double[] preds = booster.getPredict(0);
        assertEquals(preds.length, 116);
        dataset.close();
        booster.close();
    }

    @Test void testUpdateOneIterCustom() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=none metric=rmse label=name:Classification");
        int size = dataset.getNumData();
        booster.updateOneIterCustom(randomArray(size), randomArray(size));
        booster.updateOneIterCustom(randomArray(size), randomArray(size));
        booster.updateOneIterCustom(randomArray(size), randomArray(size));
        double[] preds = booster.getPredict(0);
        assertEquals(preds.length, 116);
        dataset.close();
        booster.close();
    }

    @Test void testDoubleClose() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(ds, "");
        booster.close();
        assertThrows(LGBMException.class, booster::close);

    }

    @Test void testUseAfterClose() throws LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        booster.updateOneIter();
        booster.updateOneIter();
        booster.updateOneIter();
        booster.close();
        assertThrows(LGBMException.class, () -> booster.getPredict(0));
    }

    private float[] randomArray(int size) {
        float[] result = new float[size];
        Random rnd = new Random();
        for (int i=0; i<size; i++) {
            result[i] = rnd.nextFloat();
        }
        return result;
    }

}
