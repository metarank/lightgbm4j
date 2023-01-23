package io.github.metarank.lightgbm4j;

import io.github.metarank.lightgbm4j.LGBMDataset;
import io.github.metarank.lightgbm4j.LGBMException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;


public class LGBMDatasetTest {
    @Test
    public void testCreateFromFileFail() {
        Assertions.assertThrows(LGBMException.class, () -> {
            LGBMDataset.createFromFile("whatever", "", null);
        });
    }

    @Test void testCreateFromFile() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset.createFromFile("./src/test/resources/categorical.data", "", null);
        });
    }

    @Test void testCreateFromMatFloat() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        });
    }

    @Test void testCreateFromMatDouble() {
        Assertions.assertDoesNotThrow( () -> {
            LGBMDataset.createFromMat(new double[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        });
    }

    @Test void testGetNumData() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        assertEquals(2, ds.getNumData());
        ds.close();
    }

    @Test void testGetNumFeature() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        assertEquals(2, ds.getNumFeatures());
        ds.close();
    }

    @Test void testSetField() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        ds.setField("label", new float[] {1.0f, 0.0f});
        ds.close();
    }

    @Test void testGetFieldFloat() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        ds.setField("label", new float[] {1.0f, 0.0f});
        float[] labels = ds.getFieldFloat("label");
        assertEquals(labels[0], 1.0f);
        assertEquals(labels[1], 0.0f);
        ds.close();
    }

    @Test void testGetFieldInt() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 5, 2, true, "", null);
        ds.setField("label", new float[] {1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
        ds.setField("group", new int[] { 3, 2 });
        int[] groups = ds.getFieldInt("group");
        assertEquals(groups[0], 0);
        assertEquals(groups[1], 3);
        assertEquals(groups[2], 5);
        ds.close();
    }

    @Test void testSetFeatureNames() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        Assertions.assertDoesNotThrow( () -> {
            ds.setFeatureNames(new String[] {"foo", "bar"});
        });
        ds.close();
    }

    @Test void testGetFeatureNames() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        Assertions.assertDoesNotThrow( () -> {
            ds.setFeatureNames(new String[] {"foo", "bar"});
        });
        String[] names = ds.getFeatureNames();
        assertArrayEquals(names, new String[] {"foo", "bar"});
        ds.close();
    }

    @Test void testDoubleClose() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        ds.close();
        ds.close();
    }
}
