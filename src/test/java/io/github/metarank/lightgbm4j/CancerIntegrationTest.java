package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

public class CancerIntegrationTest {


    @ParameterizedTest
    @MethodSource("datasets")
    public void testCancer(LGBMDataset dataset) throws LGBMException {
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
        for (int i=0; i<10; i++) {
            booster.updateOneIter();
            double[] eval = booster.getEval(0);
            System.out.println(eval[0]);
            assertTrue(eval[0] > 0);
        }
        String[] names = booster.getFeatureNames();
        double[] weights = booster.featureImportance(0, LGBMBooster.FeatureImportanceType.GAIN);
        assertTrue(names.length > 0);
        assertTrue(weights.length > 0);
        booster.close();
        dataset.close();
    }

    public static Stream<Arguments> datasets() throws LGBMException, IOException {
        return Stream.of(
                Arguments.of(datasetFromFile()),
                Arguments.of(datasetFromMat()),
                Arguments.of(datasetReadmeExample())
        );
    }
    public static LGBMDataset datasetFromFile() throws LGBMException, IOException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
        return dataset;
    }

    private static LGBMDataset datasetFromMat() throws LGBMException, IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(CancerIntegrationTest.class.getResourceAsStream("/cancer.csv")));
        String[] names = Arrays.copyOfRange(reader.readLine().split(","), 0, 9);
        ArrayList<Double> valuesBuffer = new ArrayList<>();
        ArrayList<Float> labelsBuffer = new ArrayList<>();
        String line = reader.readLine();
        int rows = 0;
        while (line != null) {
            rows++;
            String[] parts = line.split(",");
            for (int i=0; i < parts.length; i++) {
                if (i < parts.length - 1) {
                    valuesBuffer.add(Double.parseDouble(parts[i]));
                } else {
                    labelsBuffer.add(Float.parseFloat(parts[i]));
                }
            }
            line = reader.readLine();
        }
        double[] values = new double[valuesBuffer.size()];
        for (int i=0; i < valuesBuffer.size(); i++) {
            values[i] = valuesBuffer.get(i);
        }
        float[] labels = new float[labelsBuffer.size()];
        for (int i=0; i < labelsBuffer.size(); i++) {
            labels[i] = labelsBuffer.get(i);
        }
        reader.close();
        LGBMDataset dataset = LGBMDataset.createFromMat(values, rows, names.length, true, "", null);
        dataset.setFeatureNames(names);
        dataset.setField("label", labels);
        return dataset;
    }

    public static LGBMDataset datasetReadmeExample() throws LGBMException {
        String[] columns = new String[] {
                "Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1"
        };
        double[] values = new double[] {
                71,30.3,102,8.34,2.098344,56.502,8.13,4.2989,200.976,
                66,27.7,90,6.042,1.341324,24.846,7.652055,6.7052,225.88,
                75,25.7,94,8.079,1.8732508,65.926,3.74122,4.49685,206.802,
                78,25.3,60,3.508,0.519184,6.633,10.567295,4.6638,209.749,
                69,29.4,89,10.704,2.3498848,45.272,8.2863,4.53,215.769,
                85,26.6,96,4.462,1.0566016,7.85,7.9317,9.6135,232.006,
                76,27.1,110,26.211,7.111918,21.778,4.935635,8.49395,45.843,
                77,25.9,85,4.58,0.960273333,13.74,9.75326,11.774,488.829,
                45,21.30394858,102,13.852,3.4851632,7.6476,21.056625,23.03408,552.444,
                45,20.82999519,74,4.56,0.832352,7.7529,8.237405,28.0323,382.955,
                49,20.9566075,94,12.305,2.853119333,11.2406,8.412175,23.1177,573.63,
                34,24.24242424,92,21.699,4.9242264,16.7353,21.823745,12.06534,481.949,
                42,21.35991456,93,2.999,0.6879706,19.0826,8.462915,17.37615,321.919,
                68,21.08281329,102,6.2,1.55992,9.6994,8.574655,13.74244,448.799,
                51,19.13265306,93,4.364,1.0011016,11.0816,5.80762,5.57055,90.6,
                62,22.65625,92,3.482,0.790181867,9.8648,11.236235,10.69548,703.973
        };

        float[] labels = new float[] {
                0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1
        };
        LGBMDataset dataset = LGBMDataset.createFromMat(values, 16, columns.length, true, "", null);
        dataset.setFeatureNames(columns);
        dataset.setField("label", labels);
        return dataset;
    }

}
