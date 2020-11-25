package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

public class CancerIntegrationTest {

    @Test
    public void testCancer() throws IOException, LGBMException {
        LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification");
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


}
