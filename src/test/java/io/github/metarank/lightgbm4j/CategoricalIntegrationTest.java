package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;

public class CategoricalIntegrationTest {
    @Test
    void testCategorical() throws LGBMException {
        LGBMDataset ds = LGBMDataset.createFromFile("./src/test/resources/categorical.data", "", null);
        ds.setFeatureNames(new String[]{"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"});
        String params = "objective=binary label=name:Classification categorical_feature=f0,f1,f2,f3,f4,f5,f6,f7";
        LGBMBooster booster = LGBMBooster.create(ds, params);
        for (int i=0; i<10; i++) {
            booster.updateOneIter();
            double[] eval1 = booster.getEval(0);
            System.out.println("train " + eval1[0]);
        }
    }
}
