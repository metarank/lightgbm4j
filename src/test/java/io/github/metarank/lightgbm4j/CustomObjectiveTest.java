package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;

import java.io.IOException;

public class CustomObjectiveTest {

    @Test
    void testCancerCustomObjective() throws LGBMException, IOException {
        LGBMDataset dataset = CancerIntegrationTest.datasetFromFile();
        LGBMBooster booster = LGBMBooster.create(dataset, "objective=none metric=rmse label=name:Classification");
        // actual ground truth label values
        float y[] = dataset.getFieldFloat("label");
        for (int it=0; it<10; it++) {
            // predictions for current iteration
            double[] yhat = booster.getPredict(0);
            float[] grad = new float[y.length];
            float[] hess = new float[y.length];
            for (int i=0; i<y.length; i++) {
                // 1-st derivative of squared error
                grad[i] = (float)(2 * (yhat[i]-y[i]));
                // 2-nd derivative of squared error
                hess[i] = (float)(0 * (yhat[i]-y[i]) + 2);
            }
            booster.updateOneIterCustom(grad, hess);
            // print the computed average error
            double[] err = booster.getEval(0);
            System.out.println("it " + it + " err=" + err[0]);
        }
        booster.close();
        dataset.close();
    }
}
