package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class RankingIntegrationTest {
    @Test
    public void testLetor() throws LGBMException, IOException {
        LGBMDataset train = datasetFromResource("/mq2008/train.txt.gz", null);
        LGBMDataset test = datasetFromResource("/mq2008/test.txt.gz", train);
        LGBMBooster booster = LGBMBooster.create(train, "objective=lambdarank metric=ndcg lambdarank_truncation_level=10 max_depth=5 learning_rate=0.1 num_leaves=8");
        booster.addValidData(test);
        for (int i=0; i<100; i++) {
            booster.updateOneIter();
            double[] eval1 = booster.getEval(0);
            double[] eval2 = booster.getEval(1);
            System.out.println("train " + eval1[0] + " test " + eval2[0]);
            assertTrue(eval1[0] > 0.5);
        }
        String[] names = booster.getFeatureNames();
        double[] weights = booster.featureImportance(0, LGBMBooster.FeatureImportanceType.GAIN);
        assertTrue(names.length > 0);
        assertTrue(weights.length > 0);
        booster.close();
        train.close();
        test.close();
    }


    private static LGBMDataset datasetFromResource(String file, LGBMDataset parent) throws LGBMException, IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(RankingIntegrationTest.class.getResourceAsStream(file))));
        ArrayList<String> lines = reader.lines().map(line -> {
            int commentIndex = line.indexOf('#');
            if (commentIndex >= 0) {
                return line.substring(0, commentIndex);
            } else {
                return line;
            }
        }).collect(Collectors.toCollection(ArrayList::new));
        int maxFeatureId = 0; // features are 1-indexed!
        Set<Integer> queriesSet = new HashSet<>();
        for (String line: lines) {
            String[] tokens = line.split(" ");
            int group = Integer.parseInt(tokens[1].split(":")[1]);
            queriesSet.add(group);
            for (int i = 2; i < tokens.length; i++) {
                String[] parts = tokens[i].split(":");
                int featureId = Integer.parseInt(parts[0]);
                if (featureId > maxFeatureId) {
                    maxFeatureId = featureId;
                }
            }
        }

        int rows = lines.size();
        int queries = queriesSet.size();
        double[] features = new double[maxFeatureId * rows];
        float[] labels = new float[rows];
        int[] groups = new int[queries];
        String[] featureNames = new String[maxFeatureId];
        for (int i=1; i <= maxFeatureId; i++) {
            featureNames[i-1] = "f"+i;
        }
        int lastGroup = Integer.MIN_VALUE;
        int lastCount = 0;
        int groupIndex = 0;
        for (int row = 0; row < rows; row++) {
            String line = lines.get(row);
            String[] tokens = line.split(" ");
            float label = Float.parseFloat(tokens[0]);
            labels[row] = label;
            int group = Integer.parseInt(tokens[1].split(":")[1]);
            if (group != lastGroup) {
                // next query
                if (lastCount > 0) {
                    // so it's not the first one
                    groups[groupIndex] = lastCount;
                    groupIndex++;
                }
                lastGroup = group;
                lastCount = 1;
            } else {
                lastCount++;
            }

            for (int i=2; i < tokens.length; i++) {
                String[] feature = tokens[i].split(":");
                int id = Integer.parseInt(feature[0]) - 1;
                double value = Double.parseDouble(feature[1]);
                features[row * maxFeatureId + id] = value;
            }
        }
        groups[groupIndex] = lastCount;


        reader.close();
        LGBMDataset dataset = LGBMDataset.createFromMat(features, rows, maxFeatureId, true, "", parent);
        dataset.setFeatureNames(featureNames);
        dataset.setField("label", labels);
        dataset.setField("group", groups);
        return dataset;
    }

}
