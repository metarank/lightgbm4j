# LightGBM4j: a java wrapper for LightGBM

[![CI Status](https://github.com/metarank/lightgbm4j/workflows/Java%20CI%20with%20Maven/badge.svg)](https://github.com/metarank/lightgbm4j/actions)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j/badge.svg?style=plastic)](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LightGBM4j** is a zero-dependency Java wrapper for the LightGBM project. Its main goal is to provide a 1-1 mapping for all
LightGBM API methods in a Java-friendly flavor. 

## Purpose

LightGBM itself has a SWIG-generated JNI interface, which is possible to use directly from Java. The problem with SWIG wrappers
is that they are extremely low-level. For example, to pass a java array thru SWIG, you need to do something horrible:
```java
        SWIGTYPE_p_float dataBuffer = new_floatArray(input.length);
        for (int i = 0; i < input.length; i++) {
            floatArray_setitem(dataBuffer, i, input[i]);
        }
        int result = <...>
        if (result < 0) {
            delete_floatArray(dataBuffer);
            throw new Exception(LGBM_GetLastError());
        } else {
            delete_floatArray(dataBuffer);
            <...>
        }
```
This wrapper does all the dirty job for you:
* exposes native java types for all supported API methods (so `float[]` instead `SWIGTYPE_p_float`)
* handles manual memory management internally (so you don't need to care about JNI memory leaks)
* supports both `float[]` and `double[]` API flavours.
* reduces the amount of boilerplate for basic tasks.

The library is in an early development stage and does not cover all 100% of LightGBM API, but the eventual future 
goal will be merging with the upstream LightGBM and becoming an official Java binding for the project.

## Installation

To install, use the following maven coordinates:
```xml
<dependency>
  <groupId>io.github.metarank</groupId>
  <artifactId>lightgbm4j</artifactId>
  <version>4.4.0-1</version>
</dependency>
```

Versioning schema attempts to match the upstream, but with extra `-N` suffix, if there were a couple of extra lightgbm4j-specific
changes released on top.

### MacOS & Linux native library dependencies installation 

LightGBM native library requires the `libomp` dependency for OpenMP support, but this library is often missing on some systems by default.

For MacOS:
```
brew install libomp
```

For Debian Linux:
```
apt install libgomp1
```

### GPU support

It is possible to force GPU support for a training:
* rebuild the [LightGBM with GPU support]: use `-DUSE_CUDA=1 -DUSE_SWIG=ON` CMake options. You should also match the native/JNI versions precisely.
* LightGBM4j loads native libraries by default from bundled resources. This can be overridden by setting the `LIGHTGBM_NATIVE_LIB_PATH` environment variable. It should point to a directory with `lib_lightgbm.so` and `lib_lightgbm_swig.so` files (or with `dll`/`dylib` extensions on Windows/MacOS).

If the native override was able to successfully load a custom library you've built, then you'll see the following line in logs:
```
LIGHTGBM_NATIVE_LIB_PATH is set: loading /home/user/code/LightGBM/lib_lightgbm.so
LIGHTGBM_NATIVE_LIB_PATH is set: loading /home/user/code/LightGBM/lib_lightgbm_swig.so
```

## Usage

There are two main classes available: 
1. `LGBMDataset` to manage input training and validation data.
2. `LGBMBooster` to do training and inference.

All the public API methods in these classes should map to the [LightGBM C API](https://lightgbm.readthedocs.io/en/latest/C-API.html) methods directly.

Note that both `LGBMBooster` and `LGBMDataset` classes contain handles of native memory
data structures from the LightGBM, so you need to explicitly call `.close()` when they are not used. Otherwise, you may catch
a native code memory leak.

To load an existing model and run it:
```java
LGBMBooster loaded = LGBMBooster.loadModelFromString(model);
float[] input = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
double[] pred = booster.predictForMat(input, 2, 2, true);
```

To load a dataset from a java matrix:
```java
float[] matrix = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
LGBMDataset ds = LGBMDataset.createFromMat(matrix, 2, 2, true, "", null);
```

There are some rough parts in the LightGBM API in loading the dataset from matrices:
* `createFromMat` parameters cannot set the [label](https://lightgbm.readthedocs.io/en/latest/Parameters.html#label_column) or [weight](https://lightgbm.readthedocs.io/en/latest/Parameters.html#weight_column) column. 
So if you do `parameters = "label=some_column_name"`, it will be ignored by the LightGBM.
* label/weight/group columns are magical and should NOT be included in the input matrix for
`createFromMat`
* to set these magical columns, you need to explicitly call `LGBMDataset.setField()` method.
* `label` and `weight` columns [must be](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField) `float[]`
* `group` and `position` column [must be](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField) `int[]`

A full example of loading dataset from a matrix for a cancer dataset:
```java
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
```

Also, see [a working example](https://github.com/metarank/lightgbm4j/blob/main/src/test/java/io/github/metarank/lightgbm4j/CancerIntegrationTest.java) 
of different ways to deal with input datasets in the LightGBM4j tests.
## Example

```java
// cancer dataset from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
// with labels altered to fit the [0,1] range
LGBMDataset train = LGBMDataset.createFromFile("cancer.csv", "header=true label=name:Classification", null);
LGBMDataset test = LGBMDataset.createFromFile("cancer-test.csv", "header=true label=name:Classification", train);
LGBMBooster booster = LGBMBooster.create(train, "objective=binary label=name:Classification");
booster.addValidData(test);

for (int i=0; i<10; i++) {
     booster.updateOneIter();
     double[] evalTrain = booster.getEval(0);
     double[] evalTest = booster.getEval(1);
     System.out.println("train: " + eval[0] + " test: " + );
}
booster.close();
train.close();
test.close();
```

### Categorical features

LightGBM supports defining features as categorical. To make this work with LightGBM4j, you need to do the following:

* Set their names with `setFeatureNames` so you can reference them later in options
* Mark them as `categorical_feature` in booster options.

Given the dataset file in the LibSVM format, where categories are index-encoded:

```
1 0:7 1:2 2:3 3:20 4:15 5:38 6:29 7:201
0 0:5 1:15 2:2 3:1859 4:1 5:156 6:164 7:2475
0 0:2 1:12 2:6 3:648 4:13 5:29 6:38 7:201
1 0:10 1:26 2:5 3:1235 4:14 5:82 6:205 7:931
0 0:6 1:18 2:1 3:737 4:12 5:224 6:162 7:2176
0 0:4 1:12 3:1845 4:18 5:83 6:49 7:1491
0 0:3 2:3 3:1652 4:20 5:2 6:180 7:332
0 0:3 1:21 2:3 3:2010 4:16 5:216 6:69 7:911
0 0:3 1:3 3:1555 4:1 5:84 6:81 7:1192
0 0:8 1:2 2:6 3:1008 4:16 5:216 6:228 7:130
```

You can load and use them in the following way:

```java
LGBMDataset ds = LGBMDataset.createFromFile("./src/test/resources/categorical.data", "", null);
ds.setFeatureNames(new String[]{"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"});
String params = "objective=binary label=name:Classification categorical_feature=f0,f1,f2,f3,f4,f5,f6,f7";
LGBMBooster booster = LGBMBooster.create(ds, params);
for (int i=0; i<10; i++) {
    booster.updateOneIter();
    double[] eval1 = booster.getEval(0);
    System.out.println("train " + eval1[0]);
}
```

### Position bias removal

LightGBM 4.1+ can perform a [position-bias aware LTR/LambdaMART](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#support-for-position-bias-treatment) training. To perform it with lightgbm4j you need to explicitly define the `position` field as described in the upstream LightGBM docs:

```java
float[] matrix = new float[] {
        // query group 1
        1.0f, 2.0f, // doc1
        3.0f, 4.0f, // doc2
        // query group 2
        1.0f, 2.0f, // doc1
        3.0f, 4.0f}; // doc2
LGBMDataset ds = LGBMDataset.createFromMat(matrix, 4, 2, true, "", null);
ds.setField("label", new float[] {1.0, 0.0, 1.0, 0.0}); // set relevance labels
ds.setField("group", new int[] {2, 2}); // set document-to-group mapping
ds.setField("position", new int[] {0, 1, 2, 3, 0, 1, 2, 3}); // bias classes
LGBMBooster booster = LGBMBooster.create(ds, "objective=lambdarank");
```

### Custom objectives

LightGBM4j supports using custom objective functions, but it doesn't provide any high-level wrappers as python API does. 

LightGBM needs a tuple of 1st and 2nd order derivatives (gradients and hessians) computed for each datapoint. With LightGBM4j it looks like this for an MSE metric:

```java
LGBMDataset dataset = LGBMDataset.createFromFile("cancer.csv", "header=true label=name:Classification", null);
LGBMBooster booster = LGBMBooster.create(dataset, "objective=none metric=rmse label=name:Classification");
// actual ground truth label values
float y[] = dataset.getFieldFloat("label");

for (int it=0; it<10; it++) {
    // predictions for current iteration
    double[] yhat = booster.getPredict(0); // 0 - training dataset
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
```

Note the following change in the LightGBM4 behavior:

* you need to set `objective=none metric=<eval metric>` parameters to signal that we're going to use custom objective. Otherwise the LightGBM will complain on incorrect objective.

### Low-latency predictions

Raw LGBM API exposes multiple low-level ways to make predictions with lower latency:
* Instead of `predictForMat`, you can use a single-row optimized `predictForMatSingleRow` method
* LGBM my default still uses paralellism for single-row predictions, which still affects final latency. Opt for including `threads=1` parameter for your prediction method calls.
* LightGBM4J also exposes a low-level `predictForMatSingleRowFast` method, which pre-allocates internal structures once, and reuses them on each next call.

#### Single-row prediction

```java
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
```

#### Single-row fast prediction

```java
LGBMDataset dataset = LGBMDataset.createFromFile("src/test/resources/cancer.csv", "header=true label=name:Classification", null);
LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
booster.updateOneIter();
booster.updateOneIter();
booster.updateOneIter();
LGBMBooster.FastConfig config = booster.predictForMatSingleRowFastInit(PredictionType.C_API_PREDICT_NORMAL, C_API_DTYPE_FLOAT32,9, "");
double pred = booster.predictForMatSingleRowFast(config, new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, PredictionType.C_API_PREDICT_NORMAL);
assertTrue(Double.isFinite(pred));
config.close();
dataset.close();
booster.close();

```

## Supported platforms

This code is tested to work well with Linux (Ubuntu 20.04), Windows (Server 2019) and MacOS 10.15/11. Mac M1 is also supported.
Supported Java versions are 11, 17 and 21.

## LightGBM API Coverage

Not all LightGBM API methods are covered in this wrapper. PRs are welcome!

Supported methods:
* [LGBM_BoosterAddValidData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterAddValidData)
* [LGBM_BoosterCreate](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCreate)
* [LGBM_BoosterCreateFromModelfile](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCreateFromModelfile)
* [LGBM_BoosterFree](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterFree)
* [LGBM_BoosterGetEval](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEval)
* [LGBM_BoosterGetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetFeatureNames)
* [LGBM_BoosterFeatureImportance](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterFeatureImportance)
* [LGBM_BoosterGetEvalNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEvalNames)
* [LGBM_BoosterGetNumFeature](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumFeature)
* [LGBM_BoosterGetNumClasses](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumClasses)
* [LGBM_BoosterGetNumPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumPredict)
* [LGBM_BoosterGetPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetPredict)
* [LGBM_BoosterLoadModelFromString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterLoadModelFromString)
* [LGBM_BoosterPredictForMat](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMat)
* [LGBM_BoosterPredictForMatSingleRow](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRow)
* [LGBM_BoosterPredictForMatSingleRowFast](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRowFast)
* [LGBM_BoosterPredictForMatSingleRowFastInit](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRowFastInit)
* [LGBM_BoosterSaveModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModel)
* [LGBM_BoosterSaveModelToString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModelToString)
* [LGBM_BoosterUpdateOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIter)
* [LGBM_BoosterUpdateOneIterCustom](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIterCustom)
* [LGBM_DatasetCreateFromFile](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromFile)
* [LGBM_DatasetCreateFromMat](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromMat)
* [LGBM_DatasetFree](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetFree)
* [LGBM_DatasetGetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetFeatureNames)
* [LGBM_DatasetGetNumData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetNumData)
* [LGBM_DatasetGetNumFeature](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetNumFeature)
* [LGBM_GetLastError](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_GetLastError)
* [LGBM_DatasetSetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetFeatureNames)
* [LGBM_DatasetSetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField)
* [LGBM_DatasetDumpText](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetDumpText)

Not yet supported:
* [LGBM_BoosterCalcNumPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterCalcNumPredict)
* [LGBM_BoosterDumpModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterDumpModel)
* [LGBM_BoosterFreePredictSparse](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterFreePredictSparse)
* [LGBM_BoosterGetCurrentIteration](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetCurrentIteration)
* [LGBM_BoosterGetEvalCounts](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetEvalCounts)
* [LGBM_BoosterGetLeafValue](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetLeafValue)
* [LGBM_BoosterGetLowerBoundValue](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetLowerBoundValue)
* [LGBM_BoosterGetUpperBoundValue](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetUpperBoundValue)
* [LGBM_BoosterMerge](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterMerge)
* [LGBM_BoosterNumberOfTotalModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterNumberOfTotalModel)
* [LGBM_BoosterNumModelPerIteration](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterNumModelPerIteration)
* [LGBM_BoosterPredictForCSC](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSC)
* [LGBM_BoosterPredictForCSR](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSR)
* [LGBM_BoosterPredictForCSRSingleRow](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSRSingleRow)
* [LGBM_BoosterPredictForCSRSingleRowFast](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSRSingleRowFast)
* [LGBM_BoosterPredictForCSRSingleRowFastInit](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSRSingleRowFastInit)
* [LGBM_BoosterPredictForFile](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForFile)
* [LGBM_BoosterPredictForMats](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMats)
* [LGBM_BoosterPredictSparseOutput](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictSparseOutput)
* [LGBM_BoosterRefit](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterRefit)
* [LGBM_BoosterResetParameter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterResetParameter)
* [LGBM_BoosterResetTrainingData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterResetTrainingData)
* [LGBM_BoosterRollbackOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterRollbackOneIter)
* [LGBM_BoosterSetLeafValue](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSetLeafValue)
* [LGBM_BoosterShuffleModels](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterShuffleModels)
* [LGBM_DatasetAddFeaturesFrom](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetAddFeaturesFrom)
* [LGBM_DatasetCreateByReference](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateByReference)
* [LGBM_DatasetCreateFromCSC](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromCSC)
* [LGBM_DatasetCreateFromCSR](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromCSR)
* [LGBM_DatasetCreateFromCSRFunc](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromCSRFunc)
* [LGBM_DatasetCreateFromMats](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromMats)
* [LGBM_DatasetCreateFromSampledColumn](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetCreateFromSampledColumn)
* [LGBM_DatasetGetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetField)
* [LGBM_DatasetGetSubset](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetSubset)
* [LGBM_DatasetPushRows](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetPushRows)
* [LGBM_DatasetPushRowsByCSR](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetPushRowsByCSR)
* [LGBM_DatasetSaveBinary](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSaveBinary)
* [LGBM_DatasetUpdateParamChecking](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetUpdateParamChecking)
* [LGBM_FastConfigFree](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_FastConfigFree)


## License

As LightGBM4j repackages bits of SWIG wrapper code from original LightGBM authors, it 
also uses exactly the [same license](https://github.com/microsoft/LightGBM/blob/master/LICENSE).

```
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
