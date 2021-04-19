# LightGBM4j: a java wrapper for LightGBM

[![CI Status](https://github.com/metarank/lightgbm4j/workflows/Java%20CI%20with%20Maven/badge.svg)](https://github.com/metarank/lightgbm4j/actions)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j/badge.svg?style=plastic)](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LightGBM4j** is a zero-dependency Java wrapper for the LightGBM project. Its main goal is to provide an 1-1 mapping for all
LightGBM API methods in Java-friendly flavour. 

## Purpose

LightGBM itself has a SWIG-generated JNI interface, which is possible to use directly from Java. The problem with SWIG wrappers
is that they are extremely low-level. For example, to pass a java array thru SWIG, you need to do someting horrible:
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

The library is in an early development stage and not covering all 100% of LightGBM API, but eventual future goal will be 
merging with the upstream LightGBM and to become an official Java bindings for the project.

## Installation

To install, use the following maven coordinates:
```xml
<dependency>
  <groupId>io.github.metarank</groupId>
  <artifactId>lightgbm4j</artifactId>
  <version>3.2.1-1</version>
</dependency>
```

Versioning schema attempts to match the upstream, but with extra `-N` suffix, if there were a couple of extra lightgbm4j-specific
changes released on top.

## Usage

There are two main classes available: 
1. `LGBMDataset` to manage input training and validation data.
2. `LGBMBooster` to do training and inference.

All the public API methods in these classes should map to the [LightGBM C API](https://lightgbm.readthedocs.io/en/latest/C-API.html) methods directly.

Note that both `LGBMBooster` and `LGBMDataset` classes contain handles of native memory
data structures from the LightGBM, so you need to explicitly call `.close()` when they are not used. Otherwise you may catch
a native code memory leak.

To load existing model and run it:
```java
LGBMBooster loaded = LGBMBooster.loadModelFromString(model);
float[] input = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
double[] pred = booster.predictForMat(input, 2, 2, true);
```

To load dataset from java matrix:
```java
float[] matrix = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
LGBMDataset ds = LGBMDataset.createFromMat(matrix, 2, 2, true, "");
```

There are some rough parts in the LightGBM API in loading the dataset from matrices:
* `createFromMat` parameters cannot set the [label](https://lightgbm.readthedocs.io/en/latest/Parameters.html#label_column) or [weight](https://lightgbm.readthedocs.io/en/latest/Parameters.html#weight_column) column. 
So if you do `parameters = "label=some_column_name"`, it will be ignored by the LightGBM.
* label/weight/group columns are magical and should NOT be included in the input matrix for
`createFromMat`
* to set these magical columns, you need to explicitly call `LGBMDataset.setField()` method.
* `label` and `weight` columns [must be](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField) `float[]`
* `group` column [must be](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField) `int[]`

A full example of loading dataset from matrix for a cancer dataset:
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
        LGBMDataset dataset = LGBMDataset.createFromMat(values, 16, columns.length, true, "");
        dataset.setFeatureNames(columns);
        dataset.setField("label", labels);
        return dataset;
```

Also see [a working example](https://github.com/metarank/lightgbm4j/blob/main/src/test/java/io/github/metarank/lightgbm4j/CancerIntegrationTest.java) 
of different ways to deal with input datasets in the LightGBM4j tests.
## Example

```java
// cancer dataset from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
// with labels altered to fit the [0,1] range
LGBMDataset dataset = LGBMDataset.createFromFile("cancer.csv", "header=true label=name:Classification");
LGBMBooster booster = LGBMBooster.create(dataset, "objective=binary label=name:Classification");
for (int i=0; i<10; i++) {
     booster.updateOneIter();
     double[] eval = booster.getEval(0);
     System.out.println(eval[0]);
}
booster.close();
dataset.close();
```

## Supported platforms

This code is tested to work well with Linux (Ubuntu 20.04), Windows (Server 2019) and MacOS 10.15. Supported Java versions are 8 and 11 (probably it will also work with anything >11).

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
* [LGBM_BoosterLoadModelFromString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterLoadModelFromString)
* [LGBM_BoosterPredictForMat](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMat)
* [LGBM_BoosterPredictForMatSingleRow](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRow)
* [LGBM_BoosterSaveModel](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModel)
* [LGBM_BoosterSaveModelToString](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSaveModelToString)
* [LGBM_BoosterUpdateOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIter)
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
* [LGBM_BoosterGetNumClasses](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumClasses)
* [LGBM_BoosterGetNumPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetNumPredict) 
* [LGBM_BoosterGetPredict](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterGetPredict)
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
* [LGBM_BoosterPredictForMatSingleRowFast](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRowFast)
* [LGBM_BoosterPredictForMatSingleRowFastInit](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMatSingleRowFastInit)
* [LGBM_BoosterPredictSparseOutput](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictSparseOutput)
* [LGBM_BoosterRefit](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterRefit)
* [LGBM_BoosterResetParameter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterResetParameter)
* [LGBM_BoosterResetTrainingData](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterResetTrainingData)
* [LGBM_BoosterRollbackOneIter](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterRollbackOneIter)
* [LGBM_BoosterSetLeafValue](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterSetLeafValue)
* [LGBM_BoosterShuffleModels](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterShuffleModels)
* [LGBM_BoosterUpdateOneIterCustom](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterUpdateOneIterCustom)
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