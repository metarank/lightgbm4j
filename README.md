# LightGBM4j: a java wrapper for LightGBM

[![CI Status](https://github.com/metarank/lightgbm4j/workflows/Java%20CI%20with%20Maven/badge.svg)](https://github.com/metarank/lightgbm4j/actions)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j/badge.svg?style=plastic)](https://maven-badges.herokuapp.com/maven-central/io.github.metarank/lightgbm4j)

LightGBM is a zero-dependency Java wrapper for the LightGBM project.

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

## Installation

To install, use the following maven coordinates:
```xml
<dependency>
  <groupId>io.github.metarank</groupId>
  <artifactId>lightgbm4j</artifactId>
  <version>3.1.0-2</version>
</dependency>
```

Versioning schema attempts to match the upstream, but with extra `-N` suffix, if there were a couple of extra lightgbm4j-specific
changes released.

## Usage

There are two main classes available: 
1. `LGBMDataset` to manage input training and validation data.
2. `LGBMBooster` to do training and inference.

All the public API methods in these classes should map to the [LightGBM C API](https://lightgbm.readthedocs.io/en/latest/C-API.html) methods directly.

Note that both `LGBMBooster` and `LGBMDataset` classes contain handles of native memory
data structures from the LightGBM, so you need to explicitly call `.close()` when they are not used. Otherwise you may catch
a native code memory leak.


To load dataset from java matrix:
```java
float[] matrix = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
LGBMDataset ds = LGBMDataset.createFromMat(matrix, 2, 2, true, "");
```

To load existing model and run it:
```java
LGBMBooster loaded = LGBMBooster.loadModelFromString(model);
float[] input = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
double[] pred = booster.predictForMat(input, 2, 2, true);
```

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
* [LGBM_DatasetDumpText](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetDumpText)
* [LGBM_DatasetGetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetField)
* [LGBM_DatasetGetSubset](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetGetSubset)
* [LGBM_DatasetPushRows](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetPushRows)
* [LGBM_DatasetPushRowsByCSR](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetPushRowsByCSR)
* [LGBM_DatasetSaveBinary](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSaveBinary)
* [LGBM_DatasetSetFeatureNames](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetFeatureNames)
* [LGBM_DatasetSetField](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetSetField)
* [LGBM_DatasetUpdateParamChecking](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_DatasetUpdateParamChecking)
* [LGBM_FastConfigFree](https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_FastConfigFree)


## License

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/