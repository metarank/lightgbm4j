package com.microsoft.ml.lightgbm;

public enum PredictionType {

    C_API_PREDICT_NORMAL(lightgbmlibJNI.C_API_PREDICT_NORMAL_get(), "Normal prediction"),
    C_API_PREDICT_RAW_SCORE(lightgbmlibJNI.C_API_PREDICT_RAW_SCORE_get(), "Raw score"),
    C_API_PREDICT_LEAF_INDEX(lightgbmlibJNI.C_API_PREDICT_LEAF_INDEX_get(), "Leaf index"),
    C_API_PREDICT_CONTRIB(lightgbmlibJNI.C_API_PREDICT_CONTRIB_get(), "Feature contributions (SHAP values)");

    private final int type;
    private final String description;

    PredictionType(int type, String description) {
        this.type = type;
        this.description = description;
    }

    public int getType() {
        return type;
    }

    public String getDescription() {
        return description;
    }

    public boolean equals(PredictionType that) {
        return this.type == that.type;
    }

    @Override
    public String toString() {
        return "PredictionType{" +
                "type=" + type +
                ", description='" + description + '\'' +
                '}';
    }
}