package com.example.covid_19_detector;

public class FlowersConfig extends ModelConfig {

    @Override
    public String getModelFilename() {
        return "quantized_model.tflite";
    }

    @Override
    public String getLabelsFilename() {
        return "labels.txt";
    }

    @Override
    public int getInputWidth() {
        return 500;
    }

    @Override
    public int getInputHeight() {
        return 500;
    }

    @Override
    public int getInputSize() {
        return getInputWidth() * getInputHeight() * getChannelsCount() * FLOAT_BYTES_COUNT;
    }

    @Override
    public int getChannelsCount() {
        return 1;
    }

    @Override
    public float getStd() {
        return 128.0f;
    }

    @Override
    public float getMean() {
        return 0.0f;
    }
}
