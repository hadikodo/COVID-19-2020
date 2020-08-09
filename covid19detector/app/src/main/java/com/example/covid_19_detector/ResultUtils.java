package com.example.covid_19_detector;

import com.example.covid_19_detector.ClassificationResult;

import java.util.List;

public class ResultUtils {
    public static String resultsToStr(List<ClassificationResult> classificationResults) {
        StringBuilder results = new StringBuilder("Classification:\n");
        if (classificationResults.size() == 0) {
            results.append("No results");
        } else {
            for (ClassificationResult classificationResult : classificationResults) {
                results.append(classificationResult.title);
                        /*.append("(")
                        .append(classificationResult.confidence * 100)
                        .append("%)\n");*/
            }
        }

        return results.toString();
    }
}
