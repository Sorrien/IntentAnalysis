using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace IntentAnalysis
{
    public class IntentTrainer
    {
        const string dataPath = "Data/intents.txt";

        public PredictionModel<IntentData, IntentPrediction> Train()
        {
            var pipeline = GetPipeline();
            var model = pipeline.Train<IntentData, IntentPrediction>();
            return model;
        }

        public LearningPipeline GetPipeline()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<IntentData>(dataPath, separator: "tab"));
            pipeline.Add(new TextFeaturizer(outputColumn: "Features", inputColumns: "text"));
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            return pipeline;
        }
    }
}
