using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using System.Linq;
using Microsoft.ML;

namespace IntentAnalysis
{
    class Program
    {
        const string dataPath = "Data/intents.txt";
        const string testPath = "Data/testData.txt";
        static void Main(string[] args)
        {
            var trainer = new IntentTrainer();
            var model = trainer.Train();
            Evaluate(model);
            var input = Console.ReadLine();

            while (input != "exit")
            {
                var prediction = model.Predict(new IntentData
                {
                    text = "",
                    Label = input
                });
                Console.WriteLine(prediction.PredictedLabel);
                input = Console.ReadLine();
            }
        }
        public static void Evaluate(PredictionModel<IntentData, IntentPrediction> model)
        {
            var testData = new TextLoader<IntentData>(testPath, useHeader: false, separator: "tab");
            var evaluator = new ClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");            
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro * 100}%");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro * 100}%");
        }
    }
}