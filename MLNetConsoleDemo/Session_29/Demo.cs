using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_29
{
    public class Demo
    {
        public static void Execute()
        {
            // Create new MLContext
            MLContext context = new MLContext();

            // Load data
            var trainingDataView = context.Data.LoadFromTextFile<InputModel>(
                path: "D:\\!Хабаров\\Проекты C#\\MLNetDemo\\MLNetConsoleDemo\\Session_29\\Amazon-reviews-train-dataset.csv", hasHeader: true, separatorChar: ',', allowQuoting: true);

            // Prepare data & create pipeline
            var dataPipeline = context.Transforms.SelectColumns(nameof(InputModel.Summary), nameof(InputModel.ReviewText), nameof(InputModel.Recommend))
                .Append(context.Transforms.Text.FeaturizeText("Featurize_Summary", nameof(InputModel.Summary)))
                .Append(context.Transforms.Text.FeaturizeText("Featurize_ReviewText", nameof(InputModel.ReviewText)))
                .Append(context.Transforms.DropColumns(nameof(InputModel.Summary), nameof(InputModel.ReviewText)))
                .Append(context.Transforms.Concatenate("Features", "Featurize_Summary", "Featurize_ReviewText"))
                .Append(context.Transforms.Conversion.ConvertType("Label", nameof(InputModel.Recommend)));

            var trainer = context.BinaryClassification.Trainers.SgdCalibrated(new SgdCalibratedTrainer.Options
            {
                LabelColumnName = nameof(InputModel.Recommend),
                NumberOfIterations = 100,
                Shuffle = true,
            });

            var trainingPipeline = dataPipeline.Append(trainer);

            // Train Model
            var p = trainingDataView.Preview();
            var model = trainingPipeline.Fit(trainingDataView);

            // Create predator 
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            PrintResult(predictionEngine.Predict(new InputModel
            {
                Summary = "Awesome Magazine",
                ReviewText = "I read this magazine and its pretty amazing.",
            }));

            PrintResult(predictionEngine.Predict(new InputModel
            {
                Summary = "Not A Good Deal, Disappointing content",
                ReviewText = "Not worth buying. Too many ads and tabloid style writing make this publication not worth your while.",
            }));

        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.PredictedRecommendation} | Score: {result.Score}");
        }
    }
}
