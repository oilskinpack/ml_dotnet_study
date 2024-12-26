using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_33
{
    public class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext(seed:1);

            // Load data
            var dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "D:\\!Хабаров\\Проекты C#\\MLNetDemo\\MLNetConsoleDemo\\Session_33\\train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Load data
            var trainAndTestDataview = context.Data.TrainTestSplit(dataView, testFraction: 0.02);

            // Create Pipeline
            var binaryTrainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

            var pipeline = context.Transforms.Conversion.MapValueToKey(nameof(InputModel.Label))
                .Append(context.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer));

            // Create Model
            var model = pipeline.Fit(trainAndTestDataview.TrainSet);

            // Verify with testset
            var testData = model.Transform(trainAndTestDataview.TestSet);
            var predictions = context.Data.CreateEnumerable<ResultModel>(testData, reuseRowObject: false).ToList();
            foreach (var prediction in predictions)
            {
                System.Console.WriteLine($"Original value: {prediction.Label} | Predicted value: {prediction.Prediction}");
            }




        }
        

    }
}
