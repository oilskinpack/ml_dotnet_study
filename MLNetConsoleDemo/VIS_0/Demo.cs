using Microsoft.Identity.Client;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using OfficeOpenXml;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.TrainCatalogBase;

namespace MLNetConsoleDemo.VIS_0
{
    /// <summary>
    /// Модель собрана по принципу кросс-валидации [Плохо]
    /// </summary>
    public class Demo
    {
        public static void Execute()
        {
            //Создание контекста
            MLContext context = new MLContext(seed:1);

            //Загрузка данных
            var dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "D:\\!Хабаров\\Проекты C#\\ВИС.Машинное обучение\\translatedDataSet.csv", hasHeader: true, separatorChar: ';', allowQuoting: true);

            //Фильтрация и смешивание данных
            dataView = context.Data.ShuffleRows( dataView );

            var splitData = context.Data.TrainTestSplit(dataView, testFraction: 0.9);





            //Создание пайплайн
            var dataPipeLine = context.Transforms.SelectColumns(nameof(InputModel.Name)
                                                                , nameof(InputModel.TypeName)
                                                                , nameof(InputModel.Connections)
                                                                , nameof(InputModel.Type))
                .Append(context.Transforms.Text.FeaturizeText(outputColumnName: "Name_FEATURIZED", nameof(InputModel.Name)))
                .Append(context.Transforms.Text.FeaturizeText(outputColumnName: "TypeName_FEATURIZED", nameof(InputModel.TypeName))
                .Append(context.Transforms.Conversion.ConvertType("Connectors_INT", nameof(InputModel.Connections), DataKind.Single))
                .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(InputModel.Type))))
                .Append(context.Transforms.Concatenate("Features", "Name_FEATURIZED", "TypeName_FEATURIZED", "Connectors_INT"));
            var preview = dataPipeLine.Preview(dataView);

            //Добавление алгоритма
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(maximumNumberOfIterations: 50);
            var trainPipeline = dataPipeLine.Append(trainer)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //Кросс валидация
            var crossValidationResult = context.MulticlassClassification.CrossValidate(splitData.TrainSet, trainPipeline, numberOfFolds: 5, labelColumnName: "Label");








            var allMetrics = crossValidationResult.Select(x => x.Metrics);
            var allModels = crossValidationResult.Select(x => x.Model);

            foreach (var metrics in allMetrics)
            {
                Console.WriteLine($"Micro-Accuracy (1) : {metrics.MicroAccuracy} | " +
                    $"Macro-Accuracy (1) {metrics.MacroAccuracy} | " +
                    $"Log-loss (0) {metrics.LogLoss} | " + 
                    $"Log-Loss Reducion (1) {metrics.LogLossReduction} | ");
            }
            Console.WriteLine("\n");



            var model = allModels.ElementAt(2);

            //Модель предсказания
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(allModels.ElementAt(3));

            var modelMetric = context.MulticlassClassification.Evaluate(model.Transform(splitData.TrainSet), labelColumnName: "Label");

            Console.WriteLine($"Micro-Accuracy (1) : {modelMetric.MicroAccuracy} | " +
                                $"Macro-Accuracy (1) {modelMetric.MacroAccuracy} | " +
                                $"Log-loss (0) {modelMetric.LogLoss} | " +
                                $"Log-Loss Reducion (1) {modelMetric.LogLossReduction} | ");
            Console.WriteLine("\n");


            PrintResult(predictionEngine.Predict(new InputModel
            {
                Name = TextHelper.TransliterateText("Врезка воздуховода прямоугольная 800x600-800x600. Толщина стали 0,8"),
                Type = TextHelper.TransliterateText("ADSK_ВрезкаВоздуховода_Прямоугольная: Стандарт"),
                Connections = 2
            }));

        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.PredictedLabel} | Score: {result.Score}");
        }
    }
}
