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
using MLNetConsoleDemo.VIS_0;

namespace MLNetConsoleDemo.VIS_1
{
    /// <summary>
    /// Модель собранная по принципу 100% в дататест [Отлично]
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

            var splitData = context.Data.TrainTestSplit(dataView);




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


            //Модель предсказания
            var transformer = trainPipeline.Fit(splitData.TrainSet);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(transformer);

            var modelMetric = context.MulticlassClassification.Evaluate(transformer.Transform(splitData.TrainSet), labelColumnName: "Label");

            Console.WriteLine($"Micro-Accuracy (1) : {modelMetric.MicroAccuracy} | " +
                                $"Macro-Accuracy (1) {modelMetric.MacroAccuracy} | " +
                                $"Log-loss (0) {modelMetric.LogLoss} | " +
                                $"Log-Loss Reducion (1) {modelMetric.LogLossReduction} | ");
            Console.WriteLine("\n");


            PrintResult(predictionEngine.Predict(new InputModel
            {
                Name = TextHelper.TransliterateText("Переход воздуховода из стали круглой диаметром 160мм толщиной 0,8мм"),
                Type = TextHelper.TransliterateText("СоедДетальСтандарт"),
                Connections = 2
            }));

        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.PredictedLabel} | Score: {result.Score}");
        }
    }
}
