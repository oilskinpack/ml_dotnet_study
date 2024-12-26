using Microsoft.AspNetCore.Http;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_4
{
    class Demo
    {
        /// <summary>
        /// Стоп-слова
        /// </summary>
        static public string[] stopWords = new string[]
            {
                TextHelper.TransliterateText("прямоугольная"),
                TextHelper.TransliterateText("прямоугольный"),
                TextHelper.TransliterateText("прямоугольного"),
                TextHelper.TransliterateText("прямоугольное"),

                TextHelper.TransliterateText("круглая"),
                TextHelper.TransliterateText("круглый"),
                TextHelper.TransliterateText("круглого"),

                TextHelper.TransliterateText("стандартное"),
                TextHelper.TransliterateText("стандартная"),
                TextHelper.TransliterateText("стандартный"),

                TextHelper.TransliterateText("воздуховода"),
                TextHelper.TransliterateText("воздуховодов"),

                TextHelper.TransliterateText("сечение"),
                TextHelper.TransliterateText("сечения"),

                TextHelper.TransliterateText("с"),
                TextHelper.TransliterateText("на"),
                TextHelper.TransliterateText("bru"),
                TextHelper.TransliterateText("adsk"),

                TextHelper.TransliterateText("стандарт"),
                TextHelper.TransliterateText("стандартная"),
                TextHelper.TransliterateText("стандартное"),
                TextHelper.TransliterateText("стандартный"),
            };

        /// <summary>
        /// Список разделителей
        /// </summary>
        static public char[] sepList = new char[]
            {
                ':',
                '_',
                '-',
                ' '
            };

        /// <summary>
        /// Контекст
        /// </summary>
        static public MLContext Context = new MLContext();


        /// <summary>
        /// Загружаемые данные
        /// </summary>
        static public IDataView Dataview;

        /// <summary>
        /// Пайплайн
        /// </summary>
        static public EstimatorChain<ColumnConcatenatingTransformer> DataPipeline;

        /// <summary>
        /// Алгоритм
        /// </summary>
        static public SdcaMaximumEntropyMulticlassTrainer Trainer;

        /// <summary>
        /// Пайплайн с алгоритмом
        /// </summary>
        static public EstimatorChain<KeyToValueMappingTransformer> TrainPipeview;


        static public TransformerChain<KeyToValueMappingTransformer> Transformer;

        /// <summary>
        /// Предиктор
        /// </summary>
        static public PredictionEngine<InputModel, ResultModel> PredictionEngine;
        static public void Execute()
        {

            GetDataView();

            CreatePipeLine();

            AddTrained();

            CreateAndSaveModel();

            SaveModel();

            PrintResult(PredictionEngine.Predict(new InputModel
            {
                Name = TextHelper.TransliterateText("Заглушка воздуховода из оцинкованной стали диаметром 150"),
                Type = TextHelper.TransliterateText("ЗаглушкаВоздуховодаПрямоугольная"),
                Connections = 1
            }));

        }

        private static void SaveModel()
        {
            Context.Model.Save(Transformer, Dataview.Schema, "D:\\!Хабаров\\Проекты C#\\ВИС.Машинное обучение\\СдвВидМодель\\DuctFittingTypeModel.zip");
        }

        private static void CreateAndSaveModel()
        {
            //модель
            Transformer = TrainPipeview.Fit(Dataview);
            PredictionEngine = Context.Model.CreatePredictionEngine<InputModel, ResultModel>(Transformer);
        }

        private static void AddTrained()
        {
            //Задаем алгоритм
            Trainer = Context.MulticlassClassification.Trainers.SdcaMaximumEntropy(maximumNumberOfIterations: 100);
            TrainPipeview = DataPipeline
                .Append(Trainer)
                .Append(Context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        private static void CreatePipeLine()
        {
            //Создание пайплайн
            DataPipeline = Context.Transforms
                //Загружаем колонки с модели
                .SelectColumns(nameof(InputModel.Name),
                                nameof(InputModel.TypeName),
                                nameof(InputModel.Connections),
                                nameof(InputModel.Type))
                //Нормализуем наименование
                .Append(Context.Transforms.Text.NormalizeText(
                                                                inputColumnName: nameof(InputModel.Name),
                                                                outputColumnName: "Name_Normalized",
                                                                caseMode: Microsoft.ML.Transforms.Text.TextNormalizingEstimator.CaseMode.Lower,
                                                                keepDiacritics: false,
                                                                keepPunctuations: false,
                                                                keepNumbers: true))
                //Нормализация наименования типа
                .Append(Context.Transforms.Text.NormalizeText(
                                                                inputColumnName: nameof(InputModel.TypeName),
                                                                outputColumnName: "TypeName_Normalized",
                                                                caseMode: Microsoft.ML.Transforms.Text.TextNormalizingEstimator.CaseMode.Lower,
                                                                keepDiacritics: false,
                                                                keepPunctuations: true,
                                                                keepNumbers: true))
                //Токенизация имени
                .Append(Context.Transforms.Text.TokenizeIntoWords(
                                                                inputColumnName: "Name_Normalized"
                                                                , outputColumnName: "Name_Tokens",
                                                                separators: sepList))
                .Append(Context.Transforms.Text.RemoveStopWords(
                                                                outputColumnName: "Name_Cleared",
                                                                inputColumnName: "Name_Tokens",
                                                                stopWords))
                //Токенизация имени типа
                .Append(Context.Transforms.Text.TokenizeIntoWords(
                                                                inputColumnName: "TypeName_Normalized"
                                                                , outputColumnName: "TypeName_Tokens",
                                                                separators: sepList))
                .Append(Context.Transforms.Text.RemoveStopWords(
                                                                outputColumnName: "TypeName_Cleared",
                                                                inputColumnName: "TypeName_Tokens",
                                                                stopWords))
                //Фичуризация наименования
                .Append(Context.Transforms.Text.FeaturizeText(inputColumnName: "Name_Cleared",
                                                              outputColumnName: "Name_Featurized"))
                //Фичуризация наименования типа
                .Append(Context.Transforms.Text.FeaturizeText(inputColumnName: "TypeName_Cleared",
                                                              outputColumnName: "TypeName_Featurized"))
                //Преобразование числа коннекторов в int
                .Append(Context.Transforms.Conversion.ConvertType(
                                                            inputColumnName: nameof(InputModel.Connections),
                                                            outputColumnName: "Connectors_INT",
                                                            outputKind: Microsoft.ML.Data.DataKind.Single))
                //Перевод в лейбл
                .Append(Context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: nameof(InputModel.Type),
                    outputColumnName: "Label"))
                //Конкатенация фич
                .Append(Context.Transforms.Concatenate(outputColumnName: "Features", "Name_Featurized", "TypeName_Featurized", "Connectors_INT"));

            //Проверка
            //var p = dataPipeline.Preview(dataview);
        }

        private static void GetDataView()
        {
            //Загрузка оригинальных данных
            Dataview = Context.Data.LoadFromTextFile<InputModel>
                (path: "D:\\!Хабаров\\Проекты C#\\ВИС.Машинное обучение\\translatedDataSet.csv",
                hasHeader: true,
                separatorChar: ';',
                allowQuoting: true);

            //Перемешивание данных
            Dataview = Context.Data.ShuffleRows(Dataview);
        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.PredictedLabel} | Score: {result.Score}");
        }

    }
}
