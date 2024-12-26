using Microsoft.ML;
using Microsoft.ML.Data;
using MLNetConsoleDemo.Session_25;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_27
{
    public class Demo
    {
        public static void Execute()
        {
            //Создаем контекст
            MLContext context = new MLContext();

            //Подгружаем оригинальные данные
            var dataView = context.Data.LoadFromTextFile<InputModel>
                (path: "D:\\!Хабаров\\Проекты C#\\MLNetDemo\\MLNetConsoleDemo\\Session_27\\Flight-Delay-train-dataset.csv",
                hasHeader: true,
                separatorChar: ',');
            var oldPreview = dataView.Preview();

            //Создаем пайплайн
            var pipeline = context.Transforms.SelectColumns(nameof(InputModel.Origin),
                                                            nameof(InputModel.Destination),
                                                            nameof(InputModel.DepartureTime),
                                                            nameof(InputModel.ExpectedArrivalTime),
                                                            nameof(InputModel.IsDelayBy15Minutes))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_ORIGINAL", nameof(InputModel.Origin)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_DESTINATION", nameof(InputModel.Destination)))
                .Append(context.Transforms.DropColumns(nameof(InputModel.Origin), nameof(InputModel.Destination)))
                .Append(context.Transforms.Concatenate("Features",
                                                       "Encoded_ORIGINAL",
                                                       "Encoded_DESTINATION",
                                                       nameof(InputModel.DepartureTime), 
                                                       nameof(InputModel.ExpectedArrivalTime)))
                .Append(context.Transforms.Conversion.ConvertType("Label", nameof(InputModel.IsDelayBy15Minutes), Microsoft.ML.Data.DataKind.Boolean))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(maximumNumberOfIterations:100));

            var model = pipeline.Fit(dataView);
            var preview = model.Transform(dataView).Preview();


            // Create predator 
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel { Origin = "JFK", Destination = "ATL", DepartureTime = 1930, ExpectedArrivalTime = 2225 };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { Origin = "MSP", Destination = "SEA", DepartureTime = 1745, ExpectedArrivalTime = 1930 };
            PrintResult(predictionEngine.Predict(input));



        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.WillDelayBy15Minutes} | Score: {result.Score} | Probability: {result.Probability}");
        }
    }
}
