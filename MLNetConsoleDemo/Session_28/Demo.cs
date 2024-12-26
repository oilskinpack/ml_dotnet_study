using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_28
{
    public class Demo
    {
        public static void Execute()
        {
            //Создаем контекст
            MLContext context = new MLContext();

            //Подгружаем оригинальные данные
            var dataView = context.Data.LoadFromTextFile<InputModel>
                (path: "D:\\!Хабаров\\Проекты C#\\MLNetDemo\\MLNetConsoleDemo\\Session_28\\IMDB-reviews-train-dataset.tsv",
                hasHeader: true);
            var oldPreview = dataView.Preview();

            // Prepare data & create pipeline
            var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(InputModel.SentimentText))
                .Append(context.BinaryClassification.Trainers.AveragedPerceptron(
                    labelColumnName: nameof(InputModel.Sentiment), numberOfIterations: 100));

            // Train Model
            var model = pipeline.Fit(dataView);

            var previe = model.Transform(dataView).Preview();

            // Create predator 
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel { SentimentText = "I liked this movie." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "Movie was just ok." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "It's a really good film. Outragously entertaining, loved it." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "It's worst movie I have ever seen. Boring, badly written." };
            PrintResult(predictionEngine.Predict(input));





        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.IsPositiveReview} | Score: {result.Score}");
        }
    }
}
