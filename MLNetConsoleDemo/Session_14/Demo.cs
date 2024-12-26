using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_14
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            List<InputModel> data = new List<InputModel>
            {
                new InputModel { YearsOfExperience = 1, Salary= 39000 },
                new InputModel { YearsOfExperience = 1.3F, Salary= 46200 },
                new InputModel { YearsOfExperience = 1.5F, Salary= 37700 },
                new InputModel { YearsOfExperience = 2, Salary= 43500 },
                new InputModel { YearsOfExperience = 2.2F, Salary= 40000 },
                new InputModel { YearsOfExperience = 2.9F, Salary= 56000 },
                new InputModel { YearsOfExperience = 3, Salary= 60000 },
                new InputModel { YearsOfExperience = 3.2F, Salary= 54000 },
                new InputModel { YearsOfExperience = 3.3F, Salary= 64000 },
                new InputModel { YearsOfExperience = 3.7F, Salary= 57000 },
                new InputModel { YearsOfExperience = 3.9F, Salary= 63000 },
                new InputModel { YearsOfExperience = 4, Salary= 55000 },
                new InputModel { YearsOfExperience = 4, Salary= 58000 },
                new InputModel { YearsOfExperience = 4.1F, Salary= 57000 },
                new InputModel { YearsOfExperience = 4.5F, Salary= 61000 },
                new InputModel { YearsOfExperience = 4.9F, Salary= 68000 },
                new InputModel { YearsOfExperience = 5.3F, Salary= 83000 },
                new InputModel { YearsOfExperience = 5.9F, Salary= 82000 },
                new InputModel { YearsOfExperience = 6, Salary= 94000 },
                new InputModel { YearsOfExperience = 6.8F, Salary= 91000 },
                new InputModel { YearsOfExperience = 7.1F, Salary= 98000 },
                new InputModel { YearsOfExperience = 7.9F, Salary= 101000 },
                new InputModel { YearsOfExperience = 8.2F, Salary= 114000 },
                new InputModel { YearsOfExperience = 8.9F, Salary= 109000 },
            };

            IDataView trainingData = context.Data.LoadFromEnumerable(data);

            var estimator = context.Transforms.Concatenate("Features", new[] { "YearsOfExperience" });

            var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 10,l1Regularization:2, l2Regularization:2));

            var model = pipeline.Fit(trainingData);

            List<InputModel> testData = new List<InputModel>
            {
                new InputModel { YearsOfExperience = 1.3F, Salary= 46200 },
                new InputModel { YearsOfExperience = 2.9F, Salary= 56000 },
                new InputModel { YearsOfExperience = 3.2F, Salary= 54000 },
                new InputModel { YearsOfExperience = 3.9F, Salary= 63000 },
                new InputModel { YearsOfExperience = 4.1F, Salary= 57000 },
                new InputModel { YearsOfExperience = 7.1F, Salary= 98000 },
            };

            var testDataView = context.Data.LoadFromEnumerable(testData);

            var metrics = context.Regression.Evaluate(model.Transform(testDataView), labelColumnName: "Salary");

            Console.WriteLine($"R^2: {metrics.RSquared:0.00}");


        }
    }
}
