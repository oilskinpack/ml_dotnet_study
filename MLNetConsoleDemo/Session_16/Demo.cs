using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_16
{
    public class Demo
    {
            public static void Execute()
            {
                MLContext context = new MLContext();

                var columnsToLoad = new TextLoader.Column[]
                {
                        new TextLoader.Column("YearsOfExperience", DataKind.Single, 0),
                        new TextLoader.Column("Salary", DataKind.Single, 1),
                };

            //IDataView dataView = context.Data.LoadFromTextFile(
            //    path: "Session_16/train-dataset.csv", hasHeader: true, separatorChar: ',', columns: columnsToLoad);

            IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_16/train-dataset.csv", hasHeader: true, separatorChar: ',');

            //IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
            //        path: "Session_16/train-dataset.tsv", hasHeader: true);

            var preview = dataView.Preview();
            }
    }
}
