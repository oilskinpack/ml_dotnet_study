using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_21
{
    public class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "D:\\!Хабаров\\Проекты C#\\MLNetDemo\\MLNetConsoleDemo\\Session_21\\combined-dataset.csv", hasHeader: true, separatorChar: ',');

            var preview = dataView.Preview();

            //Перемешивание данных
            var suffledData = context.Data.ShuffleRows(dataView);
            preview = suffledData.Preview();

            //Скипнуть первые 8 рядов
            var skipedData = context.Data.SkipRows(dataView, 8);
            preview = skipedData.Preview();

            //Взять первые 8 рядов
            var takeData = context.Data.TakeRows(dataView, 8);
            preview = takeData.Preview();

            //Брать только строки, где годы от 3 до 6
            var filterByValue = context.Data.FilterRowsByColumn(dataView, nameof(InputModel.YearsOfExperience), lowerBound: 3, upperBound: 6);
            preview = filterByValue.Preview();

            //Скипнуть все строки, где Зарплата без данных
            var filterByMissingValue = context.Data.FilterRowsByMissingValues(dataView, nameof(InputModel.Salary));
            preview = filterByMissingValue.Preview();
        }
    }
}
