﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_17
{
    public class Demo
    {
            public static void Execute()
            {
                MLContext context = new MLContext();

                //IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
                //    path: "Session_17/training-dataset/*", hasHeader: false, separatorChar: ',');

                var textLoader = context.Data.CreateTextLoader<InputModel>(separatorChar: ',');
                IDataView dataView = textLoader.Load(
                    "Session_17/training-dataset/01.csv",
                    "Session_17/training-dataset/02.csv");

                var preview = dataView.Preview();
            }
    }
}
