﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_4
{
    class ResultModel:InputModel
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        public float[] Score { get; set; }
    }
}
