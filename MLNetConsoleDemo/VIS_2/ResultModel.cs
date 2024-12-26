using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_2
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel { get; set; }

        public float[] Score { get; set; }
    }
}
