using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_32
{
    internal class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public int PredictedRating { get; set; }

        public float[] Score { get; set; }

    }
}
