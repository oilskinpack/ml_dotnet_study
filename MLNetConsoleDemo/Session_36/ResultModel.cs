using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_36
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public string FruitName { get; set; }

        public float[] Score { get; set; }
    }
}
