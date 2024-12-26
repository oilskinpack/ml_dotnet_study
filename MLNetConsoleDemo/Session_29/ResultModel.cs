using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_29
{
    internal class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedRecommendation { get; set; }

        public float Score { get; set; }

    }
}
