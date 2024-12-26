using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_33
{
    class ResultModel : InputModel
    {
        [ColumnName("PredictedLabel")]
        public uint Prediction { get; set; }
    }
}
