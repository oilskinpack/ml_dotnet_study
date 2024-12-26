using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_34
{
    class ResultModel : InputModel
    {
        public uint PredictedLabel { get; set; }

        public string PredictedClass { get; set; }
    }
}
