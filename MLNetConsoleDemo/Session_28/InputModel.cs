using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_28
{
        class InputModel
        {
            [LoadColumn(0)]
            public bool Sentiment { get; set; }

            [LoadColumn(1)]
            public string SentimentText { get; set; }
        }
}
