using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_29
{
        class InputModel
        {
            [LoadColumn(2)]
            public string Summary { get; set; }

            [LoadColumn(3)]
            public string ReviewText { get; set; }

            [LoadColumn(4)]
            public bool Recommend { get; set; }
        }
}
