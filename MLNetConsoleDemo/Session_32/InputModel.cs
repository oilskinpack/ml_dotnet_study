using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_32
{
        class InputModel
        {
            [LoadColumn(1)]
            public string Summary { get; set; }

            [LoadColumn(2)]
            public string ReviewText { get; set; }

            [LoadColumn(3)]
            public int Ratings { get; set; }


        }
}
