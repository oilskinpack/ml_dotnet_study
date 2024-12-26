using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_34
{
        class InputModel
        {
            [LoadColumn(3)]
            public string Class { get; set; }

            [VectorType(3)]
            [LoadColumn(new[] { 0, 1, 2 })]
            public float[] Features { get; set; }


    }
}
