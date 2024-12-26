using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_13
{
    class OutputModel
    {
        [ColumnName("Score")]
        public float Salary { get; set; }
    }
}
