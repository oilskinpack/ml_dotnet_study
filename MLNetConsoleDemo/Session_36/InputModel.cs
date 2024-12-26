using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.Session_36
{
        class InputModel
        {
                public string ImagePath { get; set; }
                public string FruitName { get; set; }

                [ColumnName("Features")]
                public byte[] ImageBytes { get; set; }

        }
}
