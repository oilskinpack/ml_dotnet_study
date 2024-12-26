using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_3
{
        class InputModel
        {
            [LoadColumn(0)]
            public string Name { get; set; }

            [LoadColumn(1)]
            public string TypeName { get; set; }

            [LoadColumn(2)]
            public int Connections { get; set; }

            [LoadColumn(3)]
            public string Type { get; set; }

            [LoadColumn(4)]
            public string View { get; set; }


        //[LoadColumn(5)]
        //public string Name_Translated => TextHelper.TransliterateText(Name);



    }
}
