using OfficeOpenXml;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_0
{
    static public class ExcelTranslateHelper
    {
        static public void TransliterateExcel(string path,string nameOfSheet)
        {
            FileInfo fileInfo = new FileInfo(path);

            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
            using (ExcelPackage package = new ExcelPackage(fileInfo))
            {
                ExcelWorksheet worksheet = package.Workbook.Worksheets.Where(it => it.Name == nameOfSheet).First();

                for (int column = 1; column < 7; column++)
                {
                    for (int row = 1; row < 1000; row++)
                    {
                        var oldValue = worksheet.Cells[row, column].Value;
                        if (oldValue == null) continue;

                        worksheet.Cells[row, column].Value = TextHelper.TransliterateText(oldValue.ToString());
                    }
                }

                package.SaveAs(path);
            }
        }
    }
}
