using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo.VIS_3
{
    static class TextHelper
    {
        static public Dictionary<char, string> TranslitDic = new Dictionary<char, string>()
        {
            ['а'] = "a",
            ['б'] = "b",
            ['в'] = "v",
            ['г'] = "g",
            ['д'] = "d",
            ['е'] = "e",
            ['ё'] = "jo",
            ['ж'] = "zh",
            ['з'] = "z",
            ['и'] = "i",
            ['й'] = "jj",
            ['к'] = "k",
            ['л'] = "l",
            ['м'] = "m",
            ['н'] = "n",
            ['о'] = "o",
            ['п'] = "p",
            ['р'] = "r",
            ['с'] = "s",
            ['т'] = "t",
            ['у'] = "u",
            ['ф'] = "f",
            ['х'] = "kh",
            ['ц'] = "c",
            ['ч'] = "ch",
            ['ш'] = "sh",
            ['щ'] = "shh",
            ['ъ'] = "'",
            ['ы'] = "y",
            ['ь'] = "",
            ['э'] = "eh",
            ['ю'] = "yu",
            ['я'] = "ya",

            ['А'] = "A",
            ['Б'] = "B",
            ['В'] = "V",
            ['Г'] = "G",
            ['Д'] = "D",
            ['Е'] = "E",
            ['Ё'] = "Jo",
            ['Ж'] = "Zh",
            ['З'] = "Z",
            ['И'] = "I",
            ['Й'] = "Jj",
            ['К'] = "K",
            ['Л'] = "L",
            ['М'] = "M",
            ['Н'] = "N",
            ['О'] = "O",
            ['П'] = "P",
            ['Р'] = "R",
            ['С'] = "S",
            ['Т'] = "T",
            ['У'] = "U",
            ['Ф'] = "F",
            ['Х'] = "Kh",
            ['Ц'] = "C",
            ['Ч'] = "Ch",
            ['Ш'] = "Sh",
            ['Щ'] = "Shh",
            ['Ъ'] = "'",
            ['Ы'] = "Y",
            ['Ь'] = "",
            ['Э'] = "Eh",
            ['Ю'] = "Yu",
            ['Я'] = "Ya",

            ['°'] = "degrees"
        };

        public static string TransliterateText(string text)
        {
            StringBuilder resultText = new StringBuilder();
            List<char> charList =  text.ToCharArray().ToList();

            foreach (char oneChar in charList)
            {
                string newStr;
                var keyAndValue = TranslitDic.Where(it => it.Key == oneChar).FirstOrDefault();
                if (keyAndValue.Value == null) newStr = oneChar.ToString();
                else
                {
                    newStr = keyAndValue.Value;
                }
                resultText.Append(newStr);
            }

            return resultText.ToString();
        }
    }
}
