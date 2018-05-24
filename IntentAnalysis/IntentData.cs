using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace IntentAnalysis
{
    public class IntentData
    {
        [Column("0")]
        public string text;
        [Column("1")]
        [ColumnName("Label")]
        public string Label;
    }
}
