using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace IntentAnalysis
{
    public class IntentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}