using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Supervised_Learning
{
    public struct TrainingSample
    {
        public double[] inputs, targets;
        public TrainingSample(double[] inp, double[] tar)
        {
            inputs = inp;
            targets = tar;
        }
    }
}
