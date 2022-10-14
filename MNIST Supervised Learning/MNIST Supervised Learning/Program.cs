using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNet
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        public static void Main()
        {
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            //Application.Run(new Form1());

            //set up the network
            Network.learningRate = 0.1;
            Network.momentumScalar = 0.001;
            Network.batchSize = 4;
            Network nn = new Network(new int[] { 2, 2, 1 });

            //set up training samples
            List<TrainingSample> trainingSamples = new List<TrainingSample>();
            trainingSamples.Add(new TrainingSample(new double[] {0, 0}, new double[] { 0 }));
            trainingSamples.Add(new TrainingSample(new double[] {0, 1}, new double[] { 1 }));
            trainingSamples.Add(new TrainingSample(new double[] {1, 0}, new double[] { 1 }));
            trainingSamples.Add(new TrainingSample(new double[] {1, 1}, new double[] { 0 }));

            //train network
            for (int epoch = 0; epoch < 100000; epoch++)
            {
                double mse = 0;
                trainingSamples.ForEach(sample => mse += nn.backPropagate(sample.inputs, sample.targets));
                nn.updateWeightsAndBiases();

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / Network.batchSize);
            }

            //results
            Console.WriteLine(nn.forwardPropagate(new double[] {0, 0})[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] {0, 1})[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] {1, 0})[0]);
            Console.WriteLine(nn.forwardPropagate(new double[] {1, 1})[0]);
        }
    }
}
