using System;
using System.Collections.Generic;
using System.IO;
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
            Network.batchSize = 60000;
            Network nn = new Network(new int[] { 784, 100, 100, 10 });

            //set up training samples
            //assuming a (row x column) image
            List<TrainingSample> trainingSamples = new List<TrainingSample>();
            StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            String line = sr.ReadLine(); //skips first line
            while ((line = sr.ReadLine()) != null)
            {
                String[] dividedString = line.Split(',');

                //standardize inputs
                double[] standardizedPixelValues = new double[784];
                for (int i = 1; i < dividedString.Length; i++)
                    standardizedPixelValues[i] = double.Parse(dividedString[i]) / 256.0;

                //classify output
                double[] target = new double[10];
                target[int.Parse(dividedString[0])] = 1;

                trainingSamples.Add(new TrainingSample(standardizedPixelValues, new double[] {  }));
            }

            //train network
            for (int epoch = 0; epoch < 1000; epoch++)
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
