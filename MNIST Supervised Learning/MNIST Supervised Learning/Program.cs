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
            Network nn = new Network(new int[] { 784, 32, 10 });

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
                for (int i = 0; i < standardizedPixelValues.Length; i++)
                    standardizedPixelValues[i] = double.Parse(dividedString[i+1]) / 256.0;

                //classify output
                double[] targets = new double[10];
                targets[int.Parse(dividedString[0])] = 1;

                trainingSamples.Add(new TrainingSample(standardizedPixelValues, targets));
            }

            //train network
            for (int epoch = 0; epoch < 5; epoch++)
            {
                double mse = 0;
                trainingSamples.ForEach(sample => mse += nn.backPropagate(sample.inputs, sample.targets));
                nn.updateWeightsAndBiases();

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / Network.batchSize);
            }

            //results
            StreamReader srTest = new StreamReader(File.OpenRead("mnist_test.csv"));
            line = srTest.ReadLine(); //skips first line

            int successes = 0;
            int total = 0;
            while ((line = srTest.ReadLine()) != null)
            {
                String[] dividedString = line.Split(',');

                //standardize inputs
                double[] standardizedPixelValues = new double[784];
                for (int i = 0; i < standardizedPixelValues.Length; i++)
                    standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 256.0;

                //print output
                double[] output = nn.forwardPropagate(standardizedPixelValues);
                if (Array.IndexOf(output, output.Max()) == int.Parse(dividedString[0]))
                    successes++;
                total++;
            }

            Console.WriteLine("Success Rate: " + ((double)successes / (double)total) + "%");
        }
    }
}
