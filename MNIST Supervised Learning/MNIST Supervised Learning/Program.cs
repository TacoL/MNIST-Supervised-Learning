using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;

namespace MNIST_Supervised_Learning
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
            Network.learningRate = 0.001;
            Network.momentumScalar = 0.0001;
            Network.batchSize = 200;
            Network mainNN = new Network(new int[] { 784, 200, 10 });
            int numEpochs = 20;

            //set up training samples
            //assuming a (row x column) image

            List<TrainingSample> trainingSamples = new List<TrainingSample>();
            StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            String line = sr.ReadLine(); //skips first line
            int setupIdx = 0;
            while ((line = sr.ReadLine()) != null)
            {
                if (line != null)
                    ThreadPool.QueueUserWorkItem(new WaitCallback(o => createSample(line, trainingSamples)));
                //createSample(nn, line, trainingSamples);
                Console.WriteLine($"Sample: {setupIdx}");
                setupIdx++;
            }

            //train network
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                double mse = 0;
                int numBatches = trainingSamples.Count / Network.batchSize;

                //batching and thread pooling
                for (int batchIdx = 0; batchIdx < numBatches - 1; batchIdx++)
                {
                    List<double> sampleMses = new List<double>();
                    List<Network> sampleNNs = new List<Network>();
                    for (int sampleIdx = 0; sampleIdx < Network.batchSize - 1; sampleIdx++) //remove the -1. temp fix
                    {
                        //trainSample(batchIdx, sampleIdx, nn, trainingSamples, batchMse);
                        Network sampleNN = new Network(mainNN);
                        sampleNNs.Add(sampleNN);
                        ThreadPool.QueueUserWorkItem(new WaitCallback(o => trainSample(batchIdx, sampleIdx, sampleNN, trainingSamples, sampleMses)));
                    }

                    sampleNNs.ForEach(sampleNN =>
                    {
                        for (int layerIdx = 0; layerIdx < mainNN.layers.Count; layerIdx++)
                        {
                            for (int neuronIdx = 0; neuronIdx < mainNN.layers[layerIdx].Count; neuronIdx++)
                            {
                                Neuron mainNeuron = mainNN.layers[layerIdx][neuronIdx];
                                Neuron sampleNeuron = sampleNN.layers[layerIdx][neuronIdx];
                                for (int i = 0; i < mainNeuron.weightGradients.Length; i++)
                                    mainNeuron.weightGradients[i] += sampleNeuron.weightGradients[i];
                                mainNeuron.biasGradient += sampleNeuron.biasGradient;
                            }
                        }
                    });

                    double batchMse = 0;
                    //Thread.Sleep(1000);
                    sampleMses.ForEach(sampleMse => batchMse += sampleMse);
                    mse += batchMse / Network.batchSize;
                    mainNN.updateWeightsAndBiases();
                    Console.WriteLine($"Epoch {epoch + 1} / {numEpochs}      Batch #{batchIdx + 1} / {numBatches}      BMSE = {batchMse / Network.batchSize}");
                }

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / numBatches);
            }

            #region RESULTS
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
                double[] output = mainNN.forwardPropagate(standardizedPixelValues);
                int label = int.Parse(dividedString[0]);
                int val = output.ToList().IndexOf(output.Max());
                if (val == label)
                    successes++;
                total++;
            }

            Console.WriteLine($"{successes}, {total}");
            Console.WriteLine("Success Rate: " + ((double)successes / (double)total * 100d) + "%");

            #endregion
        }

        public static void createSample(String line, List<TrainingSample> trainingSamples)
        {
            if (line == null)
                return; 

            String[] dividedString = line.Split(',');

            //standardize inputs
            double[] standardizedPixelValues = new double[784];
            for (int i = 0; i < standardizedPixelValues.Length; i++)
                standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 256.0;

            //classify output
            double[] targets = new double[10];
            int trgIndex = int.Parse(dividedString[0]);
            targets[trgIndex] = 1;

            lock (trainingSamples)
            {
                trainingSamples.Add(new TrainingSample(standardizedPixelValues, targets));
            }
        }

        public static void trainSample(int batchIdx, int sampleIdx, Network nn, List<TrainingSample> trainingSamples, List<double> sampleMses)
        {
            double sampleMse = nn.backPropagate(trainingSamples[batchIdx * Network.batchSize + sampleIdx].inputs, trainingSamples[batchIdx * Network.batchSize + sampleIdx].targets);
            lock (sampleMses)
            {
                sampleMses.Add(sampleMse);
            }
        }
    }
}
