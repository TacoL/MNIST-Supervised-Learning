using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MNIST_Supervised_Learning
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        public static async Task Main()
        {
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            
            //set up the network
            Network.learningRate = 0.2;
            Network.momentumScalar = 0.02;
            Network.batchSize = 32;
            Network mainNN = new Network(new int[] { 784, 100, 10 });
            int numEpochs = 15;

            //set up training samples
            //assuming a (row x column) image

            List<TrainingSample> trainingSamples = new List<TrainingSample>();

            StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            String line = sr.ReadLine(); //skips first line
            int setupIdx = 0;
            List<Task> samplesToAdd = new List<Task>();
            while ((line = sr.ReadLine()) != null)
            {
                String lineDuplicate = line;
                Task t = new Task(() => createSample(lineDuplicate, trainingSamples));
                samplesToAdd.Add(t);
                Console.WriteLine($"Sample: {setupIdx}");
                setupIdx++;
            }

            samplesToAdd.ForEach(task => task.Start());
            Task.WaitAll(samplesToAdd.ToArray());

            sr.Close();
            Console.WriteLine("Ready to train");

            //train network
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                double mse = 0;
                int numBatches = trainingSamples.Count / Network.batchSize;

                //batching
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) //for each batch
                {
                    double batchMse = trainBatch(batchIdx, mainNN, trainingSamples);

                    mse += batchMse / Network.batchSize;
                    mainNN.updateWeightsAndBiases();
                    //Console.WriteLine($"Epoch {epoch + 1} / {numEpochs}      Batch #{batchIdx + 1} / {numBatches}      BMSE = {batchMse / Network.batchSize}");
                }

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / numBatches);
            }

            //results
            testNetwork(mainNN, "mnist_train.csv");
            testNetwork(mainNN, "mnist_test.csv");

            Application.Run(new Form1(mainNN));
        }

        public static void testNetwork(Network mainNN, string fileName)
        {
            StreamReader srTest = new StreamReader(File.OpenRead(fileName));
            String line = srTest.ReadLine(); //skips first line

            int successes = 0;
            int total = 0;
            while ((line = srTest.ReadLine()) != null)
            {
                String lineDuplicate = line;
                String[] dividedString = lineDuplicate.Split(',');

                //standardize inputs
                double[] standardizedPixelValues = new double[784];
                for (int i = 0; i < standardizedPixelValues.Length; i++)
                    standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 255.0;

                //print output
                Network sampleNN = new Network(mainNN);
                double[] output = sampleNN.forwardPropagate(standardizedPixelValues);
                int label = int.Parse(dividedString[0]);
                int val = output.ToList().IndexOf(output.Max());
                if (val == label)
                    successes++;
                total++;
            }

            srTest.Close();
            Console.WriteLine($"{successes}, {total}");
            Console.WriteLine("Success Rate: " + ((double)successes / (double)total * 100d) + "%");
        }

        public static void addToGradients(Network mainNN, Network sampleNN)
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
        }

        public static void createSample(String line, List<TrainingSample> trainingSamples)
        {
            if (line == null)
            {
                Console.WriteLine("line is null");
                return;
            }

            String[] dividedString = line.Split(',');

            //standardize inputs
            double[] standardizedPixelValues = new double[784];
            for (int i = 0; i < standardizedPixelValues.Length; i++)
                standardizedPixelValues[i] = double.Parse(dividedString[i + 1]) / 255.0;

            //classify output
            double[] targets = new double[10];
            int trgIndex = int.Parse(dividedString[0]);
            targets[trgIndex] = 1;

            lock (trainingSamples)
            {
                trainingSamples.Add(new TrainingSample(standardizedPixelValues, targets));
            }
        }

        public static double trainSample(int batchIdx, int sampleIdx, Network mainNN, List<TrainingSample> trainingSamples)
        {
            double sampleMse = 0;

            Network sampleNN = new Network(mainNN);
            int sampleIdxToTest = batchIdx * Network.batchSize + sampleIdx;
            sampleMse = sampleNN.backPropagate(trainingSamples[sampleIdxToTest].inputs, trainingSamples[sampleIdxToTest].targets);
            lock (mainNN)
            {
                addToGradients(mainNN, sampleNN); //idea: perhaps lock the mainNN for this part?
            }

            return sampleMse;
        }

        public static double trainBatch(int batchIdx, Network mainNN, List<TrainingSample> trainingSamples)
        {
            double batchMse = 0;

            List<Task> tasks = new List<Task>();
            for (int sampleIdx = 0; sampleIdx < Network.batchSize; sampleIdx++)
            {
                int thisSampleIdx = sampleIdx; // makes it work with Tasks
                tasks.Add(new Task(() => batchMse += trainSample(batchIdx, thisSampleIdx, mainNN, trainingSamples)));
            }

            tasks.ForEach(task => task.Start());
            Task.WaitAll(tasks.ToArray());

            return batchMse;
        }
    }
}
