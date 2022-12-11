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
            //Application.Run(new Form1());

            //set up the network
            Network.learningRate = 0.22;
            Network.momentumScalar = 0.12;
            Network.batchSize = 50;
            Network mainNN = new Network(new int[] { 4, 8, 3 });
            int numEpochs = 5000;

            //set up training samples
            //assuming a (row x column) image

            List<TrainingSample> trainingSamples = new List<TrainingSample>();

            //Iris
            StreamReader sr = new StreamReader(File.OpenRead("IRIS.csv"));
            String line = sr.ReadLine(); //skips first line
            while ((line = sr.ReadLine()) != null)
            {
                double[] inputs = new double[4];
                double[] targets = new double[3];
                String[] dividedString = line.Split(',');

                for (int i = 0; i < 4; i++)
                    inputs[i] = double.Parse(dividedString[i]);

                switch (dividedString[4])
                {
                    case "Iris-setosa":
                        targets[0] = 1;
                        targets[1] = 0;
                        targets[2] = 0;
                        break;
                    case "Iris-versicolor":
                        targets[0] = 0;
                        targets[1] = 1;
                        targets[2] = 0;
                        break;
                    case "Iris-virginica":
                        targets[0] = 0;
                        targets[1] = 0;
                        targets[2] = 1;
                        break;
                }

                trainingSamples.Add(new TrainingSample(inputs, targets));
            }


            //StreamReader sr = new StreamReader(File.OpenRead("mnist_train.csv"));
            //String line = sr.ReadLine(); //skips first line
            //int setupIdx = 0;
            //List<Task> samplesToAdd = new List<Task>();
            //while ((line = sr.ReadLine()) != null)
            //{
            //    String lineDuplicate = line;
            //    Task t = new Task(() => createSample(lineDuplicate, trainingSamples));
            //    samplesToAdd.Add(t);
            //    Console.WriteLine($"Sample: {setupIdx}");
            //    setupIdx++;
            //}

            //samplesToAdd.ForEach(task => task.Start());
            //Task.WaitAll(samplesToAdd.ToArray());

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
                    Console.WriteLine($"Epoch {epoch + 1} / {numEpochs}      Batch #{batchIdx + 1} / {numBatches}      BMSE = {batchMse / Network.batchSize}");
                }

                Console.WriteLine("Epoch: {0}         MSE: {1}", epoch + 1, mse / numBatches);
            }

            #region RESULTS
            //results
            //testNetwork(mainNN, "mnist_train.csv");
            //testNetwork(mainNN, "mnist_test.csv");

            //IRIS
            int successes = 0;

            for (int sampleIdx = 0; sampleIdx < trainingSamples.Count; sampleIdx++)
            {
                double[] output = mainNN.forwardPropagate(trainingSamples[sampleIdx].inputs);

                int indexOfMaxValue = 0;
                for (int idx = 0; idx < output.Length; idx++)
                {
                    if (output[idx] > output[indexOfMaxValue])
                    {
                        indexOfMaxValue = idx;
                    }
                }

                double[] targets = trainingSamples[sampleIdx].targets;

                String[] names = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
                if (targets[indexOfMaxValue] == 1)
                {
                    Console.WriteLine("Match: " + names[indexOfMaxValue]);
                    successes++;
                }
                else
                {
                    Console.WriteLine("Error: Predicted = " + names[indexOfMaxValue] + ", Actual = ");
                }
            }

            Console.WriteLine(successes + "/" + trainingSamples.Count);
            #endregion
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

            //for (int sampleIdx = 0; sampleIdx < Network.batchSize; sampleIdx++)
            //{
            //    batchMse += trainSample(batchIdx, sampleIdx, mainNN, trainingSamples);
            //}
            return batchMse;
        }
    }
}
