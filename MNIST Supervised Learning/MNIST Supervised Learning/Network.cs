using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Supervised_Learning
{
    public class Network
    {
        public static Random r = new Random();
        public static double learningRate;
        public static double momentumScalar;
        public static int batchSize;

        public List<List<Neuron>> layers = new List<List<Neuron>>();
        public Network(int[] structure)
        {
            for (int layerIdx = 0; layerIdx < structure.Length; layerIdx++)
            {
                List<Neuron> layer = new List<Neuron>();
                for (int neuronIdx = 0; neuronIdx < structure[layerIdx]; neuronIdx++)
                {
                    if (layerIdx == 0)
                        layer.Add(new Neuron(0, false));
                    else if(layerIdx == structure.Length - 1)
                        layer.Add(new Neuron(structure[layerIdx - 1], true));
                    else
                        layer.Add(new Neuron(structure[layerIdx - 1], false));
                }
                layers.Add(layer);
            }
        }

        public Network(Network original)
        {
            for (int layerIdx = 0; layerIdx < original.layers.Count; layerIdx++)
            {
                List<Neuron> layer = new List<Neuron>();
                for (int neuronIdx = 0; neuronIdx < original.layers[layerIdx].Count; neuronIdx++)
                {
                    layer.Add(new Neuron(original.layers[layerIdx][neuronIdx]));
                }
                layers.Add(layer);
            }
        }

        public double[] forwardPropagateBeforeSoftmax(double[] inputs)
        {
            //First Layer Setup
            for (int i = 0; i < inputs.Length; i++)
                layers[0][i].firstLayerSetup(inputs[i]);
            //layers[0].Select((n, i) => (n, i)).ToList().ForEach(t => t.n.firstLayerSetup(inputs[t.i]));

            //Propagate Forward
            for (int layerIdx = 1; layerIdx < layers.Count; layerIdx++)
                layers[layerIdx].ForEach(neuron => neuron.calcActivationValue(layers[layerIdx - 1]));

            //Convert from Neuron to Double
            return convertLayerToDoubles(layers[layers.Count - 1]);
        }
        public double[] forwardPropagate(double[] inputs)
        {
            double[] outputsBeforeSoftmax = forwardPropagateBeforeSoftmax(inputs);

            //Apply Softmax
            return applySoftmax(outputsBeforeSoftmax);
        }

        public double backPropagate(double[] inputs, double[] targets)
        {
            double[] outputsBeforeSoftmax = forwardPropagateBeforeSoftmax(inputs);
            double[] outputs = applySoftmax(outputsBeforeSoftmax); //after Softmax
            double[] cost = new double[targets.Length];
            double[] costGradient = new double[targets.Length]; //technically the activation gradient for the outer layer

            int idx = 0;
            outputs.ToList().ForEach(o =>
            {
                cost[idx] = Math.Pow(o - targets[idx], 2);
                costGradient[idx] = 2 * (o - targets[idx]);
                idx++;
            });

            for (int layerIdx = layers.Count - 1; layerIdx > 0; layerIdx--) //don't count the first layer, since that's just input
            {
                List<Neuron> layer = layers[layerIdx];
                List<Neuron> previousLayer = layers[layerIdx - 1];
                for (int neuronIdx = 0; neuronIdx < layers[layerIdx].Count; neuronIdx++)
                {
                    Neuron neuron = layer[neuronIdx];
                    if (layerIdx == layers.Count - 1)
                        neuron.neuronGradient = costGradient[neuronIdx] * softmaxDerivative(outputsBeforeSoftmax)[neuronIdx]; //reset neuron gradient for each sample
                    else
                    {
                        //calculate activation gradient
                        double activationGradient = 0;
                        layers[layerIdx + 1].ForEach(frontNeuron => activationGradient += frontNeuron.neuronGradient * frontNeuron.weights[neuronIdx]);

                        neuron.neuronGradient = activationGradient * neuron.derivativeActivation(neuron.neuronValue);
                    }

                    neuron.biasGradient += neuron.neuronGradient;
                    for (int weightIdx = 0; weightIdx < neuron.weights.Length; weightIdx++)
                        neuron.weightGradients[weightIdx] += neuron.neuronGradient * previousLayer[weightIdx].activationValue;
                }
            }

            //calculate mse for this sample
            double mse = 0;
            cost.ToList().ForEach(c => mse += c);
            return mse / cost.Length;
        }

        public double[] convertLayerToDoubles(List<Neuron> layer)
        {
            double[] newArray = new double[layer.Count];
            int neuronIdx = 0;
            layer.ForEach(neuron =>
            {
                newArray[neuronIdx] = neuron.activationValue;
                neuronIdx++;
            });

            return newArray;
        }

        public double[] applySoftmax(double[] array)
        {
            double expSum = 0;
            for (int i = 0; i < array.Length; i++)
                expSum += Math.Exp(array[i]);

            double[] newArray = new double[array.Length];
            for (int i = 0; i < newArray.Length; i++)
                newArray[i] = Math.Exp(array[i]) / expSum;

            return newArray;
        }

        public double[] softmaxDerivative(double[] array)
        {
            double[] softmaxArray = applySoftmax(array);
            double[] newArray = new double[array.Length];
            for (int i = 0; i < newArray.Length; i++)
                newArray[i] = softmaxArray[i] * (1 - softmaxArray[i]);
            return newArray;
        }

        public void updateWeightsAndBiases()
        {
            layers.ForEach(layer => layer.ForEach(neuron => neuron.updateWeightsAndBias()));
        }
    }
}
