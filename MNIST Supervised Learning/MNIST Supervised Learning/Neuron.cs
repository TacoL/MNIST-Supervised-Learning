using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Neuron
    {
        public double neuronValue, activationValue, bias, neuronGradient, biasGradient, previousBG;
        public double[] weights, weightGradients, previousWG;

        public Neuron(int numInputs)
        {
            neuronValue = 0;
            activationValue = 0;
            neuronGradient = 0;

            weights = new double[numInputs];
            weightGradients = new double[numInputs];
            previousWG = new double[numInputs];
            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
                weights[weightIdx] = Network.r.NextDouble();

            bias = 0;
            biasGradient = 0;
            previousBG = 0;
        }

        public void calcActivationValue(List<Neuron> inputNeurons)
        {
            neuronValue = 0; //reset neuron value every time

            for (int i = 0; i < inputNeurons.Count; i++)
                neuronValue += inputNeurons[i].activationValue * weights[i];
            neuronValue += bias;


            activationValue = activationFunction(neuronValue);
        }

        public double activationFunction(double x)
        {
            //return Math.Tanh(x);
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double derivativeActivation(double x)
        {
            //return 1 - Math.Pow(Math.Tanh(x), 2);
            return activationFunction(x) * (1 - activationFunction(x));
        }
        public void firstLayerSetup(double actV)
        {
            activationValue = actV;
        }

        public void updateWeightsAndBias()
        {
            //update weights and biases
            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
            {
                weights[weightIdx] -= (weightGradients[weightIdx] / Network.batchSize * Network.learningRate) + (previousWG[weightIdx] * Network.momentumScalar);
                previousWG[weightIdx] = weightGradients[weightIdx] / Network.batchSize;
            }
                
            bias -= (biasGradient / Network.batchSize * Network.learningRate) + (previousBG * Network.momentumScalar);
            previousBG = biasGradient / Network.batchSize;

            //reset neuron and weight and bias GRADIENTS after each update (update every batch)
            neuronGradient = 0;
            for (int wgIdx = 0; wgIdx < weightGradients.Length; wgIdx++)
                weightGradients[wgIdx] = 0;
            biasGradient = 0;
        }
    }
}