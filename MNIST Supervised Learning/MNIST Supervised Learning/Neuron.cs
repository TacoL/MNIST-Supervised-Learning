using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Supervised_Learning
{
    public class Neuron
    {
        public double neuronValue, activationValue, bias, neuronGradient, biasGradient, previousBG;
        public double[] weights, weightGradients, previousWG;

        private bool output;

        public Neuron(int numInputs, bool output)
        {
            this.output = output;

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

        public Neuron(Neuron original)
        {
            output = original.output;

            neuronValue = 0;
            activationValue = 0;
            neuronGradient = 0;

            weights = new double[original.weights.Length];
            weightGradients = new double[original.weightGradients.Length];
            previousWG = new double[original.previousWG.Length];

            for (int weightIdx = 0; weightIdx < weights.Length; weightIdx++)
                weights[weightIdx] = original.weights[weightIdx];

            bias = original.bias;
            biasGradient = 0;
            previousBG = 0;
        }

        public void calcActivationValue(List<Neuron> inputNeurons)
        {
            neuronValue = 0; //reset neuron value every time

            for (int i = 0; i < inputNeurons.Count; i++)
                neuronValue += inputNeurons[i].activationValue * weights[i];
            neuronValue += bias;


            activationValue = output ? outputActivation(neuronValue) : activationFunction(neuronValue);
        }

        public double activationFunction(double x)
        {
            return Math.Tanh(x);

            //return Math.Tanh(x/125);

            //return x <= 0 ? 0 : activationValue;
        }

        public double derivativeActivation(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);

            //double secant = 2.0 / (Math.Exp(x/125) + Math.Exp(-x/125));
            //return Math.Pow(secant, 2) / 125.0;

            //return x <= 0 ? 0 : 1;
        }

        public double outputActivation(double x)
        {
            //softmax
            return x;
        }

        public double derivativeOutputActivation(double x)
        {
            //for softmax, not used for an individual neuron
            return x;
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