import java.util.Random;
import java.util.Vector;

public class backProbagation {
	double hiddenWeights[][], outputWeights[][], netHidden[], I[], netOutput[],
			O[], error[], outputError[], tmpOutputWeight[][], eeta = 0.3,
			hiddenError[], tmpHiddenWeight[][];
	double  totError , biasHidden = 0.5 , biasOutput = 0.1;
	int hiddenNeurons, outputNeurons, inputNeurons;
	Vector<Double> input , Y;

	void iterate(Vector<Double> in,Vector<Double> out ) {
		input = (Vector<Double>) in.clone();
		Y = (Vector<Double>) out.clone();
		computeNetHidden();
		computeNetOutput();
		updateOutputWeights();
		computeHiddenError();
		updateHiddenWeigths();
		finaLize();
		computeMSE();
	}

	backProbagation(int inp, int hid, int out) {
		inputNeurons = inp;
		hiddenNeurons = hid;
		outputNeurons = out;
		hiddenWeights = new double[inputNeurons][hiddenNeurons];
		outputWeights = new double[hiddenNeurons][outputNeurons];
		tmpOutputWeight = new double[hiddenNeurons][outputNeurons];
		tmpHiddenWeight = new double[inputNeurons][hiddenNeurons];
		netHidden = new double[hiddenNeurons];
		I = new double[hiddenNeurons];
		netOutput = new double[outputNeurons];
		O = new double[outputNeurons];
		error = new double[outputNeurons];
		outputError = new double[outputNeurons];
		hiddenError = new double[hiddenNeurons];
		initialize();
	}

	void computeNetHidden() {
		for (int j = 0; j < hiddenNeurons; ++j) {
			double sum = 0;
			for (int i = 0; i < inputNeurons; ++i)
				sum += hiddenWeights[i][j] * input.get(i);
			netHidden[j] = sum ;
			I[j] = 1.0 / (1.0 + Math.exp(-netHidden[j]));
		}
	}

	void computeNetOutput() {
		for (int j = 0; j < outputNeurons; ++j) {
			double sum = 0;
			for (int i = 0; i < hiddenNeurons; ++i)
				sum += outputWeights[i][j] * I[i];
			netOutput[j] = sum ;
			O[j] = 1.0 / (1 + Math.exp(-netOutput[j]));
			error[j] = Y.get(j) - O[j];
			outputError[j] = O[j] * (1 - O[j]) * error[j];
		}
		
	}

	void updateOutputWeights() {
		for (int i = 0; i < hiddenNeurons; ++i) {
			for (int j = 0; j < outputNeurons; ++j) {
				tmpOutputWeight[i][j] = outputWeights[i][j] + eeta
						* outputError[j] * I[i];	
			}
		}
	}

	void computeHiddenError() {
		for (int i = 0; i < hiddenNeurons; ++i) {
			double sum = 0;
			for (int j = 0; j < outputNeurons; ++j)
				sum += error[j] * outputWeights[i][j];
			hiddenError[i] = I[i] * (1 - I[i]) * sum;
		}
	}

	void updateHiddenWeigths() {
		for (int i = 0; i < inputNeurons; ++i) {
			for (int j = 0; j < hiddenNeurons; ++j)
				tmpHiddenWeight[i][j] = hiddenWeights[i][j] + eeta
						* hiddenError[j] * input.get(i);
		}
	}

	void computeMSE() {
		for (int i = 0; i < outputNeurons; ++i)
			totError += Math.abs(error[i]);
	}

	void finaLize() {
		hiddenWeights = tmpHiddenWeight.clone();
		outputWeights = tmpOutputWeight.clone();
	}

	void initialize() {
		Random rr = new Random();
		boolean b;
		double d = 2;
		for (int j = 0; j < hiddenNeurons; ++j) {
			for (int i = 0; i < inputNeurons; ++i) {
				b = rr.nextBoolean();
				if (b)
					hiddenWeights[i][j] = rr.nextDouble() * d;
				else
					hiddenWeights[i][j] = rr.nextDouble() * -d;
			}
			for (int i = 0; i < outputNeurons; ++i) {
				b = rr.nextBoolean();
				if (b)
					outputWeights[j][i] = rr.nextDouble() * d;
				else
					outputWeights[j][i] = rr.nextDouble() * -d;
			}
		}
	}
}
