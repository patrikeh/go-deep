package training

import deep "github.com/patrikeh/go-deep"

// outputDeltas sets deltas to ∂loss/∂(net input) for each neuron of the output layer.
func outputDeltas(output *deep.Layer, lossType deep.LossType, ideal, deltas []float64) {
	loss := deep.GetLoss(lossType)
	for i, n := range output.Neurons {
		deltas[i] = loss.Df(n.Value, ideal[i], n.DActivate(n.Value))
	}

	// Loss.Df sees one output at a time, which is enough unless softmax couples them.
	// Cross entropy cancels the coupling. MSE does not, so apply softmax's Jacobian:
	// delta[i] = p[i]·(error[i] - Σ p[j]·error[j]).
	if output.A != deep.ActivationSoftmax || lossType != deep.LossMeanSquared {
		return
	}
	var weightedError float64
	for i, n := range output.Neurons {
		weightedError += n.Value * deltas[i]
	}
	for i, n := range output.Neurons {
		deltas[i] = n.Value * (deltas[i] - weightedError)
	}
}
