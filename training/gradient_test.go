package training

import (
	"fmt"
	"math"
	"testing"

	deep "github.com/patrikeh/go-deep"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	gradientStep      = 1e-6
	gradientTolerance = 1e-7
)

// Test_Gradients compares the gradients of both trainers with the measured slope of the
// reported loss, for every mode's default loss and for MSE where it can be chosen instead.
func Test_Gradients(t *testing.T) {
	modes := []struct {
		name string
		mode deep.Mode
		loss deep.LossType // LossNone selects the mode's default
	}{
		{"default", deep.ModeDefault, deep.LossNone},
		{"regression", deep.ModeRegression, deep.LossNone},
		{"binary", deep.ModeBinary, deep.LossNone},
		{"binary-mse", deep.ModeBinary, deep.LossMeanSquared},
		{"multiclass", deep.ModeMultiClass, deep.LossNone},
		{"multiclass-mse", deep.ModeMultiClass, deep.LossMeanSquared},
		{"multilabel", deep.ModeMultiLabel, deep.LossNone},
		{"multilabel-mse", deep.ModeMultiLabel, deep.LossMeanSquared},
	}
	activations := []struct {
		name       string
		activation deep.ActivationType
	}{
		{"sigmoid", deep.ActivationSigmoid},
		{"tanh", deep.ActivationTanh},
		{"relu", deep.ActivationReLU},
		{"linear", deep.ActivationLinear},
		{"softmax", deep.ActivationSoftmax},
	}
	// Three outputs matter: with two, many wrong softmax Jacobians look right by symmetry.
	shapes := []struct {
		layout []int
		ideal  []float64
	}{
		{[]int{1}, []float64{1}},
		{[]int{3}, []float64{0.7, 0.2, 0.1}},
		{[]int{4, 1}, []float64{1}},
		{[]int{4, 2}, []float64{0.8, 0.2}},
		{[]int{4, 3}, []float64{0.7, 0.2, 0.1}},
		{[]int{5, 4, 3, 3}, []float64{0.7, 0.2, 0.1}},
	}

	for _, m := range modes {
		for _, a := range activations {
			for _, s := range shapes {
				config := &deep.Config{
					Inputs:     3,
					Layout:     s.layout,
					Activation: a.activation,
					Mode:       m.mode,
					Loss:       m.loss,
					Bias:       true,
					Weight:     distinctWeights(),
				}
				example := Example{Input: []float64{0.3, 0.2, 0.7}, Response: s.ideal}
				t.Run(fmt.Sprintf("%s/%s/%v", m.name, a.name, s.layout), func(t *testing.T) {
					assertGradients(t, deep.NewNeural(config), example)
				})
			}
		}
	}
}

// distinctWeights is deterministic, breaks symmetry and stays positive to avoid the ReLU kink.
func distinctWeights() deep.WeightInitializer {
	var i float64
	return func() float64 {
		i++
		return 0.2 + 0.15*math.Sin(i)
	}
}

func assertGradients(t *testing.T, n *deep.Neural, e Example) {
	t.Helper()

	require.NoError(t, n.Forward(e.Input))
	online := &OnlineTrainer{internal: newTraining(n.Layers)}
	online.calculateDeltas(n, e.Response)
	batch := &BatchTrainer{internalb: newBatchTraining(n.Layers, 1)}
	batch.calculateDeltas(n, e.Response, 0)

	// MeanSquared.Df differentiates ½Σ(y-t)², whereas MeanSquared.F reports Σ(y-t)²/outputs.
	scale := 1.0
	if n.Config.Loss == deep.LossMeanSquared {
		scale = 2 / float64(len(e.Response))
	}

	// Copy the gradients first, because measuring a slope overwrites every Synapse.In.
	gradients := make([][][]float64, len(n.Layers))
	for i, layer := range n.Layers {
		gradients[i] = make([][]float64, len(layer.Neurons))
		for j, neuron := range layer.Neurons {
			for _, synapse := range neuron.In {
				gradients[i][j] = append(gradients[i][j], scale*online.deltas[i][j]*synapse.In)
			}
		}
	}

	for i, layer := range n.Layers {
		for j, neuron := range layer.Neurons {
			for k, synapse := range neuron.In {
				slope := measureSlope(n, synapse, e)
				// Absolute, because a relative error is undefined for a zero gradient.
				if !assert.InDelta(t, slope, gradients[i][j][k], gradientTolerance, "online trainer, weight [%d][%d][%d]", i, j, k) ||
					!assert.InDelta(t, slope, scale*batch.partialDeltas[0][i][j][k], gradientTolerance, "batch trainer, weight [%d][%d][%d]", i, j, k) {
					return // one mismatch per network is enough
				}
			}
		}
	}
}

// measureSlope estimates ∂loss/∂weight by nudging the weight in both directions.
func measureSlope(n *deep.Neural, synapse *deep.Synapse, e Example) float64 {
	weight := synapse.Weight
	defer func() { synapse.Weight = weight }()

	lossAt := func(w float64) float64 {
		synapse.Weight = w
		return deep.GetLoss(n.Config.Loss).F([][]float64{n.Predict(e.Input)}, [][]float64{e.Response})
	}
	return (lossAt(weight+gradientStep) - lossAt(weight-gradientStep)) / (2 * gradientStep)
}
