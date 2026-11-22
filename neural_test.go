package deep

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestNeural(t *testing.T, c *Config) *Neural {
	t.Helper()
	n, err := NewNeural(c)
	require.NoError(t, err)
	return n
}

func newRand() *rand.Rand {
	return rand.New(rand.NewSource(0))
}

func Test_Init(t *testing.T) {
	n := newTestNeural(t, &Config{
		Inputs:     3,
		Layout:     []int{4, 4, 2},
		Activation: ActivationTanh,
		Mode:       ModeBinary,
		Weight:     NewUniform(newRand(), 0.5, 0),
		Bias:       true,
	})

	assert.Len(t, n.Layers, len(n.Config.Layout))
	for i, l := range n.Layers {
		assert.Len(t, l.Neurons, n.Config.Layout[i])
	}
}

func Test_Forward(t *testing.T) {
	n := newTestNeural(t, &Config{
		Inputs:     3,
		Layout:     []int{3, 3, 3},
		Activation: ActivationReLU,
		Mode:       ModeMultiClass,
		Weight:     NewNormal(newRand(), 1.0, 0),
		Bias:       true,
	})
	weights := [][][]float64{
		{
			{0.1, 0.4, 0.3},
			{0.3, 0.7, 0.7},
			{0.5, 0.2, 0.9},
		},
		{
			{0.2, 0.3, 0.5},
			{0.3, 0.5, 0.7},
			{0.6, 0.4, 0.8},
		},
		{
			{0.1, 0.4, 0.8},
			{0.3, 0.7, 0.2},
			{0.5, 0.2, 0.9},
		},
	}
	for _, n := range n.Layers[1].Neurons {
		n.A = ActivationSigmoid
	}
	for i, l := range n.Layers {
		for j, n := range l.Neurons {
			for k := 0; k < 3; k++ {
				n.In[k].Weight = weights[i][j][k]
			}
		}
	}
	for _, biases := range n.Biases {
		for _, bias := range biases {
			bias.Weight = 1
		}
	}

	err := n.Forward([]float64{0.1, 0.2, 0.7})
	assert.Nil(t, err)

	expected := [][]float64{
		{1.3, 1.66, 1.72},
		{0.9320110830223464, 0.9684462334302945, 0.9785427102823965},
		{0.31106226665743886, 0.27860738455524936, 0.4103303487873119},
	}
	for i := range n.Layers {
		for j, n := range n.Layers[i].Neurons {
			assert.InEpsilon(t, expected[i][j], n.Value, 1e-12)
		}
	}

	err = n.Forward([]float64{0.1, 0.2})
	assert.ErrorIs(t, err, ErrDimension)

	clone, err := n.Clone()
	require.NoError(t, err)
	want, err := n.Predict([]float64{0.1, 0.2, 0.7})
	require.NoError(t, err)
	got, err := clone.Predict([]float64{0.1, 0.2, 0.7})
	require.NoError(t, err)
	assert.Equal(t, want, got, "a clone keeps per-neuron activations")
}

func Test_PredictRejectsWrongDimension(t *testing.T) {
	n := newTestNeural(t, &Config{Inputs: 2, Layout: []int{3, 1}, Weight: NewNormal(newRand(), 1, 0)})
	_, err := n.Predict([]float64{1, 2})
	require.NoError(t, err)

	out, err := n.Predict([]float64{1})
	assert.ErrorIs(t, err, ErrDimension)
	assert.Nil(t, out, "must not return the previous prediction")
}

func Test_PredictConcurrently(t *testing.T) {
	n := newTestNeural(t, &Config{Inputs: 2, Layout: []int{16, 16, 1}, Weight: NewNormal(newRand(), 1, 0)})
	inputs := [][]float64{{1, 2}, {-3, 4}, {0, 0}, {5, -5}}
	want := make([][]float64, len(inputs))
	for i, input := range inputs {
		var err error
		want[i], err = n.Predict(input)
		require.NoError(t, err)
	}

	const rounds = 200
	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Go(func() {
			for range rounds {
				got, err := n.Predict(input)
				assert.NoError(t, err)
				assert.Equal(t, want[i], got)
			}
		})
	}
	wg.Wait()
}

func Test_NumWeights(t *testing.T) {
	n := newTestNeural(t, &Config{Inputs: 2, Layout: []int{5, 5, 3}, Weight: NewNormal(newRand(), 1, 0)})
	assert.Equal(t, 5*2+5*5+3*5, n.NumWeights())
}

func Test_RegressionHasOutputBias(t *testing.T) {
	n := newTestNeural(t, &Config{Inputs: 1, Layout: []int{1}, Mode: ModeRegression, Bias: true, Weight: NewNormal(newRand(), 1, 0)})
	assert.Len(t, n.Layers[0].Neurons[0].In, 2)
}
