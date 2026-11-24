package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func Test_ConfigValidation(t *testing.T) {
	weight := NewNormal(newRand(), 1, 0)
	tests := map[string]*Config{
		"nil":                          nil,
		"no inputs":                    {Layout: []int{1}, Weight: weight},
		"no layers":                    {Inputs: 1, Weight: weight},
		"empty layer":                  {Inputs: 1, Layout: []int{2, 0}, Weight: weight},
		"no weight initializer":        {Inputs: 1, Layout: []int{1}},
		"unknown activation":           {Inputs: 1, Layout: []int{1}, Activation: 99, Weight: weight},
		"unknown mode":                 {Inputs: 1, Layout: []int{1}, Mode: 99, Weight: weight},
		"unknown loss":                 {Inputs: 1, Layout: []int{1}, Loss: 99, Weight: weight},
		"hidden softmax":               {Inputs: 1, Layout: []int{2, 2}, Activation: ActivationSoftmax, Weight: weight},
		"softmax with a mode":          {Inputs: 1, Layout: []int{2}, Activation: ActivationSoftmax, Mode: ModeMultiClass, Weight: weight},
		"softmax with one output":      {Inputs: 1, Layout: []int{1}, Mode: ModeMultiClass, Weight: weight},
		"cross entropy for regression": {Inputs: 1, Layout: []int{1}, Mode: ModeRegression, Loss: LossCrossEntropy, Weight: weight},
		"binary CE for multi-class":    {Inputs: 1, Layout: []int{2}, Mode: ModeMultiClass, Loss: LossBinaryCrossEntropy, Weight: weight},
		"cross entropy for binary":     {Inputs: 1, Layout: []int{1}, Mode: ModeBinary, Loss: LossCrossEntropy, Weight: weight},
		"binary CE for tanh outputs":   {Inputs: 1, Layout: []int{1}, Activation: ActivationTanh, Loss: LossBinaryCrossEntropy, Weight: weight},
	}
	for name, c := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := NewNeural(c)
			assert.ErrorIs(t, err, ErrInvalidConfig)
		})
	}
}

func Test_ConfigDefaults(t *testing.T) {
	tests := []struct {
		mode    Mode
		outputs int
		loss    LossType
	}{
		{ModeDefault, 1, LossMeanSquared},
		{ModeRegression, 1, LossMeanSquared},
		{ModeBinary, 1, LossBinaryCrossEntropy},
		{ModeMultiClass, 2, LossCrossEntropy},
		{ModeMultiLabel, 2, LossBinaryCrossEntropy},
	}
	for _, test := range tests {
		c := &Config{Inputs: 1, Layout: []int{test.outputs}, Mode: test.mode, Weight: NewNormal(newRand(), 1, 0)}
		n := newTestNeural(t, c)
		assert.Equal(t, test.loss, n.Config.Loss, "mode %d", test.mode)
		assert.Equal(t, ActivationSigmoid, n.Config.Activation)
		assert.Equal(t, LossNone, c.Loss, "the caller's config is left alone")
	}
}

func Test_ConfigCanBeReused(t *testing.T) {
	c := &Config{Inputs: 1, Layout: []int{3, 1}, Activation: ActivationTanh, Mode: ModeRegression, Weight: NewNormal(newRand(), 1, 0)}
	n := newTestNeural(t, c)
	want, err := n.Predict([]float64{2})
	require.NoError(t, err)

	c.Activation = ActivationReLU
	c.Layout[0] = 5
	c.Mode = ModeMultiClass

	dumped, err := n.Marshal()
	require.NoError(t, err)
	restored, err := Unmarshal(dumped)
	require.NoError(t, err)
	got, err := restored.Predict([]float64{2})
	require.NoError(t, err)
	assert.Equal(t, want, got)
}
