package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Init(t *testing.T) {
	n := NewNeural(&Config{
		Inputs:     3,
		Layout:     []int{4, 4, 2},
		Activation: ActivationTanh,
		Weight:     NewUniform(0.5, 0),
		Bias:       0,
	})

	assert.Len(t, n.Layers, len(n.Config.Layout))
	for i, l := range n.Layers {
		assert.Len(t, l.Neurons, n.Config.Layout[i])
	}
}

func Test_Sanity(t *testing.T) {
	n := NewNeural(&Config{
		Inputs:     3,
		Layout:     []int{3, 3, 3},
		Activation: ActivationReLU,
		Mode:       ModeMulti,
		Bias:       1,
	})
	weights := [][][]float64{
		[][]float64{
			[]float64{0.1, 0.4, 0.3},
			[]float64{0.3, 0.7, 0.7},
			[]float64{0.5, 0.2, 0.9},
		},
		[][]float64{
			[]float64{0.2, 0.3, 0.5},
			[]float64{0.3, 0.5, 0.7},
			[]float64{0.6, 0.4, 0.8},
		},
		[][]float64{
			[]float64{0.1, 0.4, 0.8},
			[]float64{0.3, 0.7, 0.2},
			[]float64{0.5, 0.2, 0.9},
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

	n.Forward([]float64{0.1, 0.2, 0.7})

	expected := [][]float64{
		[]float64{1.3, 1.66, 1.72},
		[]float64{0.9320110830223464, 0.9684462334302945, 0.9785427102823965},
		[]float64{0.31106226665743886, 0.27860738455524936, 0.4103303487873119},
	}
	for i := range n.Layers {
		for j, n := range n.Layers[i].Neurons {
			assert.Equal(t, expected[i][j], n.Value)
		}
	}

}
