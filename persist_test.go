package deep

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_RestoreFromDump(t *testing.T) {
	rand.Seed(0)

	data := Examples{
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{5}, []float64{1}},
		Example{[]float64{5}, []float64{1}},
	}

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{5, 3, 1},
		Activation: ActivationSigmoid,
		Weight:     Random,
		Bias:       1,
	})

	for i := 0; i < 1000; i++ {
		for _, data := range data {
			n.Learn(data, 0.5, 0)
		}
	}

	v := n.Forward([]float64{0})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)

	dump := n.Dump()

	new := FromDump(dump)

	for i, biases := range n.Biases {
		for j, bias := range biases {
			assert.Equal(t, bias.Weight, new.Biases[i][j].Weight)
		}
	}
	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Forward([]float64{0}), new.Forward([]float64{0}))
}

func Test_Marshal(t *testing.T) {
	rand.Seed(0)

	data := Examples{
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{5}, []float64{1}},
		Example{[]float64{5}, []float64{1}},
	}

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{3, 3, 1},
		Activation: ActivationSigmoid,
		Weight:     Random,
		Bias:       1,
	})

	for i := 0; i < 1000; i++ {
		for _, data := range data {
			n.Learn(data, 0.5, 0)
		}
	}

	v := n.Forward([]float64{0})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)

	dump, err := n.Marshal()
	assert.Nil(t, err)

	new, err := Unmarshal(dump)
	assert.Nil(t, err)

	for i, biases := range n.Biases {
		for j, bias := range biases {
			assert.Equal(t, bias.Weight, new.Biases[i][j].Weight)
		}
	}
	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Forward([]float64{0}), new.Forward([]float64{0}))
}
