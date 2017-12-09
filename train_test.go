package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Train(t *testing.T) {
	data := []Example{
		Example{[]float64{-5}, []float64{0}},
		Example{[]float64{-5}, []float64{0}},
		Example{[]float64{-5}, []float64{0}},
		Example{[]float64{5}, []float64{1}},
		Example{[]float64{5}, []float64{1}},
	}

	nn := NewNeural(1, []int{1}, Sigmoid, Random)

	for i := 0; i < 1000; i++ {
		for _, data := range data {
			Backpropagate(nn, data, 0.1, 0)
		}
	}
	v := nn.Feed([]float64{-5})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)
	v = nn.Feed([]float64{5})
	assert.InEpsilon(t, 1.0, v[0], 0.1)

}
