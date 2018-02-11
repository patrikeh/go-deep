package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Loss(t *testing.T) {

	tests := []struct {
		loss          LossType
		input, target [][]float64
		res           float64
	}{
		{
			loss:   LossMeanSquared,
			input:  [][]float64{{0.5, 1.0, 1.5}},
			target: [][]float64{{0.0, 2.0, 2.0}},
			res:    0.5,
		},
		{
			loss:   LossCrossEntropy,
			input:  [][]float64{{0.5, 1.0, 1.5}},
			target: [][]float64{{0.0, 1.0, 1.0}},
			res:    -0.4,
		},
		{
			loss:   LossBinaryCrossEntropy,
			input:  [][]float64{{0.5}},
			target: [][]float64{{0.5}},
			res:    0.175,
		},
	}
	for _, test := range tests {
		loss := GetLoss(test.loss)
		assert.InEpsilon(t, loss.F(test.input, test.target), test.res, 1e-1)
		assert.NotEqual(t, "N/A", test.loss.String())
	}
}
