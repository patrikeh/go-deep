package deep

import (
	"fmt"
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
			res:    0.69,
		},
	}
	for _, test := range tests {
		loss := GetLoss(test.loss)
		estimate := loss.F(test.input, test.target)
		assert.InEpsilon(t, test.res, estimate, 1e-1, fmt.Sprintf("%s estimate: %.2f expected: %.2f", test.loss.String(), estimate, test.res))
		assert.NotEqual(t, "N/A", test.loss.String())
	}
}
