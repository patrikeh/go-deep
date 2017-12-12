package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_softmax(t *testing.T) {
	assert.Equal(t, Sum(Softmax([]float64{0.5, 1, 1, 2.5})), 1.0)

	s := Softmax([]float64{1, 2, 3, 4, 1, 2, 3})
	e := []float64{0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175}
	for i := range s {
		assert.InEpsilon(t, e[i], s[i], 0.05)
	}
}
