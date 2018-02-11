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

func Test_Standardize(t *testing.T) {

	s := []float64{10.0, 5.0, 0.0}

	assert.Equal(t, Mean(s), 5.0)
	assert.Equal(t, StandardDeviation(s), 5.0)

	Standardize(s)

	zscores := []float64{1, 0, -1}

	for i, x := range s {
		assert.Equal(t, zscores[i], x)
	}
}

func Test_Normalize(t *testing.T) {

	s := []float64{10.0, 5.0, 0.0}

	assert.Equal(t, Mean(s), 5.0)
	assert.Equal(t, StandardDeviation(s), 5.0)

	Normalize(s)

	zscores := []float64{1, 0.5, 0}

	for i, x := range s {
		assert.Equal(t, zscores[i], x)
	}
}

func Test_MinMax(t *testing.T) {
	s := []float64{5.0, 10.0, 0.0}

	assert.Equal(t, 0.0, Min(s))
	assert.Equal(t, 10.0, Max(s))
	assert.Equal(t, 1, ArgMax(s))
}

func Test_Dot(t *testing.T) {
	assert.Equal(t, 17.0, Dot([]float64{1.0, 6.0, 3.0}, []float64{2.0, 2.0, 1.0}))
}

func Test_Sgn(t *testing.T) {
	assert.Equal(t, Sgn(0), 0.)
	assert.Equal(t, Sgn(-5), -1.)
	assert.Equal(t, Sgn(3), 1.)
}
