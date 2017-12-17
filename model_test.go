package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_SplitN(t *testing.T) {
	data := Examples{
		{[]float64{2.7810836, 2.550537003}, []float64{1, 0}},
		{[]float64{1.465489372, 2.362125076}, []float64{1, 0}},
		{[]float64{3.396561688, 4.400293529}, []float64{1, 0}},
		{[]float64{1.38807019, 1.850220317}, []float64{1, 0}},
		{[]float64{3.06407232, 3.005305973}, []float64{1, 0}},
		{[]float64{7.627531214, 2.759262235}, []float64{0, 1}},
		{[]float64{5.332441248, 2.088626775}, []float64{0, 1}},
		{[]float64{6.922596716, 1.77106367}, []float64{0, 1}},
		{[]float64{8.675418651, -0.242068655}, []float64{0, 1}},
	}

	batches := data.SplitN(2)
	assert.Len(t, batches, 5)
	for _, batch := range batches {
		assert.True(t, len(batch) <= 2)
	}
}
