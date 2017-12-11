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
		Weight:     Random,
		Bias:       0,
	})

	assert.Len(t, n.Layers, len(n.Config.Layout))
	for i, l := range n.Layers {
		assert.Len(t, l, n.Config.Layout[i])
	}
}
