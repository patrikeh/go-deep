package deep

import (
	"testing"
)

func Test_Init(t *testing.T) {
	n := NewNeural(3, []int{4, 4, 2}, Tanh, Random)
	n.Feed([]float64{1, 1, 3})

}
