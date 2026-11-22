package deep

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Tanh(t *testing.T) {
	for _, x := range []float64{math.Inf(-1), -1000, -355, -1, 0, 1e-12, 1, 355, 1000, math.Inf(1)} {
		t.Run(fmt.Sprint(x), func(t *testing.T) {
			assert.Equal(t, math.Tanh(x), Tanh{}.F(x))
		})
	}
}
