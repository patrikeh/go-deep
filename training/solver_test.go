package training

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The expected updates are worked out by hand from each solver's definition.
func Test_Solvers(t *testing.T) {
	tests := []struct {
		name      string
		solver    Solver
		gradients []float64
		updates   []float64
	}{
		{
			name:      "sgd",
			solver:    NewSGD(0.1, 0, 0, false),
			gradients: []float64{1, -2, 0.5},
			updates:   []float64{-0.1, 0.2, -0.05},
		},
		{
			name:      "sgd with decay",
			solver:    NewSGD(0.1, 0, 1, false),
			gradients: []float64{1, -2, 0.5},
			updates:   []float64{-0.05, 0.1, -0.025},
		},
		{
			// v = 0.9v - 0.1g, update = v
			name:      "momentum",
			solver:    NewSGD(0.1, 0.9, 0, false),
			gradients: []float64{1, -2, 0.5},
			updates:   []float64{-0.1, 0.11, 0.049},
		},
		{
			// v = 0.9v - 0.1g, update = 0.9v - 0.1g, and only v is remembered
			name:      "nesterov",
			solver:    NewSGD(0.1, 0.9, 0, true),
			gradients: []float64{1, -2, 0.5},
			updates:   []float64{-0.19, 0.299, -0.0059},
		},
		{
			// with bias correction, a constant gradient moves by exactly lr at every step
			name:      "adam, constant gradient",
			solver:    NewAdam(0.1, 0.9, 0.999, 1e-8),
			gradients: []float64{0.5, 0.5, 0.5, 0.5, 0.5},
			updates:   []float64{-0.1, -0.1, -0.1, -0.1, -0.1},
		},
		{
			// m = -0.01/0.19, v = 1
			name:      "adam, reversed gradient",
			solver:    NewAdam(0.1, 0.9, 0.999, 1e-8),
			gradients: []float64{1, -1},
			updates:   []float64{-0.1, 0.1 * 0.01 / 0.19},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const weights, epoch = 2, 1 // every step is in the first epoch

			// the second pass checks that Init forgets everything from the first
			for range 2 {
				test.solver.Init(weights)
				for step, gradient := range test.gradients {
					for idx := range weights {
						assert.InDelta(t, test.updates[step], test.solver.Update(0, gradient, epoch, idx), 1e-6, "step %d, weight %d", step, idx)
					}
				}
			}
		})
	}
}
