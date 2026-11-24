package training

import (
	"errors"
	"fmt"
	"math/rand"

	deep "github.com/patrikeh/go-deep"
)

// ErrInvalidTraining means training can't start with the given arguments
var ErrInvalidTraining = errors.New("invalid training")

// checkTraining validates everything up front, so that training never stops halfway
func checkTraining(n *deep.Neural, examples, validation Examples, iterations int, solver Solver, random *rand.Rand) error {
	switch {
	case n == nil:
		return fmt.Errorf("network is nil: %w", ErrInvalidTraining)
	case solver == nil:
		return fmt.Errorf("solver is nil: %w", ErrInvalidTraining)
	case random == nil:
		return fmt.Errorf("random source is nil: %w", ErrInvalidTraining)
	case iterations < 0:
		return fmt.Errorf("%d iterations: %w", iterations, ErrInvalidTraining)
	}

	inputs, outputs := n.Config.Inputs, len(n.Layers[len(n.Layers)-1].Neurons)
	sets := []struct {
		name     string
		examples Examples
	}{
		{"example", examples},
		{"validation example", validation},
	}
	for _, set := range sets {
		for i, e := range set.examples {
			if len(e.Input) != inputs {
				return fmt.Errorf("%s %d has %d inputs, want %d: %w", set.name, i, len(e.Input), inputs, deep.ErrDimension)
			}
			if len(e.Response) != outputs {
				return fmt.Errorf("%s %d has %d responses, want %d: %w", set.name, i, len(e.Response), outputs, deep.ErrDimension)
			}
		}
	}
	return nil
}
