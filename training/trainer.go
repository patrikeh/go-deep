package training

import (
	"fmt"
	"math/rand"
	"slices"
	"time"

	deep "github.com/patrikeh/go-deep"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *deep.Neural, examples, validation Examples, iterations int) error
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	solver    Solver
	printer   *StatsPrinter
	verbosity int
	random    *rand.Rand
}

// NewTrainer creates a new trainer, which shuffles the examples with random
func NewTrainer(solver Solver, verbosity int, random *rand.Rand) *OnlineTrainer {
	return &OnlineTrainer{
		solver:    solver,
		printer:   NewStatsPrinter(),
		verbosity: verbosity,
		random:    random,
	}
}

type internal struct {
	deltas [][]float64
}

func newTraining(layers []*deep.Layer) *internal {
	deltas := make([][]float64, len(layers))
	for i, l := range layers {
		deltas[i] = make([]float64, len(l.Neurons))
	}
	return &internal{
		deltas: deltas,
	}
}

// Train trains n
func (t *OnlineTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) error {
	if err := checkTraining(n, examples, validation, iterations, t.solver, t.random); err != nil {
		return fmt.Errorf("training: %w", err)
	}
	t.internal = newTraining(n.Layers)
	t.solver.Init(n.NumWeights())
	report := t.verbosity > 0 && len(validation) > 0
	if report {
		if err := t.printer.Init(n); err != nil {
			return fmt.Errorf("training: %w", err)
		}
	}

	train := slices.Clone(examples)

	ts := time.Now()
	for i := 1; i <= iterations; i++ {
		train.Shuffle(t.random)
		for _, e := range train {
			if err := t.learn(n, e, i); err != nil {
				return fmt.Errorf("training: %w", err)
			}
		}
		if report && i%t.verbosity == 0 {
			if err := t.printer.PrintProgress(n, validation, time.Since(ts), i); err != nil {
				return fmt.Errorf("training: %w", err)
			}
		}
	}
	return nil
}

func (t *OnlineTrainer) learn(n *deep.Neural, e Example, it int) error {
	if err := n.Forward(e.Input); err != nil {
		return err
	}
	t.calculateDeltas(n, e.Response)
	t.update(n, it)
	return nil
}

func (t *OnlineTrainer) calculateDeltas(n *deep.Neural, ideal []float64) {
	last := len(n.Layers) - 1
	outputDeltas(n.Layers[last], n.Config.Loss, ideal, t.deltas[last])

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * t.deltas[i+1][k]
			}
			t.deltas[i][j] = neuron.DActivate(neuron.Value) * sum
		}
	}
}

func (t *OnlineTrainer) update(n *deep.Neural, it int) {
	var idx int
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				update := t.solver.Update(l.Neurons[j].In[k].Weight,
					t.deltas[i][j]*l.Neurons[j].In[k].In,
					it,
					idx)
				l.Neurons[j].In[k].Weight += update
				idx++
			}
		}
	}
}
