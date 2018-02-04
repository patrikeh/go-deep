package training

import (
	"time"

	deep "github.com/patrikeh/go-deep"
)

type Trainer struct {
	*internal
	optimizer Optimizer
	printer   *StatsPrinter
	verbosity int
}

func NewTrainer(optimizer Optimizer, verbosity int) *Trainer {
	return &Trainer{
		optimizer: optimizer,
		printer:   NewStatsPrinter(),
		verbosity: verbosity,
	}
}

type internal struct {
	deltas  [][]float64
	moments [][][]float64
}

func newTraining(layers []*deep.Layer) *internal {
	deltas := make([][]float64, len(layers))
	moments := make([][][]float64, len(layers))

	for i, l := range layers {
		deltas[i] = make([]float64, len(l.Neurons))
		moments[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			moments[i][j] = make([]float64, len(n.In))
		}
	}
	return &internal{
		deltas:  deltas,
		moments: moments,
	}
}

func (t *Trainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internal = newTraining(n.Layers)

	train := make(Examples, len(examples))
	copy(train, examples)

	t.printer.Init(n)

	ts := time.Now()
	for i := 0; i <= iterations; i++ {
		t.train(n, train, 1)
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), i)
		}
	}
}

func (t *Trainer) train(n *deep.Neural, examples Examples, epochs int) {
	for i := 0; i < epochs; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j])
		}
	}
}

func (t *Trainer) learn(n *deep.Neural, e Example) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n)
}

func (t *Trainer) calculateDeltas(n *deep.Neural, ideal []float64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
			neuron.Value,
			ideal[i],
			neuron.DActivate(neuron.Value))
	}

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

func (t *Trainer) update(n *deep.Neural) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				update := t.optimizer.Update(l.Neurons[j].In[k].Weight,
					t.deltas[i][j]*l.Neurons[j].In[k].In,
					t.moments[i][j][k])
				l.Neurons[j].In[k].Weight += update
				t.moments[i][j][k] = update
			}
		}
	}
}
