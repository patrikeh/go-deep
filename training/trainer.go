package training

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	deep "github.com/patrikeh/go-deep"
)

type Trainer struct {
	lr, lambda, momentum float64
	*training
	verbosity int
}

func NewTrainer(lr, lambda, momentum float64, verbosity int) *Trainer {
	return &Trainer{
		lr:        lr,
		lambda:    lambda,
		momentum:  momentum,
		verbosity: verbosity,
	}
}

type training struct {
	deltas    [][]float64
	oldDeltas [][][]float64
}

func newTraining(layers []*deep.Layer) *training {
	deltas := make([][]float64, len(layers))
	oldDeltas := make([][][]float64, len(layers))

	for i, l := range layers {
		deltas[i] = make([]float64, len(l.Neurons))
		oldDeltas[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			oldDeltas[i][j] = make([]float64, len(n.In))
		}
	}
	return &training{
		deltas:    deltas,
		oldDeltas: oldDeltas,
	}
}

func (t *Trainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.training = newTraining(n.Layers)

	train := make(Examples, len(examples))
	copy(train, examples)

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprint(w, "Epochs\tElapsed\tError\t\n---\t---\t---\t\n")

	ts := time.Now()
	for i := 0; i < iterations; i++ {
		t.train(n, train, 1, t.lr, t.lambda, t.momentum)
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			rms := t.CrossValidate(n, validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+t.verbosity, time.Since(ts).String(), rms)
			w.Flush()
		}
	}
}

func (t *Trainer) train(n *deep.Neural, examples Examples, epochs int, lr, lambda, momentum float64) {
	for i := 0; i < epochs; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], lr, lambda, momentum)
		}
	}
}

func (t *Trainer) CrossValidate(n *deep.Neural, validation Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return n.Config.Error(responses, predictions)
}

func (t *Trainer) learn(n *deep.Neural, e Example, lr, lambda, momentum float64) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n, lr, lambda, momentum)
}

func (t *Trainer) calculateDeltas(n *deep.Neural, ideal []float64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[len(n.Layers)-1][i] = deep.Act(neuron.A).Df(neuron.Value) * (neuron.Value - ideal[i])
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * t.deltas[i+1][k]
			}
			t.deltas[i][j] = deep.Act(neuron.A).Df(neuron.Value) * sum
		}
	}
}

func (t *Trainer) update(n *deep.Neural, lr, lambda, momentum float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := lr*t.deltas[i][j]*l.Neurons[j].In[k].In + momentum*t.oldDeltas[i][j][k]
				var reg float64
				if !l.Neurons[j].In[k].IsBias {
					reg = l.Neurons[j].In[k].Weight * lr * lambda
				}
				l.Neurons[j].In[k].Weight -= delta + reg
				t.oldDeltas[i][j][k] = delta
			}
		}
	}
}
