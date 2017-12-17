package deep

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"
)

func (n *Neural) TrainM(examples, validation Examples, iterations, batchSize int, lr, lambda, momentum float64) {
	train := make(Examples, len(examples))
	copy(train, examples)

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprint(w, "Epochs\tElapsed\tError\t\n---\t---\t---\t\n")

	ts := time.Now()
	for i := 0; i < iterations; i++ {
		n.trainM(train, batchSize, lr, lambda, momentum)
		if n.Config.Verbosity > 0 && i%n.Config.Verbosity == 0 && len(validation) > 0 {
			rms := n.CrossValidate(validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+n.Config.Verbosity, time.Since(ts).String(), rms)
			w.Flush()
		}
	}
}

func (n *Neural) trainM(examples Examples, batchSize int, lr, lambda, momentum float64) {
	examples.Shuffle()
	batches := examples.SplitN(batchSize)
	for i := 0; i < len(batches); i++ {
		n.LearnM(batches[i], lr, lambda, momentum)
	}
}

func (n *Neural) LearnM(batch []Example, lr, lambda, momentum float64) {
	for _, e := range batch {
		n.Forward(e.Input)
		n.CalculateDeltasM(e.Response)
	}
	n.UpdateM(lr, lambda/float64(n.Config.Inputs), momentum, float64(len(batch)))
}

func (n *Neural) CalculateDeltasM(ideal []float64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		n.t.deltas[len(n.Layers)-1][i] = Act(neuron.A).df(neuron.Value) * (neuron.Value - ideal[i])
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * n.t.deltas[i+1][k]
			}

			n.t.deltas[i][j] = Act(neuron.A).df(neuron.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				n.t.accumulatedDeltas[i][j][k] += n.t.deltas[i][j] * l.Neurons[j].In[k].In
			}
		}
	}
}

func (n *Neural) UpdateM(lr, lambda, momentum, batchSize float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := (1/batchSize)*lr*n.t.accumulatedDeltas[i][j][k] - l.Neurons[j].In[k].Weight*lr*lambda
				l.Neurons[j].In[k].Weight -= delta + momentum*n.t.oldDeltas[i][j][k]
				n.t.oldDeltas[i][j][k] = delta
				n.t.accumulatedDeltas[i][j][k] = 0
			}
		}
	}
}

/*

workers int // in param long lived
work := make(chan Examples)
done := make(chan struct{})

weights := n.Weights()
nets := make([]*Neural, workers)
for i := range nets {
	net[i] := NewNeural(n.c)
	net[i].ApplyWeights(weights)
}


while training:

for each batch:
	wg.Add(n)
	in <- batches
	wg.Wait()
apply nets[...].t.deltas
accumulatedDeltas [][][]float64



*/
