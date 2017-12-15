package deep

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"text/tabwriter"
	"time"
)

type Example struct {
	Input    []float64
	Response []float64
}

type Examples []Example

func (e Examples) Shuffle() {
	for i := range e {
		j := rand.Intn(i + 1)
		e[i], e[j] = e[j], e[i]
	}
}

func (e Examples) Split(p float64) (first, second Examples) {
	for i := 0; i < len(e); i++ {
		if p > rand.Float64() {
			first = append(first, e[i])
		} else {
			second = append(second, e[i])
		}
	}
	return
}

func (n *Neural) train(examples Examples, epochs int, lr, lambda, momentum float64) {
	for i := 0; i < epochs; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			n.Learn(examples[j], lr, lambda, momentum)
		}
	}
}

func (n *Neural) Train(examples, validation Examples, iterations int, lr, lambda, momentum float64) {
	train := make(Examples, len(examples))
	copy(train, examples)

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprint(w, "Epochs\tElapsed\tError\t\n---\t---\t---\t\n")

	ts := time.Now()
	for i := 0; i < iterations; i++ {
		n.train(train, 1, lr, lambda, momentum)
		if n.Config.Verbosity > 0 && i%n.Config.Verbosity == 0 && len(validation) > 0 {
			rms := n.CrossValidate(validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+n.Config.Verbosity, time.Since(ts).String(), rms)
			w.Flush()
		}
	}
}

func (n *Neural) CrossValidate(validation Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return n.Config.Error(responses, predictions)
}

func (n *Neural) Learn(e Example, lr, lambda, momentum float64) {
	n.Forward(e.Input)
	n.CalculateDeltas(e.Response)
	n.Update(lr, lambda/float64(len(e.Input)), momentum)
}

func mse(estimate, actual []float64) float64 {
	n := len(estimate)

	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Pow(estimate[i]-actual[i], 2)
	}

	return sum / float64(n)
}

func (n *Neural) CalculateDeltas(ideal []float64) {
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
}

func (n *Neural) Update(lr, lambda, momentum float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := lr*n.t.deltas[i][j]*l.Neurons[j].In[k].In - l.Neurons[j].In[k].Weight*lr*lambda
				l.Neurons[j].In[k].Weight -= delta + momentum*n.t.oldDeltas[i][j][k]
				n.t.oldDeltas[i][j][k] = delta
			}
		}
	}
}
