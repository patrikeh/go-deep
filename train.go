package deep

import (
	"fmt"
	"math"
	"math/rand"
)

type Example struct {
	Input    []float64
	Response []float64
}

type Examples []Example

func (e Examples) Shuffle() {
	for i := len(e) - 1; i >= 0; i-- {
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

func (n *Neural) Train(examples Examples, epochs int, lr, lambda, momentum float64) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(examples); j++ {
			n.Learn(examples[j], lr, lambda, momentum)
		}
	}
}

func (n *Neural) TrainWithCrossValidation(examples, validation Examples, iterations, xvi int, lr, reg, momentum float64) {
	for i := 0; i < iterations; i++ {
		n.Train(examples, 1, lr, reg, momentum)
		if xvi > 0 && i%xvi == 0 {
			e := n.CrossValidate(examples)
			fmt.Printf("Iteration %d | Error: %.5f\n", i, e)
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
	n.Back(e.Response, lr, lambda/float64(len(e.Input)), momentum)
}

func mse(estimate, actual []float64) float64 {
	n := len(estimate)

	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Pow(estimate[i]-actual[i], 2)
	}

	return sum / float64(n)
}

func (n *Neural) Back(ideal []float64, lr, lambda, momentum float64) {
	last := len(n.Layers) - 1

	for i, neuron := range n.Layers[last].Neurons {
		n.t.deltas[last][i] = Act(neuron.A).df(neuron.Value) * (neuron.Value - ideal[i])
	}

	for i := last - 1; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * n.t.deltas[i+1][k]
			}
			n.t.deltas[i][j] = Act(neuron.A).df(neuron.Value) * sum
		}
	}

	for i, l := range n.Layers {
		idx := 0
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := lr*n.t.deltas[i][j]*l.Neurons[j].In[k].In - l.Neurons[j].In[k].Weight*lr*lambda
				l.Neurons[j].In[k].Weight -= (delta + momentum*n.t.oldDeltas[i][idx])
				n.t.oldDeltas[i][idx] = delta
				idx++
			}
		}
	}
}
