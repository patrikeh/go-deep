package deep

import (
	"fmt"
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

func (n *Neural) Train(examples Examples, epochs int, lr, lambda float64) {
	for i := 0; i < epochs; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			n.Learn(examples[j], lr, lambda)
		}
	}
}

func (n *Neural) TrainWithCrossValidation(examples, validation Examples, iterations, xvi int, lr, reg float64) {
	for i := 0; i < iterations; i++ {
		n.Train(examples, 1, lr, reg)
		e := n.CrossValidate(examples)
		if i%xvi == 0 {
			fmt.Printf("Iteration %d | Error: %.5f\n", i, e)
		}
	}
}

func (n *Neural) CrossValidate(validation Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Feed(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return n.Config.Error(responses, predictions)
}

func (n *Neural) Learn(e Example, lr, lambda float64) {
	n.Feed(e.Input)
	n.Backpropagate(e, lr, lambda)
}

func (n *Neural) Backpropagate(e Example, lr, lambda float64) {
	lambda = lambda / float64(len(e.Input))
	deltas := make([][]float64, len(n.Layers))

	last := len(n.Layers) - 1
	deltas[last] = make([]float64, len(n.Layers[last]))
	for i, n := range n.Layers[last] {
		deltas[last][i] = n.Activation.df(n.Value) * (e.Response[i] - n.Value)
	}

	for i := last - 1; i >= 0; i-- {
		l := n.Layers[i]
		deltas[i] = make([]float64, len(l))
		for j, n := range l {
			var sum float64
			for k, s := range n.Out {
				sum += s.Weight * deltas[i+1][k]
			}
			deltas[i][j] = n.Activation.df(n.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j, n := range l {
			for _, s := range n.In {
				s.Weight -= s.Weight * (lr * lambda) // L2 regularization lambda in (0,1)
				s.Weight += lr * deltas[i][j] * s.In
			}
		}
	}
}
