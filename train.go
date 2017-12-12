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
		if xvi > 0 && i%xvi == 0 {
			e := n.CrossValidate(examples)
			fmt.Printf("Iteration %d | Error: %.5f\n", i, e)
		}
	}
}

func (n *Neural) CrossValidate(validation Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Forward(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return n.Config.Error(responses, predictions)
}

func (n *Neural) Learn(e Example, lr, lambda float64) {
	n.Forward(e.Input)
	n.Back(e.Response, lr, lambda/float64(len(e.Input)))
}

func mse(estimate, actual []float64) float64 {
	n := len(estimate)

	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Pow(estimate[i]-actual[i], 2)
	}

	return sum / float64(n)
}

func (n *Neural) Back(ideal []float64, lr, lambda float64) {
	errors := make([][]float64, len(n.Layers))

	last := len(n.Layers) - 1
	errors[last] = make([]float64, len(n.Layers[last].Neurons))

	for i, n := range n.Layers[last].Neurons {
		errors[last][i] = Act(n.A).df(n.Value) * (ideal[i] - n.Value)
	}

	for i := last - 1; i >= 0; i-- {
		errors[i] = make([]float64, len(n.Layers[i].Neurons))
		for j, n := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range n.Out {
				sum += s.Weight * errors[i+1][k]
			}
			errors[i][j] = Act(n.A).df(n.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j, n := range l.Neurons {
			for _, s := range n.In {
				s.Weight += lr * errors[i][j] * s.In
				s.Weight -= s.Weight * lr * lambda // L2 regularization lambda in (0,1)
			}
		}
	}
}
