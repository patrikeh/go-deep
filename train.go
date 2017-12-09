package deep

type Example struct {
	Input    []float64
	Response []float64
}

/*
func TrainWithValidation(n *Neural, examples []Example, validation []Example, epochs int, lr, lambda float64) {
}
func Train(n *Neural, examples []Example, epochs int, lr, lambda float64) {}
*/
func Backpropagate(n *Neural, e Example, lr, lambda float64) {
	n.Feed(e.Input)

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
