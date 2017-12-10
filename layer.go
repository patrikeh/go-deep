package deep

import "fmt"

type Layer []*Neuron

func NewLayer(n int, activation Activation) Layer {
	l := make([]*Neuron, n)
	for i := 0; i < n; i++ {
		l[i] = NewNeuron(activation)
	}
	return l
}

func (l Layer) Connect(next Layer, weight WeightInitializer) {
	for i := range l {
		for j := range next {
			syn := NewSynapse(weight())
			l[i].Out = append(l[i].Out, syn)
			next[j].In = append(next[j].In, syn)
		}
	}
}

func (l Layer) ApplyBias(weight WeightInitializer) []*Synapse {
	biases := make([]*Synapse, len(l))
	for i := range l {
		biases[i] = NewSynapse(weight())
		l[i].In = append(l[i].In, biases[i])
	}
	return biases
}

func (l Layer) Fire() {
	for _, neuron := range l {
		neuron.Fire()
	}
}

func (l Layer) String() string {
	weights := make([][]float64, len(l))
	for i, n := range l {
		weights[i] = make([]float64, len(n.In))
		for j, s := range n.In {
			weights[i][j] = s.Weight
		}
	}
	return fmt.Sprintf("%+v", weights)
}
