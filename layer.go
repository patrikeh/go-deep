package deep

import "fmt"

type Layer struct {
	Neurons []*Neuron
	A       ActivationType
}

func NewLayer(n int, activation ActivationType) *Layer {
	neurons := make([]*Neuron, n)

	for i := 0; i < n; i++ {
		act := activation
		if activation == ActivationSoftmax {
			act = ActivationLinear
		}
		neurons[i] = NewNeuron(act)
	}
	return &Layer{
		Neurons: neurons,
		A:       activation,
	}
}

func (l *Layer) Fire() {
	for _, n := range l.Neurons {
		n.Fire()
	}
	if l.A == ActivationSoftmax {
		outs := make([]float64, len(l.Neurons))
		for i, neuron := range l.Neurons {
			outs[i] = neuron.Value
		}
		sm := Softmax(outs)
		for i, neuron := range l.Neurons {
			neuron.Value = sm[i]
		}
	}
}

func (l *Layer) Connect(next *Layer, weight WeightInitializer) {
	for i := range l.Neurons {
		for j := range next.Neurons {
			syn := NewSynapse(weight())
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			next.Neurons[j].In = append(next.Neurons[j].In, syn)
		}
	}
}

func (l *Layer) ApplyBias(weight WeightInitializer) []*Synapse {
	biases := make([]*Synapse, len(l.Neurons))
	for i := range l.Neurons {
		biases[i] = NewSynapse(weight())
		biases[i].IsBias = true
		l.Neurons[i].In = append(l.Neurons[i].In, biases[i])
	}
	return biases
}

func (l Layer) String() string {
	weights := make([][]float64, len(l.Neurons))
	for i, n := range l.Neurons {
		weights[i] = make([]float64, len(n.In))
		for j, s := range n.In {
			weights[i][j] = s.Weight
		}
	}
	return fmt.Sprintf("%+v", weights)
}
