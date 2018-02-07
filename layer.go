package deep

import "fmt"

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Neurons []*Neuron
	A       ActivationType
}

// NewLayer creates a new layer with n nodes
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

func (l *Layer) fire() {
	for _, n := range l.Neurons {
		n.fire()
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

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
func (l *Layer) Connect(next *Layer, weight WeightInitializer) {
	for i := range l.Neurons {
		for j := range next.Neurons {
			syn := NewSynapse(weight())
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			next.Neurons[j].In = append(next.Neurons[j].In, syn)
		}
	}
}

// ApplyBias creates and returns a bias synapse for each neuron in l
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
