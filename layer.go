package deep

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

func (l Layer) ApplyBias(biases []*Synapse, weight WeightInitializer) {
	for i := range l {
		biases[i] = NewSynapse(weight())
		l[i].In = append(l[i].In, biases[i])
	}
}
func (l Layer) Fire() {
	for _, neuron := range l {
		neuron.Fire()
	}
}
