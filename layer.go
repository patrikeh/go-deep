package deep

type Layer []*Neuron

func NewLayer(n int, activation Activation) Layer {
	l := make([]*Neuron, n)
	for i := 0; i < n; i++ {
		l[i] = NewNeuron(activation)
	}
	return l
}

func (l Layer) Connect(next Layer, bias *Synapse, weight WeightInitializer) {
	for i := range l {
		for j := range next {
			syn := NewSynapse(weight())
			l[i].Out = append(l[i].Out, syn)
			next[j].In = append(next[j].In, syn, bias)
		}
	}
}

func (l Layer) Fire() {
	for _, neuron := range l {
		neuron.Fire()
	}
}
