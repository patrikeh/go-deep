package deep

type Neuron struct {
	Activation Activation
	In         []*Synapse
	Out        []*Synapse
	Value      float64
}

func NewNeuron(activation Activation) *Neuron {
	return &Neuron{
		Activation: activation,
	}
}

func (n *Neuron) Fire() {
	var sum float64
	for _, s := range n.In {
		sum += s.Out
	}

	n.Value = n.Activation.f(sum)

	for _, s := range n.Out {
		s.Fire(n.Value)
	}
}

type Synapse struct {
	Weight  float64
	In, Out float64
}

func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

func (s *Synapse) Fire(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
