package deep

type Neuron struct {
	A     ActivationType `json:"-"`
	In    []*Synapse
	Out   []*Synapse
	Value float64 `json:"-"`
}

func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) Fire() {
	var sum float64
	for _, s := range n.In {
		sum += s.Out
	}
	n.Value = Act(n.A).F(sum)

	for _, s := range n.Out {
		s.Fire(n.Value)
	}
}

type Synapse struct {
	Weight  float64
	In, Out float64 `json:"-"`
	IsBias  bool
}

func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

func (s *Synapse) Fire(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
