package deep

// Neuron is a neural network node
type Neuron struct {
	A     ActivationType `json:"-"`
	In    []*Synapse
	Out   []*Synapse
	Value float64 `json:"-"`
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire() {
	var sum float64
	for _, s := range n.In {
		sum += s.Out
	}
	n.Value = n.Activate(sum)

	nVal := n.Value
	for _, s := range n.Out {
		s.fire(nVal)
	}
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x float64) float64 {
	return GetActivation(n.A).F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x float64) float64 {
	return GetActivation(n.A).Df(x)
}

// Synapse is an edge between neurons
type Synapse struct {
	Weight  float64
	In, Out float64 `json:"-"`
	IsBias  bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

func (s *Synapse) fire(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
