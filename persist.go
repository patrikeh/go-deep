package deep

import (
	"encoding/json"
)

type Dump struct {
	Config  *Config
	Weights [][][]float64
}

func (n Neural) Dump() *Dump {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([]float64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weight
			}
		}
	}

	return &Dump{
		Config:  n.Config,
		Weights: weights,
	}
}

func FromDump(dump *Dump) *Neural {
	neural := NewNeural(dump.Config)

	for i, l := range neural.Layers {
		for j, n := range l.Neurons {
			for k := range n.In {
				neural.Layers[i].Neurons[j].In[k].Weight = dump.Weights[i][j][k]
			}
		}
	}

	return neural
}

func (n Neural) Marshal() ([]byte, error) {
	d := n.Dump()
	return json.Marshal(d)
}

func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}
