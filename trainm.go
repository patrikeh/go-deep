package deep

import (
	"fmt"
	"os"
	"sync"
	"text/tabwriter"
	"time"
)

func (n *Neural) TrainM(examples, validation Examples, iterations, batchSize, workers int, lr, lambda, momentum float64) {
	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Examples)
	nets := make([]*Neural, workers)
	wg := sync.WaitGroup{}
	for i := 0; i < workers; i++ {
		nets[i] = NewNeural(n.Config)
		go func(id int, workCh <-chan Examples) {
			for {
				batch := <-workCh
				nets[id].ApplyWeights(n.Weights())
				for _, e := range batch {
					nets[id].Forward(e.Input)
					nets[id].CalculateDeltasM(e.Response)
				}
				wg.Done()
			}
		}(i, workCh)
	}

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprint(w, "Epochs\tElapsed\tError\t\n---\t---\t---\t\n")
	ts := time.Now()
	for i := 0; i < iterations; i++ {
		train.Shuffle()
		batches := examples.SplitSize(batchSize)
		for i := 0; i < len(batches); i++ {
			workloads := batches[i].SplitN(workers)
			wg.Add(workers)
			for _, workload := range workloads {
				workCh <- workload
			}

			wg.Wait()
			for _, net := range nets {
				for i := range net.Layers {
					for j := range net.Layers[i].Neurons {
						for k := range net.Layers[i].Neurons[j].In {
							n.t.accumulatedDeltas[i][j][k] += net.t.accumulatedDeltas[i][j][k]
							net.t.accumulatedDeltas[i][j][k] = 0
						}
					}
				}
			}
			n.UpdateM(lr, lambda/float64(n.Config.Inputs), momentum, float64(len(batches[i])))
		}

		if n.Config.Verbosity > 0 && i%n.Config.Verbosity == 0 && len(validation) > 0 {
			rms := n.CrossValidate(validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+n.Config.Verbosity, time.Since(ts).String(), rms)
			w.Flush()
		}
	}
}

func (n *Neural) CalculateDeltasM(ideal []float64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		n.t.deltas[len(n.Layers)-1][i] = Act(neuron.A).df(neuron.Value) * (ideal[i] - neuron.Value)
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * n.t.deltas[i+1][k]
			}

			n.t.deltas[i][j] = Act(neuron.A).df(neuron.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				n.t.accumulatedDeltas[i][j][k] += n.t.deltas[i][j] * l.Neurons[j].In[k].In
			}
		}
	}
}

func (n *Neural) UpdateM(lr, lambda, momentum, batchSize float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := (1/batchSize)*lr*n.t.accumulatedDeltas[i][j][k] - l.Neurons[j].In[k].Weight*lr*lambda
				l.Neurons[j].In[k].Weight += delta + momentum*n.t.oldDeltas[i][j][k]
				n.t.oldDeltas[i][j][k] = delta
				n.t.accumulatedDeltas[i][j][k] = 0
			}
		}
	}
}
