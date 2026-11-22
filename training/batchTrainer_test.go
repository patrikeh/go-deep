package training

import (
	"math/rand"
	"runtime"
	"testing"

	deep "github.com/patrikeh/go-deep"
	"github.com/stretchr/testify/assert"
)

func Benchmark_xor(b *testing.B) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{32, 32, 1},
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.NewUniform(.25, 0),
		Bias:       true,
	})
	exs := Examples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}
	const minExamples = 4000
	var dupExs Examples
	for len(dupExs) < minExamples {
		dupExs = append(dupExs, exs...)
	}

	for i := 0; i < b.N; i++ {
		const iterations = 20
		solver := NewAdam(0.001, 0.9, 0.999, 1e-8)
		trainer := NewBatchTrainer(solver, iterations, len(dupExs)/2, runtime.NumCPU())
		trainer.Train(n, dupExs, dupExs, iterations)
	}
}

func batchTestNet() *deep.Neural {
	var i float64
	weight := func() float64 {
		i++
		return 0.1 + i/50
	}
	return deep.NewNeural(&deep.Config{Inputs: 1, Layout: []int{3, 1}, Weight: weight})
}

func Test_BatchGradientIsAveraged(t *testing.T) {
	example := Example{[]float64{0.5}, []float64{1}}
	for _, size := range []int{1, 2, 32, 128} {
		online, batched := batchTestNet(), batchTestNet()
		NewTrainer(NewSGD(0.1, 0, 0, false), 0).Train(online, Examples{example}, nil, 1)

		var copies Examples
		for range size {
			copies = append(copies, example)
		}
		NewBatchTrainer(NewSGD(0.1, 0, 0, false), 0, size, 3).Train(batched, copies, nil, 1)

		want, got := online.Weights(), batched.Weights()
		for i := range want {
			for j := range want[i] {
				assert.InDeltaSlice(t, want[i][j], got[i][j], 1e-12, "batch size %d", size)
			}
		}
	}
}

func Test_BatchTrainerStopsItsWorkers(t *testing.T) {
	const trainings, workers = 20, 8
	before := runtime.NumGoroutine()
	for range trainings {
		NewBatchTrainer(NewSGD(0.1, 0, 0, false), 0, 1, workers).Train(batchTestNet(), Examples{{[]float64{0}, []float64{0}}}, nil, 1)
	}
	// workers may still be returning: yield to them rather than sleep
	for i := 0; i < 100_000 && runtime.NumGoroutine() > before; i++ {
		runtime.Gosched()
	}
	assert.LessOrEqual(t, runtime.NumGoroutine(), before)
}

func Test_TrainLeavesExamplesInOrder(t *testing.T) {
	var examples Examples
	for i := range 16 {
		examples = append(examples, Example{[]float64{float64(i)}, []float64{0}})
	}
	original := append(Examples(nil), examples...)
	NewTrainer(NewSGD(0.1, 0, 0, false), 0).Train(batchTestNet(), examples, nil, 3)
	assert.Equal(t, original, examples)
}
