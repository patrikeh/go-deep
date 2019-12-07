package training

import (
	"math/rand"
	"runtime"
	"testing"

	deep "github.com/patrikeh/go-deep"
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
