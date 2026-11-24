package deep

import "math/rand"

// A WeightInitializer returns a (random) weight
type WeightInitializer func() float64

// NewUniform returns a weight generator sampling from u(mean-width/2, mean+width/2)
func NewUniform(random *rand.Rand, width, mean float64) WeightInitializer {
	return func() float64 { return Uniform(random, width, mean) }
}

// Uniform samples a value from u(mean-width/2, mean+width/2)
func Uniform(random *rand.Rand, width, mean float64) float64 {
	return (random.Float64()-0.5)*width + mean
}

// NewNormal returns a weight generator sampling from N(μ, σ)
func NewNormal(random *rand.Rand, stdDev, mean float64) WeightInitializer {
	return func() float64 { return Normal(random, stdDev, mean) }
}

// Normal samples a value from N(μ, σ)
func Normal(random *rand.Rand, stdDev, mean float64) float64 {
	return random.NormFloat64()*stdDev + mean
}

func zeroWeight() float64 { return 0 }
