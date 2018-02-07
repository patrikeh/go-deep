package deep

import "math"

// Mean of xx
func Mean(xx []float64) float64 {
	var sum float64
	for _, x := range xx {
		sum += x
	}
	return sum / float64(len(xx))
}

// Variance of xx
func Variance(xx []float64) float64 {
	if len(xx) == 1 {
		return 0.0
	}
	m := Mean(xx)

	var variance float64
	for _, x := range xx {
		variance += math.Pow((x - m), 2)
	}

	return variance / float64(len(xx)-1)
}

// StandardDeviation of xx
func StandardDeviation(xx []float64) float64 {
	return math.Sqrt(Variance(xx))
}

// Standardize (z-score) shifts distribution to μ=0 σ=1
func Standardize(xx []float64) {
	m := Mean(xx)
	s := StandardDeviation(xx)

	if s == 0 {
		s = 1
	}

	for i, x := range xx {
		xx[i] = (x - m) / s
	}
}

// Normalize scales to (0,1)
func Normalize(xx []float64) {
	min, max := Min(xx), Max(xx)
	for i, x := range xx {
		xx[i] = (x - min) / (max - min)
	}
}

// Min is the smallest element
func Min(xx []float64) float64 {
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

// Max is the largest element
func Max(xx []float64) float64 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

// ArgMax is the index of the largest element
func ArgMax(xx []float64) int {
	max, idx := xx[0], 0
	for i, x := range xx {
		if x > max {
			max, idx = xx[i], i
		}
	}
	return idx
}

// Sgn is signum
func Sgn(x float64) float64 {
	switch {
	case x < 0:
		return -1.0
	case x > 0:
		return 1.0
	}
	return 0
}

// Sum is sum
func Sum(xx []float64) (sum float64) {
	for _, x := range xx {
		sum += x
	}
	return
}

// Softmax is the softmax function
func Softmax(xx []float64) []float64 {
	out := make([]float64, len(xx))
	var sum float64
	max := Max(xx)
	for i, x := range xx {
		out[i] = math.Exp(x - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Round to nearest integer
func Round(x float64) float64 {
	return math.Floor(x + .5)
}

// Dot product
func Dot(xx, yy []float64) float64 {
	var p float64
	for i := range xx {
		p += xx[i] * yy[i]
	}
	return p
}
