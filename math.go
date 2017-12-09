package deep

import "math"

func mean(xx []float64) float64 {
	var sum float64
	for _, x := range xx {
		sum += x
	}
	return sum / float64(len(xx))
}

func variance(xx []float64) float64 {
	if len(xx) == 1 {
		return 0.0
	}
	m := mean(xx)

	var variance float64
	for _, x := range xx {
		variance += math.Pow((x - m), 2)
	}

	return variance / float64(len(xx)-1)
}

func standardDeviation(xx []float64) float64 {
	return math.Sqrt(variance(xx))
}

// z-score μ=0 σ=1
func standardize(xx []float64) {
	m := mean(xx)
	s := standardDeviation(xx)

	if s == 0 {
		s = 1
	}

	for i, x := range xx {
		xx[i] = (x - m) / s
	}
}

// scale to (0,1)
func normalize(xx []float64) {
	min, max := min(xx), max(xx)
	for i, x := range xx {
		xx[i] = (x - min) / (max - min)
	}
}

func min(xx []float64) float64 {
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

func max(xx []float64) float64 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

func sgn(x float64) float64 {
	switch {
	case x < 0:
		return -1.0
	case x > 0:
		return 1.0
	}
	return 0
}
