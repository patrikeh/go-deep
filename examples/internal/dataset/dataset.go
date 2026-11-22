// Package dataset loads and prepares the CSV datasets used by the examples.
package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

// LoadCSV reads rows of a class label followed by features.
// Labels run from firstLabel to firstLabel+classes-1 and are one-hot encoded.
func LoadCSV(path string, classes, firstLabel int) (examples training.Examples, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", path, err)
	}
	defer func() { err = errors.Join(err, f.Close()) }()

	r := csv.NewReader(f)
	r.ReuseRecord = true
	for line := 1; ; line++ {
		record, err := r.Read()
		if errors.Is(err, io.EOF) {
			return examples, nil
		}
		if err != nil {
			return nil, fmt.Errorf("loading %s: %w", path, err)
		}
		e, err := toExample(record, classes, firstLabel)
		if err != nil {
			return nil, fmt.Errorf("loading %s, line %d: %w", path, line, err)
		}
		examples = append(examples, e)
	}
}

func toExample(record []string, classes, firstLabel int) (training.Example, error) {
	label, err := strconv.Atoi(record[0])
	if err != nil {
		return training.Example{}, fmt.Errorf("parsing label: %w", err)
	}
	class := label - firstLabel
	if class < 0 || class >= classes {
		return training.Example{}, fmt.Errorf("label %d is outside %d..%d", label, firstLabel, firstLabel+classes-1)
	}
	response := make([]float64, classes)
	response[class] = 1

	features := make([]float64, len(record)-1)
	for i, field := range record[1:] {
		if features[i], err = strconv.ParseFloat(field, 64); err != nil {
			return training.Example{}, fmt.Errorf("parsing feature %d: %w", i, err)
		}
	}
	return training.Example{Input: features, Response: response}, nil
}

// Standardize scales each feature to mean 0 and deviation 1, using statistics
// from fit alone so that no information leaks from the other sets into training.
func Standardize(fit training.Examples, sets ...training.Examples) {
	features := len(fit[0].Input)
	for j := range features {
		column := make([]float64, len(fit))
		for i, e := range fit {
			column[i] = e.Input[j]
		}
		mean, deviation := deep.Mean(column), deep.StandardDeviation(column)
		if deviation == 0 {
			deviation = 1
		}
		for _, set := range append([]training.Examples{fit}, sets...) {
			for _, e := range set {
				e.Input[j] = (e.Input[j] - mean) / deviation
			}
		}
	}
}

// Accuracy is the share of examples whose most likely class is the right one
func Accuracy(n *deep.Neural, examples training.Examples) (float64, error) {
	correct := 0
	for _, e := range examples {
		out, err := n.Predict(e.Input)
		if err != nil {
			return 0, fmt.Errorf("measuring accuracy: %w", err)
		}
		if deep.ArgMax(out) == deep.ArgMax(e.Response) {
			correct++
		}
	}
	return float64(correct) / float64(len(examples)), nil
}
