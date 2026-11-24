package deep

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func Test_RestoreFromDump(t *testing.T) {
	n := newTestNeural(t, &Config{
		Inputs:     1,
		Layout:     []int{5, 3, 1},
		Activation: ActivationSigmoid,
		Weight:     NewUniform(newRand(), 0.5, 0),
		Bias:       true,
	})

	restored, err := FromDump(n.Dump())
	require.NoError(t, err)
	assertSameNetwork(t, n, restored)
}

func Test_Marshal(t *testing.T) {
	n := newTestNeural(t, &Config{
		Inputs:     1,
		Layout:     []int{3, 3, 1},
		Activation: ActivationSigmoid,
		Mode:       ModeRegression,
		Weight:     NewUniform(newRand(), 0.5, 0),
		Bias:       true,
	})

	dump, err := n.Marshal()
	require.NoError(t, err)
	restored, err := Unmarshal(dump)
	require.NoError(t, err)
	assertSameNetwork(t, n, restored)
}

func assertSameNetwork(t *testing.T, want, got *Neural) {
	t.Helper()
	assert.Equal(t, want.Weights(), got.Weights())
	assert.Equal(t, want.String(), got.String())
	wantOut, err := want.Predict([]float64{0})
	require.NoError(t, err)
	gotOut, err := got.Predict([]float64{0})
	require.NoError(t, err)
	assert.Equal(t, wantOut, gotOut)
}

// The dumps in testdata were written by the version before configs were validated.
func Test_UnmarshalEarlierDumps(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("testdata", "dumps", "*.json"))
	require.NoError(t, err)
	require.NotEmpty(t, paths)

	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			require.NoError(t, err)
			var fixture struct {
				Dump       json.RawMessage
				Input      []float64
				Prediction []float64
			}
			require.NoError(t, json.Unmarshal(raw, &fixture))

			n, err := Unmarshal(fixture.Dump)
			require.NoError(t, err)
			got, err := n.Predict(fixture.Input)
			require.NoError(t, err)
			assert.InDeltaSlice(t, fixture.Prediction, got, 1e-15)
		})
	}
}

func Test_UnmarshalRejectsMalformedDumps(t *testing.T) {
	tests := map[string]string{
		"not json":          `{`,
		"no config":         `{}`,
		"null config":       `{"Config": null}`,
		"invalid config":    `{"Config": {"Inputs": 1, "Layout": []}, "Weights": []}`,
		"too few layers":    `{"Config": {"Inputs": 1, "Layout": [1, 1]}, "Weights": [[[1]]]}`,
		"too few neurons":   `{"Config": {"Inputs": 1, "Layout": [2]}, "Weights": [[[1]]]}`,
		"too few weights":   `{"Config": {"Inputs": 2, "Layout": [1]}, "Weights": [[[1]]]}`,
		"too many weights":  `{"Config": {"Inputs": 1, "Layout": [1]}, "Weights": [[[1, 2, 3]]]}`,
		"huge layer":        `{"Config": {"Inputs": 1, "Layout": [1000000000]}, "Weights": [[[1]]]}`,
		"huge input":        `{"Config": {"Inputs": 1000000000, "Layout": [1]}, "Weights": [[[1]]]}`,
		"negative layer":    `{"Config": {"Inputs": 1, "Layout": [-1]}, "Weights": [[]]}`,
		"unknown loss type": `{"Config": {"Inputs": 1, "Layout": [1], "Loss": 9}, "Weights": [[[1]]]}`,
	}
	for name, dump := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := Unmarshal([]byte(dump))
			assert.Error(t, err)
		})
	}
}
