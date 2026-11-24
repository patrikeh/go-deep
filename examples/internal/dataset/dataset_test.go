package dataset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/patrikeh/go-deep/training"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeCSV(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "data.csv")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}

func Test_LoadCSV(t *testing.T) {
	examples, err := LoadCSV(writeCSV(t, "1,0.5,2\n3,1.5,-1\n"), 3, 1)
	require.NoError(t, err)
	assert.Equal(t, training.Examples{
		{Input: []float64{0.5, 2}, Response: []float64{1, 0, 0}},
		{Input: []float64{1.5, -1}, Response: []float64{0, 0, 1}},
	}, examples)
}

func Test_LoadCSVRejectsMalformedRows(t *testing.T) {
	tests := map[string]string{
		"uneven rows":      "1,2,3\n2,5\n",
		"unclosed quote":   "1,2,3\n2,\"5\n",
		"label too large":  "4,1,2\n",
		"label too small":  "0,1,2\n",
		"non-numeric data": "1,x,2\n",
	}
	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := LoadCSV(writeCSV(t, content), 3, 1)
			assert.Error(t, err)
		})
	}
	_, err := LoadCSV(filepath.Join(t.TempDir(), "missing.csv"), 3, 1)
	assert.ErrorIs(t, err, os.ErrNotExist)
}

func Test_StandardizeFitsOnlyTheFirstSet(t *testing.T) {
	train := training.Examples{{Input: []float64{1, 7}}, {Input: []float64{3, 7}}}
	test := training.Examples{{Input: []float64{5, 9}}}
	Standardize(train, test)

	assert.InDeltaSlice(t, []float64{-0.7071, 0}, train[0].Input, 1e-4)
	assert.InDeltaSlice(t, []float64{0.7071, 0}, train[1].Input, 1e-4)
	assert.InDeltaSlice(t, []float64{2.1213, 2}, test[0].Input, 1e-4)
}
