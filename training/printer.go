package training

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	deep "github.com/patrikeh/go-deep"
)

// StatsPrinter prints training progress
type StatsPrinter struct {
	w *tabwriter.Writer
}

// NewStatsPrinter creates a StatsPrinter
func NewStatsPrinter() *StatsPrinter {
	return &StatsPrinter{tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)}
}

// Init initializes printer
func (p *StatsPrinter) Init(n *deep.Neural) error {
	header := fmt.Sprintf("Epochs\tElapsed\tLoss (%s)\t", n.Config.Loss)
	if n.Config.Mode == deep.ModeMultiClass {
		header += "Accuracy\t\n---\t---\t---\t---\t\n"
	} else {
		header += "\n---\t---\t---\t\n"
	}
	if _, err := fmt.Fprint(p.w, header); err != nil {
		return fmt.Errorf("printing header: %w", err)
	}
	return nil
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *deep.Neural, validation Examples, elapsed time.Duration, iteration int) error {
	loss, err := crossValidate(n, validation)
	if err != nil {
		return err
	}
	accuracy, err := formatAccuracy(n, validation)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(p.w, "%d\t%s\t%.4f\t%s\n", iteration, elapsed.String(), loss, accuracy); err != nil {
		return fmt.Errorf("printing progress: %w", err)
	}
	if err := p.w.Flush(); err != nil {
		return fmt.Errorf("printing progress: %w", err)
	}
	return nil
}

func formatAccuracy(n *deep.Neural, validation Examples) (string, error) {
	if n.Config.Mode != deep.ModeMultiClass {
		return "", nil
	}
	a, err := accuracy(n, validation)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%.2f\t", a), nil
}

func accuracy(n *deep.Neural, validation Examples) (float64, error) {
	correct := 0
	for _, e := range validation {
		est, err := n.Predict(e.Input)
		if err != nil {
			return 0, fmt.Errorf("measuring accuracy: %w", err)
		}
		if deep.ArgMax(e.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation)), nil
}

func crossValidate(n *deep.Neural, validation Examples) (float64, error) {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		var err error
		if predictions[i], err = n.Predict(validation[i].Input); err != nil {
			return 0, fmt.Errorf("cross validating: %w", err)
		}
		responses[i] = validation[i].Response
	}
	return deep.GetLoss(n.Config.Loss).F(predictions, responses), nil
}
