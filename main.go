package main

import (
	"bytes"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/mattn/go-sqlite3"
)

type model struct {
	W []float64 `json:"w"`
	M float64   `json:"m"`
}

type cfg struct {
	Name    string  `json:"name"`
	Rate    float64 `json:"rate"`
	NTrains int     `json:"ntrains"`
}

type logistic_regression struct {
	conn *sqlite3.SQLiteConn
	rand *rand.Rand
	cfg  *cfg
	X    []*mat.VecDense
	y    []float64
	maxy float64
}

func createLogisticRegressionTrain(conn *sqlite3.SQLiteConn) func() *logistic_regression {
	return func() *logistic_regression {
		return &logistic_regression{
			conn: conn,
			rand: rand.New(rand.NewSource(time.Now().UnixNano())),
			cfg:  nil,
			X:    nil,
			y:    nil,
		}
	}
}

func toArg(arg interface{}) float64 {
	var f float64
	switch t := arg.(type) {
	case uint8:
		f = float64(t)
	case uint16:
		f = float64(t)
	case uint32:
		f = float64(t)
	case uint64:
		f = float64(t)
	case int8:
		f = float64(t)
	case int16:
		f = float64(t)
	case int32:
		f = float64(t)
	case int64:
		f = float64(t)
	case int:
		f = float64(t)
	case uint:
		f = float64(t)
	case float32:
		f = float64(t)
	case float64:
		f = float64(t)
	}
	return f
}

func toArgs(args []interface{}) []float64 {
	fargs := make([]float64, len(args))
	for i, arg := range args {
		fargs[i] = toArg(arg)
	}
	return fargs
}

func (s *logistic_regression) Step(config string, args ...interface{}) error {
	fargs := toArgs(args)
	x := mat.NewVecDense(len(fargs)-1, fargs[0:len(fargs)-1])
	y := fargs[len(fargs)-1]

	if s.cfg == nil {
		err := json.NewDecoder(strings.NewReader(config)).Decode(&s.cfg)
		if err != nil {
			return err
		}
		s.X = []*mat.VecDense{}
		s.y = []float64{}
		s.maxy = y
	} else if y > s.maxy {
		s.maxy = y
	}

	s.X = append(s.X, x)
	s.y = append(s.y, y)
	return nil
}

func softmax(w, x *mat.VecDense) float64 {
	v := mat.Dot(w, x)
	return 1.0 / (1.0 + math.Exp(-v))
}

func (s *logistic_regression) Done() (string, error) {
	ws := make([]float64, s.X[0].Len())
	for i := range ws {
		ws[i] = s.rand.Float64()
	}
	for i := range s.y {
		s.y[i] = s.y[i] / (s.maxy + 1)
	}
	w := mat.NewVecDense(len(ws), ws)
	y := mat.NewVecDense(len(s.y), s.y)
	for n := 0; n < s.cfg.NTrains; n++ {
		for i, x := range s.X {
			t := mat.NewVecDense(x.Len(), nil)
			t.CopyVec(x)
			pred := softmax(w, x)
			perr := y.At(i, 0) - pred
			scale := s.cfg.Rate * perr * pred * (1 - pred)
			dx := mat.NewVecDense(x.Len(), nil)
			dx.CopyVec(x)
			dx.ScaleVec(scale, x)
			for j := 0; j < x.Len(); j++ {
				w.AddVec(w, dx)
			}
		}
	}

	fargs := make([]float64, w.Len())
	for i := 0; i < w.Len(); i++ {
		fargs[i] = w.AtVec(i)
	}
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(&model{
		W: fargs,
		M: s.maxy,
	})
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

func createLogisticRegressionPredict(conn *sqlite3.SQLiteConn) func(string, ...interface{}) (float64, error) {
	return func(name string, args ...interface{}) (float64, error) {
		rows, err := conn.Query(fmt.Sprintf(`
		select config from %s
		`, name), nil)
		if err != nil {
			return 0, err
		}
		defer rows.Close()

		var iargs [1]driver.Value
		err = rows.Next(iargs[:])
		if err != nil {
			return 0, err
		}

		var m model
		err = json.NewDecoder(strings.NewReader(iargs[0].(string))).Decode(&m)
		if err != nil {
			return 0, err
		}

		w := mat.NewVecDense(len(m.W), m.W)
		fargs := toArgs(args)
		x := mat.NewVecDense(len(fargs), fargs)
		f := softmax(w, x) * m.M
		return f, nil
	}
}

func main() {
	sql.Register("sqlite3_custom", &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			if err := conn.RegisterAggregator("logistic_regression_train", createLogisticRegressionTrain(conn), true); err != nil {
				return err
			}
			if err := conn.RegisterFunc("logistic_regression_predict", createLogisticRegressionPredict(conn), true); err != nil {
				return err
			}
			return nil
		},
	})

	db, err := sql.Open("sqlite3_custom", ":memory:")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	_, err = db.Exec(`attach "iris.sqlite" as iris`)
	if err != nil {
		log.Fatal(err)
	}

	_, err = db.Exec(`
	drop table if exists iris.model;
	create table iris.model(config text);
	insert into iris.model
	select
		logistic_regression_train('{
				"rate":    0.1,
				"ntrains": 5000
			}',
			sepal_length,
			sepal_width,
			petal_length,
			petal_width,
			class
		)
	from
		iris.train
	`)
	if err != nil {
		log.Fatal(err)
	}

	rows, err := db.Query(`
	select
		logistic_regression_predict('iris.model',
			sepal_length,
			sepal_width,
			petal_length,
			petal_width
		), class
	from
		iris.test
	`)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	for rows.Next() {
		var predicted, class float64
		err = rows.Scan(&predicted, &class)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf(
			"predict: %d (%d)\n",
			int(math.RoundToEven(predicted)), int(class))
	}
}
