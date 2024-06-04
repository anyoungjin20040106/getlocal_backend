package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/rocketlaunchr/dataframe-go"
	"github.com/rocketlaunchr/dataframe-go/imports"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
)

type LoacalData struct {
	local string
}

func MakeDataFrame(url string) *dataframe.DataFrame {
	reps, err := http.Get(url)
	if err != nil {
		panic(err)
	}
	data, err := ioutil.ReadAll(reps.Body)
	if err != nil {
		panic(err)
	}
	defer reps.Body.Close()
	reader := bytes.NewReader(data)
	result, err := imports.LoadFromCSV(context.Background(), reader)
	if err != nil {
		panic(err)
	}
	for _, i := range [2]int{1, 2} {
		fSeries, err := result.Series[i].(dataframe.ToSeriesFloat64).ToSeriesFloat64(context.Background(), true)
		if err != nil {
			panic(err)
		}
		result.Series[i] = fSeries
	}
	return result
}
func main() {
	df := MakeDataFrame("https://raw.githubusercontent.com/janyoungjin/localData/main/data.csv")
	instances := base.ConvertDataFrameToInstances(df, 0)
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)
	cls.AllowOptimisations = false
	cls.Fit(instances)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var x float64
		var y float64
		var err error
		if r.Method == "POST" {
			xVal := r.PostFormValue("x")
			yVal := r.PostFormValue("y")

			x, err = strconv.ParseFloat(xVal, 64)
			if err != nil {
				http.Error(w, "Invalid value for x", http.StatusBadRequest)
				return
			}

			y, err = strconv.ParseFloat(yVal, 64)
			if err != nil {
				http.Error(w, "Invalid value for y", http.StatusBadRequest)
				return
			}
		} else if r.Method == "GET" {
			y = 37.811252
			x = 127.132577
		}

		vec := base.NewDenseInstances()
		ax := base.NewFloatAttribute("x")
		ay := base.NewFloatAttribute("y")
		vec.AddAttribute(ax)
		vec.AddAttribute(ay)
		axSpec := vec.AddAttribute(ax)
		aySpec := vec.AddAttribute(ay)
		vec.AddClassAttribute(base.NewCategoricalAttribute())
		vec.Extend(1)
		vec.Set(axSpec, 0, ax.GetSysValFromString(fmt.Sprintf("%f", x)))
		vec.Set(aySpec, 0, ay.GetSysValFromString(fmt.Sprintf("%f", y)))

		predictions, err := cls.Predict(vec)
		if err != nil {
			http.Error(w, fmt.Sprintf("%s", err), http.StatusBadRequest)
		}
		local := predictions.RowString(0)
		if local == "" {
			fmt.Fprint(w, local)
			return
		}
		result := LoacalData{
			local: local,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fmt.Sprintf("%s", result))
	})
	http.ListenAndServe(":8000", nil)
}
