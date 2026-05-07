from flask import Flask, render_template, request, jsonify
import pandas as pd
import io, base64
from preprocessing.pipeline import build_preprocessor, split_data
from main import full_pipeline

try:
    from plots.plot import plot_histogram, plot_bar, plot_heatmap, plot_scatter, plot_boxplot
except ImportError:
    from plots import plot_histogram, plot_bar, plot_heatmap, plot_scatter, plot_boxplot

app = Flask(__name__)

df_global              = None
trained_model_global   = None
feature_columns_global = None
model_name_global      = None
categorical_values     = {}

@app.route("/", methods=["GET", "POST"])
def home():
    global df_global
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("index.html", error="Please upload a CSV file.")
            df = pd.read_csv(file)
            if df.empty:
                return render_template("index.html", error="The uploaded file is empty.")
            df_global = df
            return render_template(
                "index.html",
                tables=[df.head(10).to_html(classes='preview-table', index=False)],
                columns=df.columns.tolist(),
                row_count=len(df),
                col_count=len(df.columns)
            )
        except Exception as e:
            return render_template("index.html", error=f"Error reading file: {str(e)}")
    return render_template("index.html")


@app.route("/select_target", methods=["POST"])
def select_target():
    global categorical_values, df_global
    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")
    target_column = request.form["target_column"]
    categorical_values = {
        col: df_global[col].dropna().unique().tolist()
        for col in df_global.columns
        if df_global[col].dtype == "object"
    }
    return render_template("model.html", target_column=target_column)


@app.route("/train_model", methods=["POST"])
def train_model_route():
    global df_global, trained_model_global, feature_columns_global
    global categorical_values, model_name_global

    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")

    try:
        target_column     = request.form["target_column"]
        model_name        = request.form["model_name"]
        model_name_global = model_name

        params = {}
        def flt(k):
            v = request.form.get(k, "").strip(); return float(v) if v else None
        def Int(k):
            v = request.form.get(k, "").strip(); return int(v)   if v else None
        def Str(k):
            v = request.form.get(k, "").strip(); return v         if v else None

        if model_name == "Logistic Regression":
            if flt("C"):        params["C"]        = flt("C")
            if Int("max_iter"): params["max_iter"] = Int("max_iter")
            if Str("solver"):   params["solver"]   = Str("solver")
        elif model_name == "SVM":
            if flt("C"):      params["C"]      = flt("C")
            if Str("kernel"): params["kernel"] = Str("kernel")
            if Str("gamma"):  params["gamma"]  = Str("gamma")
        elif model_name == "KNN":
            if Int("n_neighbors"): params["n_neighbors"] = Int("n_neighbors")
            if Str("weights"):     params["weights"]     = Str("weights")
            if Str("metric"):      params["metric"]      = Str("metric")
        elif model_name == "Random Forest":
            if Int("n_estimators"):      params["n_estimators"]      = Int("n_estimators")
            if Int("max_depth"):         params["max_depth"]         = Int("max_depth")
            if Int("min_samples_split"): params["min_samples_split"] = Int("min_samples_split")
        elif model_name == "Linear Regression":
            params["fit_intercept"] = request.form.get("fit_intercept","true").lower() == "true"

        trained_model, metrics, feature_columns = full_pipeline(
            df_global, target_column, model_name, params)

        trained_model_global   = trained_model
        feature_columns_global = list(feature_columns)
        feature_categorical    = {c: v for c, v in categorical_values.items()
                                  if c in feature_columns_global}

        return render_template(
            "results.html",
            model_name=model_name,
            metrics={k: round(float(v), 4) for k, v in metrics.items()},
            feature_columns=feature_columns_global,
            categorical_values=feature_categorical,
            params=params
        )
    except Exception as e:
        import traceback
        return render_template("index.html", error=traceback.format_exc())


@app.route("/predict", methods=["POST"])
def predict_route():
    global trained_model_global, feature_columns_global
    if trained_model_global is None:
        return render_template("index.html", error="No trained model found.")
    try:
        input_data = {}
        for feature in feature_columns_global:
            value = request.form.get(feature, "")
            try:    value = float(value)
            except: pass
            input_data[feature] = value if value != "" else 0

        df_input   = pd.DataFrame([input_data])
        prediction = trained_model_global.predict(df_input)[0]
        probability = None
        if hasattr(trained_model_global, "predict_proba"):
            try:
                proba       = trained_model_global.predict_proba(df_input)[0]
                probability = round(float(max(proba)) * 100, 2)
            except: pass

        return render_template("prediction.html",
                               prediction=prediction,
                               probability=probability,
                               model_name=model_name_global)
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {str(e)}")


@app.route("/visualize")
def visualize():
    global df_global
    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")
    return render_template(
        "visualize.html",
        numeric_cols    =df_global.select_dtypes(include=["number"]).columns.tolist(),
        categorical_cols=df_global.select_dtypes(include=["object","category"]).columns.tolist()
    )


def _fig_to_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor="#060b14")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return b64


@app.route("/get_plot", methods=["POST"])
def get_plot():
    global df_global
    if df_global is None:
        return jsonify({"error": "No dataset loaded."})

    plot_type = request.form.get("plot_type", "")
    col       = request.form.get("column",    "")
    col_x     = request.form.get("col_x",     "")
    col_y     = request.form.get("col_y",     "")
    col_cat   = request.form.get("col_cat",   "")
    col_num   = request.form.get("col_num",   "")

    try:
        if   plot_type == "histogram": fig = plot_histogram(df_global, col)
        elif plot_type == "bar":       fig = plot_bar(df_global, col)
        elif plot_type == "heatmap":   fig = plot_heatmap(df_global)
        elif plot_type == "scatter":   fig = plot_scatter(df_global, col_x, col_y)
        elif plot_type == "boxplot":   fig = plot_boxplot(df_global, col_cat, col_num)
        else: return jsonify({"error": "Unknown plot type."})

        return jsonify({"image": _fig_to_b64(fig)})

    except Exception as e:
        import traceback
        return jsonify({"error": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True)