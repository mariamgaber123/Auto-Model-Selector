from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io, base64
from preprocessing.pipeline import build_preprocessor, split_data
from main import full_pipeline
from mainSmote import full_pipeline_S

try:
    from plots.plot import plot_histogram, plot_bar, plot_heatmap, plot_scatter, plot_boxplot, plot_stacked_bar, plot_pie
except ImportError:
    from plots import plot_histogram, plot_bar, plot_heatmap, plot_scatter, plot_boxplot, plot_stacked_bar, plot_pie

app = Flask(__name__)

df_global = None
trained_model_global = None
feature_columns_global = None
model_name_global = None
target_column_global = None
log_transformed_global = False   # ← NEW: track if target was log-transformed
categorical_values = {}


@app.route("/", methods=["GET", "POST"])
def home():
    global df_global

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", error="Please upload a CSV file.")

        try:
            df = pd.read_csv(file)

            if df.empty:
                return render_template("index.html", error="The uploaded file is empty.")

            df_global = df

        except Exception as e:
            return render_template("index.html", error=str(e))

    if df_global is not None:
        return render_template(
            "index.html",
            tables=[df_global.head(10).to_html(classes="preview-table", index=False)],
            columns=df_global.columns.tolist(),
            row_count=len(df_global),
            col_count=len(df_global.columns),
        )

    return render_template("index.html")


@app.route("/select_target", methods=["POST"])
def select_target():
    global categorical_values, df_global, target_column_global

    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")

    target_column = request.form["target_column"]
    target_column_global = target_column
    categorical_values = {
        col: df_global[col].dropna().unique().tolist()
        for col in df_global.columns
        if df_global[col].dtype == "object"
    }

    return render_template("model.html", target_column=target_column)


@app.route("/train_model", methods=["POST"])
def train_model_route():
    global df_global, trained_model_global, feature_columns_global
    global categorical_values, model_name_global, log_transformed_global

    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")

    try:
        target_column = request.form["target_column"]
        model_name = request.form["model_name"]
        model_name_global = model_name

        params = {}

        def flt(k):
            v = request.form.get(k, "").strip()
            return float(v) if v else None

        def Int(k):
            v = request.form.get(k, "").strip()
            return int(v) if v else None

        def Str(k):
            v = request.form.get(k, "").strip()
            return v if v else None

        if model_name == "Logistic Regression":
            c_value = flt("C")
            if c_value is not None:
                params["C"] = c_value
            max_iter = Int("max_iter")
            if max_iter is not None:
                params["max_iter"] = max_iter
            solver = Str("solver")
            if solver:
                params["solver"] = solver

        elif model_name == "SVM":
            c_value = flt("C")
            if c_value is not None:
                params["C"] = c_value
            kernel = Str("kernel")
            if kernel:
                params["kernel"] = kernel
            gamma = Str("gamma")
            if gamma:
                params["gamma"] = gamma

        elif model_name == "KNN":
            n_neighbors = Int("n_neighbors")
            if n_neighbors is not None:
                params["n_neighbors"] = n_neighbors
            weights = Str("weights")
            if weights:
                params["weights"] = weights
            metric = Str("metric")
            if metric:
                params["metric"] = metric

        elif model_name == "Random Forest":
            n_estimators = Int("n_estimators")
            if n_estimators is not None:
                params["n_estimators"] = n_estimators
            max_depth = Int("max_depth")
            if max_depth is not None:
                params["max_depth"] = max_depth
            min_samples_split = Int("min_samples_split")
            if min_samples_split is not None:
                params["min_samples_split"] = min_samples_split

        elif model_name == "Neural Network (MLP)":
            params["activation"] = Str("activation")
            params["solver"] = Str("solver")
            params["learning_rate"] = Str("learning_rate")
            
            max_iter = Int("max_iter")
            if max_iter is not None:
                params["max_iter"] = max_iter
            
            hidden_layers = Str("hidden_layer_sizes")
            if hidden_layers:
                params["hidden_layer_sizes"] = hidden_layers        

        elif model_name == "Linear Regression":
            params["fit_intercept"] = (
                request.form.get("fit_intercept", "true").lower() == "true"
            )

        use_smote = request.form.get("use_smote", "false") == "true"

        if use_smote:
            trained_model, metrics, feature_columns = full_pipeline_S(
                df_global, target_column, model_name, params
            )
            log_transformed_global = False
        else:
            trained_model, metrics, feature_columns = full_pipeline(
                df_global, target_column, model_name, params
            )
            # Detect if log transform was applied (regression + positive target)
            y = df_global[target_column]
            import pandas as pd2
            from main import detect_problem_type
            log_transformed_global = (
                detect_problem_type(y) == "regression" and (y > 0).all()
            )

        trained_model_global = trained_model
        feature_columns_global = list(feature_columns)

        feature_categorical = {
            c: v
            for c, v in categorical_values.items()
            if c in feature_columns_global
        }

        return render_template(
            "results.html",
            model_name=model_name,
            metrics={k: round(float(v), 5) for k, v in metrics.items()},
            feature_columns=feature_columns_global,
            categorical_values=feature_categorical,
            params=params,
        )

    except ValueError as ve:
        # Show the specific validation error (wrong model type, etc.)
        return render_template(
            "model.html",
            error=str(ve),
            target_column=request.form.get("target_column"),
        )
    except Exception:
        return render_template(
            "model.html",
            error="Training failed. Please check your data or try a different model.",
            target_column=request.form.get("target_column"),
        )


@app.route("/predict", methods=["POST"])
def predict_route():
    global trained_model_global, feature_columns_global, model_name_global, log_transformed_global

    if trained_model_global is None:
        return render_template("model.html", error="No trained model found.")

    try:
        # Better input handling
        input_dict = {}
        for col in feature_columns_global:
            value = request.form.get(col, "0")
            try:
                input_dict[col] = float(value)
            except ValueError:
                input_dict[col] = value  # keep as string for categorical

        df_input = pd.DataFrame([input_dict])

        # Make sure columns order is correct
        df_input = df_input[feature_columns_global]

        prediction = trained_model_global.predict(df_input)

        if log_transformed_global:
            prediction = np.expm1(prediction)

        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]

        if isinstance(prediction, (float, np.floating)):
            prediction = round(float(prediction), 4)

        # Probability
        probability = 0.0
        if hasattr(trained_model_global, "predict_proba"):
            try:
                proba = trained_model_global.predict_proba(df_input)[0]
                probability = "{:.2f}".format(max(proba) * 100)
            except:
                pass

        return render_template(
            "prediction.html",
            prediction=prediction,
            probability=probability,
            model_name=model_name_global,
        )

    except Exception as e:
        return render_template("model.html", error=f"Prediction error: {str(e)}")


@app.route("/model_page", methods=["GET"])
def model_page():
    global target_column_global, df_global

    if df_global is None:
        return render_template("index.html", error="Please upload data first.")

    if target_column_global is None:
        return render_template("index.html", columns=df_global.columns.tolist())

    return render_template("model.html", target_column=target_column_global)


@app.route("/visualize")
def visualize():
    global df_global

    if df_global is None:
        return render_template("index.html", error="No dataset loaded.")

    return render_template(
        "visualize.html",
        numeric_cols=df_global.select_dtypes(include=["number"]).columns.tolist(),
        categorical_cols=df_global.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
    )


def _fig_to_b64(fig):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#060b14")
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
    col = request.form.get("column", "")
    col_x = request.form.get("col_x", "")
    col_y = request.form.get("col_y", "")
    col_cat = request.form.get("col_cat", "")
    col_num = request.form.get("col_num", "")

    try:
        if plot_type == "histogram":
            fig = plot_histogram(df_global, col)
        elif plot_type == "bar":
            fig = plot_bar(df_global, col)
        elif plot_type == "heatmap":
            fig = plot_heatmap(df_global)
        elif plot_type == "pie":
            fig = plot_pie(df_global, col)
        elif plot_type == "cat_analysis":
            c1 = request.form.get("col1")
            c2 = request.form.get("col2")
            fig = plot_stacked_bar(df_global, c1, c2)
        elif plot_type == "scatter":
            fig = plot_scatter(df_global, col_x, col_y)
        elif plot_type == "boxplot":
            fig = plot_boxplot(df_global, col_cat, col_num)
        else:
            return jsonify({"error": "Unknown plot type."})

        return jsonify({"image": _fig_to_b64(fig)})

    except Exception:
        return jsonify({"error": "Plot generation failed."})


if __name__ == "__main__":
    app.run(debug=True)