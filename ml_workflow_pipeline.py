import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from mlflow.models.signature import infer_signature
from concurrent.futures import ThreadPoolExecutor

# Load dataset
df = spark.table("e2e.default.visitor_transactions_gold")
train_df, test_df = df.randomSplit([0.8, 0.2])

feature_cols = [
    "visitor_days_since_last_purchase",
    "visitor_total_events",
    "visitor_purchase_count",
    "visitor_conversion_rate",
    "session_length"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")


def run_linear_regression(train_df, test_df, parent_run_id):
    with mlflow.start_run(run_name="linear_regression_model", nested=True, parent_run_id=parent_run_id):
        lr = LinearRegression(
            featuresCol="features",
            labelCol="unique_items_basket_size",
            predictionCol="unique_items_basket_size_pred"
        )
        pipeline = Pipeline(stages=[assembler, scaler, lr])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        # Metrics
        rmse = RegressionEvaluator(
            labelCol="unique_items_basket_size", 
            predictionCol="unique_items_basket_size_pred", 
            metricName="rmse"
        ).evaluate(predictions)

        r2 = RegressionEvaluator(
            labelCol="unique_items_basket_size", 
            predictionCol="unique_items_basket_size_pred", 
            metricName="r2"
        ).evaluate(predictions)

        print(f"[LinearRegression] RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Log model + metrics
        signature = infer_signature(model.transform(train_df.limit(1)), predictions.limit(1))
        mlflow.spark.log_model(model, artifact_path="linear_regression_model", signature=signature)
        mlflow.log_metrics({"lr_rmse": rmse, "lr_r2": r2})

        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/linear_regression_model",
            name="unique_items_basket_size_model"
        )


def run_logistic_regression(train_df, test_df, parent_run_id):
    with mlflow.start_run(run_name="logistic_regression_model", nested=True, parent_run_id=parent_run_id):
        logr = LogisticRegression(
            featuresCol="features",
            labelCol="visitor_item_purchase_label",
            predictionCol="visitor_item_purchase_label_pred",
            probabilityCol="probability"
        )
        pipeline = Pipeline(stages=[assembler, scaler, logr])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="visitor_item_purchase_label", 
            predictionCol="visitor_item_purchase_label_pred"
        )

        metrics = {
            "logr_accuracy": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
            "logr_precision": evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
            "logr_recall": evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"}),
            "logr_f1": evaluator.evaluate(predictions, {evaluator.metricName: "weightedFMeasure"}),
        }

        print(f"[LogisticRegression] {metrics}")

        # Log model + metrics
        signature = infer_signature(model.transform(train_df.limit(1)), predictions.limit(1))
        mlflow.spark.log_model(model, artifact_path="logistic_regression_model", signature=signature)
        mlflow.log_metrics(metrics)

        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/logistic_regression_model",
            name="visitor_item_purchase_prediction_model"
        )


# Run both models  under one parent run
with mlflow.start_run(run_name="basket_and_purchase_models") as parent_run:
    parent_run_id = parent_run.info.run_id
    
    run_linear_regression(train_df, test_df, parent_run_id)
    run_logistic_regression(train_df, test_df, parent_run_id)

    