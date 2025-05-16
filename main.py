from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Transformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
import pandas as pd 

spark = SparkSession.builder.getOrCreate()

def Dropper(cols):
    class _Dropper(Transformer):
        def _transform(self, dataset: DataFrame) -> DataFrame:
            return dataset.drop(*cols)
    return _Dropper()

def get_metrics(spark_df, model, target_col):
    # Evaluators needed for binary classification
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, metricName='accuracy')
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=target_col, metricName='weightedPrecision')
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=target_col, metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_col, metricName='f1')

    # Compute the evaluation metrics
    accuracy = evaluator_accuracy.evaluate(model.transform(spark_df))
    f1 = evaluator_f1.evaluate(model.transform(spark_df))
    precision = evaluator_precision.evaluate(model.transform(spark_df))
    recall = evaluator_recall.evaluate(model.transform(spark_df))

    # Print the metrics
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("-----------------------------------------")

def examine_model(spark_df, model, target_col, num_folds=5):
    # Build a parameter grid for tuning hyperparameters 
    param_grid = ParamGridBuilder().build()
    
    # Cross-Validation setup for binary classification
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, metricName='accuracy')

    # Since at least one metric is required, we will use evaluator_accuracy for CV and compute the others after
    crossval = CrossValidator(estimator=model,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator_accuracy,  
                                numFolds=num_folds)

    # Fit the model with cross-validation
    cv_model = crossval.fit(spark_df)

    # Get the best model
    best_model = cv_model.bestModel
    get_metrics(spark_df, best_model, target_col)


def show_spark_df(spark_df):
    pandas_df = spark_df.toPandas()
    print(pandas_df.info())
    print(pandas_df.describe())

def preprocessing_pipeline(spark_df, target_col):
    # Extract input features 
    feature_columns = [col for col in spark_df.columns if col not in [target_col, 'id']]

    # Prepare preprocessing piepline
    dropper = Dropper(['id'])
    drop_originals = Dropper(feature_columns)

    # Encode target column into numerical format so ML models can use its values
    indexer = StringIndexer(inputCol=target_col, outputCol=f'{target_col}_results')

    # Use those columns in VectorAssembler
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='input_features')

    # Build the preprocessing pipeline 
    pipeline = Pipeline(stages=[dropper, indexer, assembler, drop_originals])
    return pipeline

def train_test_split(spark_df, train_ratio=0.8, seed=42):
    # Split the DataFrame into train and test sets based on the specified ratio
    train_df, test_df = spark_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return (train_df, test_df)

def main():
    panda_df = pd.read_csv('data.csv')
    spark_df = spark.createDataFrame(panda_df)

    # Split into training and test sets using 20% for test, 80% for training 
    spark_df_train, spark_df_test = train_test_split(spark_df)

    # Extract target 'diagnosis' for classification
    target_col = 'diagnosis'
    pipeline = preprocessing_pipeline(spark_df_train, target_col)
    pipeline_fitted = pipeline.fit(spark_df_train)
    spark_df_train_X = pipeline_fitted.transform(spark_df_train)

    # Drop original (unprocessed) target column 
    spark_df_train_X = spark_df_train_X.drop(target_col)

    # show_spark_df(spark_df_preproceesed)
    target_col_preprocessed = 'diagnosis_results'
    # Train a RandomForestClassifier
    # rf = RandomForestClassifier(
    #     featuresCol='input_features', 
    #     labelCol=target_col_preprocessed,
    #     seed=42 # For reproducibility
    # )
    # print("Printing metric results for RandomForestClassifier with cross-validation")
    # examine_model(spark_df_train_X, rf, target_col_preprocessed, 5)

    # Train a LogisticRegression Classifier (does not use a seed)
    lr = LogisticRegression(
        featuresCol='input_features', 
        labelCol=target_col_preprocessed,  # Adjust target column name as needed
    )
    examine_model(spark_df_train_X, lr, target_col_preprocessed, 5)


    # Evaluation on the test set 
    spark_df_test_X = pipeline_fitted.transform(spark_df_test)
    spark_df_test_X = spark_df_test_X.drop(target_col)

    # rf_fitted = rf.fit(spark_df_train_X)
    # print("Printing metric results for RandomForestClassifier on the test set")
    # get_metrics(spark_df_test_X, rf_fitted, target_col_preprocessed)

    lr_fitted = lr.fit(spark_df_train_X)
    print("Printing metric results for LogisticRegression on the test set")
    get_metrics(spark_df_test_X, lr_fitted, target_col_preprocessed)


if __name__ == '__main__':
    main()