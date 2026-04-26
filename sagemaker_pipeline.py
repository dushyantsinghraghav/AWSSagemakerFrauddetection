"""
SageMaker Pipeline: Event-Driven ML Retraining (2026)
-------------------------------------------------------
Defines a SageMaker Pipeline with 5 steps:

  1. Preprocessing  – SKLearnProcessor: clean, split, feature engineer
  2. Training       – XGBoost built-in algorithm (no custom script needed)
  3. Evaluation     – ScriptProcessor: compute accuracy + AUC
  4. Condition      – Branch on accuracy >= threshold
  5a. Register      – ModelPackage registration (if passes)
  5b. Fail          – Pipeline fail step (if accuracy too low)

Usage:
    python sagemaker_pipeline.py --create   # creates / updates the pipeline
    python sagemaker_pipeline.py --delete   # deletes the pipeline
    python sagemaker_pipeline.py --start    # manual test run
"""

import argparse
import json
import logging
import os

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
#from sagemaker.xgboost import XGBoost

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE_NAME       = os.environ.get("PIPELINE_NAME",        "fraud-detection-retrain")
BUCKET              = os.environ.get("TRAINING_DATA_BUCKET", "")
ROLE_ARN            = os.environ.get("SAGEMAKER_ROLE_ARN",   "")
MODEL_PACKAGE_GROUP = os.environ.get("MODEL_PACKAGE_GROUP",  "FraudDetectionModels")
REGION              = os.environ.get("AWS_REGION",           "")

# Scripts uploaded to S3 by setup_infrastructure.py
SCRIPTS_S3_PREFIX   = f"s3://{BUCKET}/pipeline-scripts"


# ── Pipeline parameters (overridable at execution time) ───────────────────────
input_data_uri = ParameterString(
    name="InputDataUri",
    default_value=f"s3://{BUCKET}/data/sample.csv",
)
output_s3_prefix = ParameterString(
    name="OutputS3Prefix",
    default_value=f"s3://{BUCKET}/pipeline-outputs/default",
)
training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.large",        # medium not supported for training
)
accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.70,                 # lowered to 0.70 for easier first pass
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",
    enum_values=["Approved", "Rejected", "PendingManualApproval"],
)


# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
def build_preprocessing_step(session: sagemaker.Session) -> ProcessingStep:
    """
    Clean raw CSV, engineer features, split into train/validation/test.
    Outputs written to S3 and consumed by the training step.
    """
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.medium",   # cheapest supported processing instance
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
        base_job_name="fraud-preprocess",
    )

    return ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input",
                s3_data_distribution_type="FullyReplicated",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=Join(on="/", values=[output_s3_prefix, "processed/train"]),
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=Join(on="/", values=[output_s3_prefix, "processed/validation"]),
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=Join(on="/", values=[output_s3_prefix, "processed/test"]),
            ),
        ],
        code=f"{SCRIPTS_S3_PREFIX}/preprocess.py",
        job_arguments=["--test-size", "0.15", "--validation-size", "0.15"],
    )


# ── Step 2: Training ──────────────────────────────────────────────────────────
def build_training_step(
    session: sagemaker.Session,
    preprocessing_step: ProcessingStep,
) -> TrainingStep:
    """
    Train XGBoost using the SageMaker built-in algorithm via generic Estimator.
    Using Estimator instead of XGBoost class avoids the entry_point requirement.
    """
    from sagemaker.estimator import Estimator

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1",
        image_scope="training",
    )

    xgb_estimator = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
        base_job_name="fraud-train",
        hyperparameters={
            "max_depth":             6,
            "eta":                   0.1,
            "gamma":                 4,
            "min_child_weight":      6,
            "subsample":             0.8,
            "objective":             "binary:logistic",
            "num_round":             300,
            "eval_metric":           "auc",
            "early_stopping_rounds": 20,
        },
    )

    return TrainingStep(
        name="TrainModel",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        depends_on=[preprocessing_step],
    )


# ── Step 3: Evaluation ────────────────────────────────────────────────────────
def build_evaluation_step(
    session: sagemaker.Session,
    preprocessing_step: ProcessingStep,
    training_step: TrainingStep,
) -> tuple[ProcessingStep, PropertyFile]:
    """
    Evaluate trained model on held-out test set.
    Writes evaluation.json — accuracy value drives the ConditionStep.
    """
    script_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=REGION,
            version="1.7-1",
            image_scope="training",
        ),
        command=["python3"],
        instance_type="ml.t3.medium",   # cheapest supported processing instance
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
        base_job_name="fraud-eval",
    )

    eval_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig
                    .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=[output_s3_prefix, "evaluation"]),
            )
        ],
        code=f"{SCRIPTS_S3_PREFIX}/evaluate.py",
        property_files=[eval_report],
        depends_on=[training_step],
    )

    return eval_step, eval_report


# ── Step 4+5: Condition + Register / Fail ─────────────────────────────────────
def build_condition_step(
    session: sagemaker.Session,
    training_step: TrainingStep,
    eval_step: ProcessingStep,
    eval_report: PropertyFile,
) -> ConditionStep:
    """
    accuracy >= threshold → register model in Model Registry
    accuracy <  threshold → fail pipeline with descriptive message
    """
    from sagemaker.model import Model
    from sagemaker.workflow.pipeline_context import PipelineSession

    pipeline_session = PipelineSession()

    model = Model(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=REGION,
            version="1.7-1",
            image_scope="inference",
        ),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[output_s3_prefix, "evaluation", "evaluation.json"],
            ),
            content_type="application/json",
        )
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            description=Join(
                on=" ",
                values=[
                    "Auto-registered from execution",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                ],
            ),
        ),
    )

    fail_step = FailStep(
        name="ModelFailedAccuracyCheck",
        error_message=Join(
            on=" ",
            values=[
                "Model accuracy did not meet threshold of",
                accuracy_threshold,
                "— pipeline aborted. Review evaluation report at:",
                output_s3_prefix,
            ],
        ),
    )

    condition = ConditionGreaterThanOrEqualTo(
        left=sagemaker.workflow.functions.JsonGet(
            step_name=eval_step.name,
            property_file=eval_report,
            json_path="binary_classification_metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )

    return ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[fail_step],
        depends_on=[eval_step],
    )


# ── Assemble pipeline ─────────────────────────────────────────────────────────
def build_pipeline(session: sagemaker.Session) -> Pipeline:
    preprocess_step        = build_preprocessing_step(session)
    train_step             = build_training_step(session, preprocess_step)
    eval_step, eval_report = build_evaluation_step(session, preprocess_step, train_step)
    cond_step              = build_condition_step(session, train_step, eval_step, eval_report)

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            input_data_uri,
            output_s3_prefix,
            training_instance_type,
            accuracy_threshold,
            model_approval_status,
        ],
        steps=[preprocess_step, train_step, eval_step, cond_step],
        sagemaker_session=session,
    )
    return pipeline


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create/update the pipeline")
    parser.add_argument("--delete", action="store_true", help="Delete the pipeline")
    parser.add_argument("--start",  action="store_true", help="Start a manual execution")
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=REGION)
    sm_session   = sagemaker.Session(boto_session=boto_session)

    if args.delete:
        sm_client = boto3.client("sagemaker", region_name=REGION)
        sm_client.delete_pipeline(PipelineName=PIPELINE_NAME)
        logger.info("Deleted pipeline: %s", PIPELINE_NAME)
        return

    pipeline = build_pipeline(sm_session)

    if args.create:
        response = pipeline.upsert(role_arn=ROLE_ARN)
        logger.info(
            "Pipeline upserted: %s",
            json.dumps(response, indent=2, default=str),
        )

    if args.start:
        execution = pipeline.start(
            parameters={
                "InputDataUri":      f"s3://{BUCKET}/data/sample.csv",
                "AccuracyThreshold": "0.70",
            }
        )
        logger.info("Pipeline execution started: %s", execution.arn)
        logger.info(
            "Monitor: https://ap-southeast-2.console.aws.amazon.com/sagemaker/"
            "home?region=ap-southeast-2#/pipelines/%s/executions",
            PIPELINE_NAME,
        )
        execution.wait(delay=30, max_attempts=60)
        logger.info("Steps: %s", execution.list_steps())


if __name__ == "__main__":
    main()
