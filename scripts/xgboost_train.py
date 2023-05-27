from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# Hyperparameters and algorithm parameters are described here
parser.add_argument("--num_round", type=int, default=100) # CHANGE TO 100
parser.add_argument("--max_depth", type=int, default=3)
parser.add_argument("--eta", type=float, default=0.2)
parser.add_argument("--subsample", type=float, default=0.9)
parser.add_argument("--colsample_bytree", type=float, default=0.8)
parser.add_argument("--objective", type=str, default="reg:squarederror")
parser.add_argument("--eval_metric", type=str, default="mae")
parser.add_argument("--nfold", type=int, default=3)
parser.add_argument("--early_stopping_rounds", type=int, default=3)

# Set location of input training data
parser.add_argument("--train_data_dir", type=str)
# Set location of input validation data
parser.add_argument("--val_data_dir", type=str)
# Set location where trained model will be stored
parser.add_argument("--model_dir", type=str)
# Set location where model artifacts will be stored
parser.add_argument("--output_artifacts_dir", type=str)

args = parser.parse_args()
print(args)

data_train = pd.read_csv(f"{args.train_data_dir}/train.csv")
train = data_train.drop("resale_price", axis=1)
label_train = pd.DataFrame(data_train["resale_price"])
dtrain = xgb.DMatrix(train, label=label_train)

data_val = pd.read_csv(f"{args.val_data_dir}/val.csv")
val = data_val.drop("resale_price", axis=1)
label_val = pd.DataFrame(data_val["resale_price"])
dval = xgb.DMatrix(val, label=label_val)

# Choose XGBoost model hyperparameters
params = {
    "max_depth": args.max_depth,
    "eta": args.eta,
    "objective": args.objective,
    "subsample" : args.subsample,
    "colsample_bytree":args.colsample_bytree
}

num_boost_round = args.num_round
nfold = args.nfold
early_stopping_rounds = args.early_stopping_rounds

# Cross-validate train XGBoost model
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=nfold,
    early_stopping_rounds=early_stopping_rounds,
    metrics=["mae"],
    seed=2023,
)

model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))
train_pred = model.predict(dtrain)
val_pred = model.predict(dval)

train_mae = mean_absolute_error(label_train, train_pred)
val_mae = mean_absolute_error(label_val, val_pred)
print(f"train-mae: {train_mae:.2f}")
print(f"validation-mae: {val_mae:.2f}")

metrics_data = {
    "hyperparameters": params,
    "reg_metrics": {
        "validation:auc": {"value": val_mae},
        "train:auc": {"value": train_mae}
    }
}

# Save the evaluation metrics to the location specified by output_data_dir
metrics_location = args.output_artifacts_dir + "/metrics.json"
print(metrics_data)
# with open(metrics_location, "w") as f:
#     json.dump(metrics_data, f)

# Save the trained model to the location specified by model_dir
model_location = args.model_dir + "/xgboost-model"
# with open(model_location, "wb") as f:
#     joblib.dump(model, f)
print(model_location)