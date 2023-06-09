from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--test_ratio", type=float, default=0.1)
parser.add_argument("--out_dir", type=str, default='./data/processed')
args, _ = parser.parse_known_args()
logger.info("Received arguments {}".format(args))

logger.info("Reading data from {}".format(args.data_path))
data = pd.read_csv(args.data_path)
data = data.sort_values('month').reset_index(drop=True)
columns = [
    "town",
    "flat_type",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
]
data = data[columns]

logger.debug("Cleaning Data.")
data = data.replace(regex=[r".*[mM]aisonette.*", "foo"], value="Maisonette")

logger.debug("Extracting number from remaining_lease.")
data["remaining_lease"] = data["remaining_lease"].str.extract(
    r"(\d+)(?= years)")
data = data.astype({"remaining_lease": "int16"})

logger.debug("Label encoding categorical columns.")
cat = data["storey_range"].astype("category")
data["storey_range"] = cat.cat.codes
storey_range_map = dict(enumerate(cat.cat.categories))

flat_type_map = {
    "1 ROOM": 0,
    "2 ROOM": 1,
    "3 ROOM": 2,
    "4 ROOM": 3,
    "5 ROOM": 4,
    "MULTI-GENERATION": 5,
    "EXECUTIVE": 6,
}
data = data.replace({"flat_type": flat_type_map})

logger.debug("One-hot encoding categorical columns.")
data = pd.get_dummies(
    data,
    columns=["town"],
    prefix=["town"],
    dtype=int,
    drop_first=True,
)  # central is baseline
data = pd.get_dummies(data, columns=["flat_model"], prefix=[
                      "model"], dtype='int8')
# remove standard, setting it as the baseline
data = data.drop("model_Standard", axis=1)

data_processed = data.copy()

# Split into train, val, test
y = data_processed["resale_price"]
X = data_processed.drop(["resale_price"], axis=1)

TRAIN_RATIO = args.train_ratio
VAL_RATIO = args.val_ratio
TEST_RATIO = args.test_ratio

logger.debug("Splitting data into train, validation, and test sets")
num_samples = len(data_processed)
train_num = round(TRAIN_RATIO * num_samples)
X_train, y_train = X[:train_num], y[:train_num]
X_val_test, y_val_test = X[train_num:], y[train_num:]
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test,
    y_val_test,
    test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)),
    random_state=2023,
)

train_df = pd.concat([y_train, X_train], axis=1)
val_df = pd.concat([y_val, X_val], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)
dataset_df = pd.concat([y, X], axis=1)

logger.info("Train data shape after preprocessing: {}".format(train_df.shape))
logger.info("Validation data shape after preprocessing: {}".format(val_df.shape))
logger.info("Test data shape after preprocessing: {}".format(test_df.shape))

# Save processed datasets to the local paths
train_output_path = os.path.join(f"{args.out_dir}/train", "train.csv")
val_output_path = os.path.join(f"{args.out_dir}/val", "val.csv")
test_output_path = os.path.join(f"{args.out_dir}/test", "test.csv")

logger.info("Saving train data to {}".format(train_output_path))
train_df.to_csv(train_output_path, index=False)
logger.info("Saving validation data to {}".format(val_output_path))
val_df.to_csv(val_output_path, index=False)
logger.info("Saving test data to {}".format(test_output_path))
test_df.to_csv(test_output_path, index=False)

# python .\scripts\preprocessing.py --data_path ./data/resale-flat-prices-2022-jan.csv -train_ratio 0.8 -val-ratio 0.1 -test-ratio 0.1 -out_dir ./data/processed
