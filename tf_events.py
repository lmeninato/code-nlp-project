from collections import defaultdict
import pandas as pd
import tensorflow as tf


def is_interesting_tag(tag):
    if "val" in tag or "train" in tag:
        return True
    else:
        return False


def parse_events_file(path: str) -> pd.DataFrame:
    metrics = defaultdict(list)
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            print(v)
            if isinstance(v.simple_value, float) and is_interesting_tag(v.tag):
                metrics[v.tag].append(v.simple_value)
            if v.tag == "loss" or v.tag == "accuracy":
                print(v.simple_value)
    metrics_df = pd.DataFrame({k: v for k, v in metrics.items() if len(v) > 1})
    return metrics_df
