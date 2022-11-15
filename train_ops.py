"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import csv


def log_training_events(array, file_name: str) -> None:
    with open(file_name, "a") as fp:
        csv.writer(fp, delimiter=",").writerow(array)
