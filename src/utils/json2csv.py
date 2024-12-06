# script to turn a json file into a csv
# Usage: python json2csv.py <input.json> <output.csv>

import json
import csv
import sys

def json2csv(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    with open(output_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data[0].keys())
        for row in data:
            csv_writer.writerow(row.values())

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    json2csv(input_file, output_file)


