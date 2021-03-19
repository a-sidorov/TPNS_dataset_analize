#!python

import pprint
import statistics
from math import isnan, log, sqrt

import numpy as np
import pandas as p
import seaborn as sns
from matplotlib import pyplot as plt

import csv
from const import CLEAN_DATA_FILE_NAME

pp = pprint.PrettyPrinter(indent=4)


def has_value(cell: str) -> bool:
    return cell and cell not in {'-', '#ЗНАЧ!', 'NaN', 'не спускался'}


def parse_data(data: list[list]) -> list[list]:
    def parse_cell(cell: str) -> any:
        if not has_value(cell):
            return float('nan')

        try:
            return float(cell.replace(',', '.'))
        except ValueError:
            return cell

    get_kgf: callable = lambda a, b: a if isnan(b) else b * 1000

    data = [list(map(parse_cell, row)) for row in data]
    data = [row[:-2] + [get_kgf(row[-2], row[-1])] for row in data]
    data = list(filter(lambda row: not (isnan(row[-1]) and isnan(row[-2])), data))

    return data


def get_classes(headers: list, data: list[list]) -> dict[str, list]:
    return {
        headers[-2]: [row[-2] for row in data if not isnan(row[-2])],
        headers[-1]: [row[-1] for row in data if not isnan(row[-1])]
    }


def plot_histogram_sns(data: list[float], name: str = "") -> None:
    np_dataset = np.array(data).astype(float)
    n = 1 + log(len(data), 2)

    if np_dataset.var() < 1e-5:
        print(name + ' has 0 variance, can not define distribution')
        sns.displot(np_dataset, bins=int(n))
        plt.title(f'{name}')
        plt.savefig(name + '.jpg')
        return

    sns.displot(np_dataset, kde=True, bins=int(n))
    plt.title(f'{name}')
    plt.savefig(name + '.jpg')


def analyze_attribute(values: list) -> dict:
    values = sorted(values)
    result: dict[str, [float, int, str]] = {}

    actual_values = [it for it in values if not isnan(it)]
    actual_values_count = len(actual_values)
    result['total'] = actual_values_count

    empty_percentage = (1 - len(actual_values) / len(values)) * 100
    result['empty %'] = empty_percentage

    unique_values = set(actual_values)
    unique_values_count = len(unique_values)
    result['unique'] = unique_values_count

    def analyze_continuous():
        result['type'] = 'continuous'
        result['min'] = min(actual_values)
        result['mean'] = np.mean(actual_values).astype(float)
        result['median'] = statistics.median(actual_values)
        result['max'] = max(actual_values)
        result['standard deviation'] = sqrt(np.var(actual_values))

        return result

    def analyze_categorical():
        result['type'] = 'categorical'

        mode = statistics.multimode(actual_values)
        m1 = mode[0]
        result['mode 1'] = m1

        result['mode 1 %'] = actual_values.count(m1) / len(actual_values)

        if len(mode) > 1:
            m2 = mode[1]

            result['mode 2'] = m2
            result['mode 2 %'] = actual_values.count(m2) / len(actual_values)

        return result

    # return analyze_continuous()

    if unique_values_count < actual_values_count * 0.8:
        return analyze_categorical()
    else:
        return analyze_continuous()


def impute_data(data: list[list], i: int, analysis_result: dict) -> None:
    if analysis_result['type'] == 'categorical':
        mode = analysis_result['mode 1']
        for row in data:
            if isnan(row[i]):
                row[i] = mode

    if analysis_result['type'] == 'continuous':
        mean = analysis_result['mean']
        for row in data:
            if isnan(row[i]):
                row[i] = mean


def compute_correlation_matrix(data: list[list], headers: list[str]) -> p.DataFrame:
    np_dataset = np.array(data).astype(float)

    data_frame = p.DataFrame(np_dataset, columns=headers)
    correlation_matrix = data_frame.corr(min_periods=1)
    correlation_matrix.values[np.tril_indices(len(correlation_matrix))] = np.nan

    correlation_matrix.to_csv('correlation.csv', sep=';', float_format='%.3f')

    return correlation_matrix


def main() -> None:
    with open(CLEAN_DATA_FILE_NAME, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=';')
        rows = list(reader)

        headers, data = rows[0][0:-1], rows[1:]
        data = parse_data(data)
        data = [row[0:] for row in data]
        print(f'{len(data)} rows of data')

        attribute_info = {}
        to_remove = []

        for i in range(len(headers)):
            values = [row[i] for row in data]
            if all(isnan(it) for it in values):
                continue
            analysis_result = analyze_attribute(values)
            if analysis_result['empty %'] > 60:
                print(f"[warning] {headers[i]}: {analysis_result['empty %']:.2f}% missing data")
                if headers[i] != 'КГФ' and headers[i] != 'G_total':
                    to_remove.append(i)

            if 0 < analysis_result['empty %'] < 30:
                print(
                    f"[can impute] {headers[i]} ({analysis_result['type']}): {analysis_result['empty %']:.2f}% "
                    f"missing data")
                impute_data(data, i, analysis_result)

            attribute_info[i] = analysis_result

        for attribute, info in attribute_info.items():
            print(headers[attribute])
            pp.pprint(info)
            unique_count = info['unique']
            total_count = info['total']

            if unique_count < total_count * 0.15:
                print("МАЛО ЗНОЧЕНИЙ")

        # for i in range(len(headers)):
        #     plot_histogram_sns([row[i] for row in data if not isnan(row[i])], headers[i])

        # to_remove.reverse()
        #
        # for index in to_remove:
        #     del headers[index]
        #     for row in data:
        #         del row[index]

        correlation_matrix = compute_correlation_matrix(data, headers)

        sns.heatmap(correlation_matrix)
        plt.savefig('heatmap.jpg')

        attr_correlated_elements = np.extract(correlation_matrix >= 0.95, correlation_matrix)
        i, j = np.where(correlation_matrix >= 0.95)
        print(attr_correlated_elements)
        print(i, j)

        for k in range(len(i)):
            print(headers[i[k]] + ' ' + headers[j[k]])

        # with open('out.csv', 'w', encoding='UTF-8') as out:
        #     writer = csv.writer(out, delimiter=';', lineterminator='\t\n')
        #     writer.writerow(headers)
        #     writer.writerows(data)
        #
        # with open('correlation.csv', 'w', encoding='UTF-8') as out:
        #     writer = csv.writer(out, delimiter=';', lineterminator='\t\n')
        #     writer.writerow(headers)
        #     writer.writerows(correlation_matrix)


if __name__ == '__main__':
    main()
