#!python
import json
import statistics
from math import isnan, log, sqrt

import numpy as np
import pandas as p
import seaborn as sns
from matplotlib import pyplot as plt

import csv
from const import CLEAN_DATA_FILE_NAME


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


def reverse_enum(arr: np.ndarray):
    for index in reversed(range(len(arr))):
        yield index, arr[index]


def unique_counts(np_arr: np.ndarray) -> (np.ndarray, np.ndarray):
    np_arr_unique, np_arr_counts = np.unique(np_arr, return_counts=True, axis=0)
    if len(np_arr.shape) == 1:
        np_arr_unique = np_arr_unique[~np.isnan(np_arr_unique)]
        actual_length = len(np_arr_unique)
        if len(np_arr_counts) != actual_length:
            np_arr_unique = np.append(np_arr_unique, np.nan)
            isnan_count = np_arr_counts[np_arr_unique.shape[0] - 1:].sum()
            np_arr_counts = np_arr_counts[:np_arr_unique.shape[0]]
            np_arr_counts[np_arr_unique.shape[0] - 1] = isnan_count
    else:
        if np_arr.shape[1] == 2:
            for index, value in reverse_enum(np_arr_unique):
                if index != 0:
                    if np.isnan(value[0]) and np.isnan(np_arr_unique[index - 1][0]) and np_arr_unique[index - 1][1] == \
                            value[1]:
                        np_arr_unique = np.delete(np_arr_unique, index, axis=0)
                        np_arr_counts[index - 1] += np_arr_counts[index]
                        np_arr_counts = np.delete(np_arr_counts, index, axis=0)

    return np_arr_unique, np_arr_counts


def entropy(feature_count: float, domain_size: float) -> float:
    feature_probability = feature_count / domain_size
    return - feature_probability * log(feature_probability, 2)


def gain_ratio(class_: list[list], feature: list[float]) -> float:
    class_ = np.array(class_).astype(float)
    feature = np.array(feature).astype(float)

    info_class = 0
    classes_unique, classes_counts = unique_counts(class_)

    for c_iter in classes_counts:
        info_class += entropy(c_iter, class_.shape[0])

    info_class_by_feature = 0
    split_info = 0
    feat_unique, feat_counts = unique_counts(feature)

    for feat_index in range(len(feat_unique)):
        feature_probability = feat_counts[feat_index] / feature.shape[0]

        if np.isnan(feat_unique[feat_index]):
            # TODO: считать ли NaN за отдельный класс?
            indices = [i for i in range(0, feature.shape[0]) if np.isnan(feature[i])]
        else:
            indices = [i for i in range(0, feature.shape[0]) if feature[i] == feat_unique[feat_index]]

        classes_of_feature = class_[indices]

        classes_of_feature_unique, classes_of_feature_count = unique_counts(classes_of_feature)
        for class_index in range(classes_of_feature_unique.shape[0]):
            class_of_feature_count = classes_of_feature_count[class_index]
            info_class_by_feature += feature_probability * entropy(class_of_feature_count, classes_of_feature.shape[0])
        split_info += - feature_probability * log(feature_probability, 2)

    information_gain = info_class - info_class_by_feature

    return information_gain / split_info


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

    # result['gain'] = 0.0

    def analyze_continuous():
        result['type'] = 'continuous'
        result['min'] = min(actual_values)
        result['mean'] = np.mean(actual_values).astype(float)
        result['median'] = statistics.median(actual_values)
        result['max'] = max(actual_values)
        result['standard deviation'] = sqrt(np.var(actual_values))

        chi_025, chi_075 = np.percentile(actual_values, [25, 75])

        result['chi_25'] = chi_025
        result['chi_75'] = chi_075

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

    correlation_matrix.to_csv('correlation.csv', sep=';', float_format='%.3f')

    correlation_matrix.values[np.tril_indices(len(correlation_matrix))] = np.nan

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

        target = [row[-2:] for row in data]

        for attribute, info in attribute_info.items():
            values = [row[attribute] for row in data]
            info |= {'gain_ratio': gain_ratio(target, values)}

            print(headers[attribute])
            unique_count = info['unique']
            total_count = info['total']

            if unique_count < total_count * 0.15:
                print('few unique values')
            print(json.dumps(info, indent=4), '\n')

        for i in range(len(headers)):
            plot_histogram_sns([row[i] for row in data if not isnan(row[i])], headers[i])

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

        print("====================================================")
        for k in range(len(i)):
            print(headers[i[k]] + ' ' + headers[j[k]] + ' ', round(attr_correlated_elements[k], 2))

        with open('out.csv', 'w', encoding='UTF-8') as out:
            writer = csv.writer(out, delimiter=';', lineterminator='\t\n')
            writer.writerow(headers)
            writer.writerows(data)


if __name__ == '__main__':
    main()
