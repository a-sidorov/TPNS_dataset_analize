#!python

import csv
from const import CLEAN_DATA_FILE_NAME
from math import isnan, log, sqrt
from matplotlib import pyplot as plt
import numpy as np
import pandas as p
import seaborn as sns
from gauss import gauss
import pprint
import statistics

pp = pprint.PrettyPrinter(indent=4)


def has_value(cell: str) -> bool:
    return cell and cell not in {'-', '#ЗНАЧ!', 'NaN', 'не спускался'}


def column(data: list[list[float]], idx: int, filter_not_nan=False) -> list[float]:
    if filter_not_nan:
        return [row[idx] for row in data if not isnan(row[idx])]

    return [row[idx] for row in data]


def parse_data(data: list[list]) -> list[list]:
    def parse_cell(cell: str) -> any:
        if not has_value(cell):
            return float('nan')

        try:
            return float(cell.replace(',', '.'))
        except ValueError:
            return cell

    def get_kgf(a, b):
        return a if isnan(b) else b * 1000

    data = [list(map(parse_cell, row)) for row in data]
    data = [row[:-2] + [get_kgf(row[-2], row[-1])] for row in data]
    data = list(filter(lambda row: not (isnan(row[-1]) and isnan(row[-2])), data))

    return data


def get_classes(headers: list, data: list[list]) -> dict[str, list]:
    return {
        headers[-2]: [row[-2] for row in data if not isnan(row[-2])],
        headers[-1]: [row[-1] for row in data if not isnan(row[-1])]
    }


def plot_histogram(
        col: list[float],
        name: str = '',
        low=None, high=None,
        continuous=False,
        density=False) -> None:
    col = sorted(col)
    n = int(1 + log(len(col), 2))
    arr = plt.hist(col, bins=n, density=density)
    for i in range(n):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])) if not density else f'{arr[0][i]:.4f}')

    title = name

    if continuous:
        mean = float(np.mean(col))  # матожидание
        variance = float(np.var(col))  # дисперсия

        if variance < 1e-5:
            return

        title += f'\nmean = {mean:.2f} variance = {variance:.2f}'

        dist = gauss(mean, variance)
        plt.plot(col, list(map(dist, col)))

    if low and high:
        plt.axvline(x=low)
        plt.axvline(x=high)

    plt.title(title)


def analyze_attribute(values: list[float], treat_as_continuous=False) -> dict:
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
        result['variance'] = sqrt(np.var(actual_values))

        chi_025, chi_075 = np.percentile(actual_values, [25, 75])

        result['chi_025'] = chi_025
        result['chi_075'] = chi_075

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

    if treat_as_continuous:
        return analyze_continuous()

    if unique_values_count < actual_values_count * 0.24:
        return analyze_categorical()
    else:
        return analyze_continuous()


def analyze_attributes(data: list[list[float]], treat_as_continuous=False) -> dict[int, dict]:
    columns = len(data[0])

    return {idx: analyze_attribute(column(data, idx), treat_as_continuous) for idx in range(columns)}


def save_histograms(data: list[list[float]], headers: list[str], attribute_info: dict[int, dict]) -> None:
    for idx, info in attribute_info.items():
        header = headers[idx]

        low, high = None, None
        plot_norm = False

        if info['type'] == 'continuous':
            plot_norm = True

            chi_025 = info['chi_025']
            chi_075 = info['chi_075']

            low = chi_025 - 1.5 * (chi_075 - chi_025)
            high = chi_075 + 1.5 * (chi_075 - chi_025)

        plt.clf()
        plot_histogram(column(data, idx, filter_not_nan=True), f'{header} [{info["type"]}]', low, high, plot_norm, True)
        plt.savefig(f'img/{header}.png')


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


def remove_outliers(data: list[list[float]], idx: int, low: float, high: float) -> list[list[float]]:
    def is_outlier(value):
        return not isnan(value) and (value < low or value > high)

    result = []

    for row in data:
        val = row[idx]

        if is_outlier(val) and idx == KGF_IDX:
            continue

        if not is_outlier(val) or not isnan(row[-2]):
            result.append(row)

    return result


def has_outliers(col: list[float], low: float, high: float) -> bool:
    return not all(low <= item <= high or isnan(item) for item in col)


def save_outliers(
        data: list[list[float]],
        headers: list[str],
        attribute_info: dict[int, dict]):
    for idx, header in enumerate(headers):
        if idx not in attribute_info:
            continue

        chi_025 = attribute_info[idx]['chi_025']
        chi_075 = attribute_info[idx]['chi_075']

        low = chi_025 - 1.5 * (chi_075 - chi_025)
        high = chi_075 + 1.5 * (chi_075 - chi_025)

        if has_outliers(column(data, idx), low, high):
            plt.clf()
            plot_histogram(
                column(data, idx, filter_not_nan=True),
                f'{header}\nlow = {low:.2f} high = {high:.2f}',
                low, high
            )
            plt.savefig(f'img/outliers/{header}.png')


def correlation_matrix(data: list[list], headers: list[str]) -> p.DataFrame:
    np_dataset = np.array(data).astype(float)

    data_frame = p.DataFrame(np_dataset, columns=headers)
    matrix = data_frame.corr()

    matrix.to_csv('csv/correlation.csv', sep=';', float_format='%.3f')

    matrix.values[np.tril_indices(len(matrix))] = np.nan

    return matrix


CORR_THRESHOLD = 0.95
CORR_DIFF_THRESHOLD = 0.3


def show_heatmap(corr_mat: p.DataFrame, headers: list[str]) -> None:
    sns.heatmap(corr_mat, xticklabels=range(len(headers)), yticklabels=1, linewidths=.5)
    plt.show()


def inspect_correlations(data: list[list[float]], headers: list[str], target: list[int]) -> list[tuple[str, str]]:
    corr_mat = correlation_matrix(data, headers)

    attr_correlated_elements = np.extract(corr_mat >= CORR_THRESHOLD, corr_mat)
    i, j = np.where(corr_mat >= CORR_THRESHOLD)

    highly_correlated = []

    print("\n===\n")

    for idx, (i1, i2) in enumerate(zip(i, j)):
        h1 = headers[i1]
        h2 = headers[i2]

        highly_correlated.append((h1, h2))

    for (h1, h2) in highly_correlated:
        print(f'{"[target] " if h1 in target or h2 in target else ""}"{h1}" x "{h2}" = {corr_mat[h2][h1]:.2f}')

    attributes = corr_mat.columns.values

    print("\n===\n")

    leave_only_one = []

    for (h1, h2) in highly_correlated:
        need_both = False
        for other in attributes:
            if other == h1 or other == h2:
                continue

            corr_h1_other = corr_mat[other][h1] if isnan(corr_mat[h1][other]) else corr_mat[h1][other]
            corr_h2_other = corr_mat[h2][other] if isnan(corr_mat[other][h2]) else corr_mat[other][h2]

            if isnan(corr_h2_other) or isnan(corr_h1_other):
                continue

            if abs(corr_h2_other - corr_h1_other) >= CORR_DIFF_THRESHOLD:
                need_both = True
                break

        if not need_both:
            leave_only_one.append((h1, h2))
            print(f'"{h1}" and "{h2}" are very much alike, leave one with the higher gain ratio')

    print("\n===\n")

    show_heatmap(corr_mat, headers)

    return leave_only_one


def classes(col: list[float]) -> list[tuple[float, float]]:
    min_value = min(col)
    max_value = max(col)
    span = max_value - min_value
    unique_count = len(set(col))
    bins_count = int(1 + log(unique_count, 2))
    bin_width = span / bins_count

    return [(min_value + bin_width * i, min_value + bin_width * (i + 1)) for i in range(bins_count)]


def within_class(value: float, cls: tuple[float, float], last_class=False) -> bool:
    min_value, max_value = cls
    return min_value <= value < max_value if not last_class else min_value <= value <= max_value + 1e-5


def apply_classes(data: list[list[float]], idx: int, cls: list[tuple[float, float]]) -> None:
    for row in data:
        value = row[idx]
        for i, c in enumerate(cls):
            if within_class(value, c, i == len(cls) - 1):
                row[idx] = i
                break

        if row[idx] not in range(len(cls)):
            raise ValueError('cell value does not belong to any class')


def rows_with_value(data: list[list[float]], idx: int, value: float) -> list[list[float]]:
    return [row for row in data if row[idx] == value or isnan(row[idx]) and isnan(value)]


def equal_classes(c1: tuple[float, float], c2: tuple[float, float]) -> bool:
    return isnan(c1[0]) and isnan(c2[0]) and c1[1] == c2[1] \
           or c1 == c2


def gain_ratios(
        data: list[list[float]],
        indices: list[int],
        target_classes: list[tuple[float, float]]) -> dict[int, float]:
    data_size = len(data)

    def freq(cj: tuple[float, float], t: list[list[float]]) -> int:
        return sum(1 for r in t if equal_classes((r[-2], r[-1]), cj))

    def info(t: list[list[float]] = None) -> float:
        if not t:
            t = data

        t_size = len(t)
        return -sum(freq(cj, t) / t_size * log(freq(cj, t) / t_size, 2) for cj in target_classes if freq(cj, t) != 0)

    def info_x(x: int) -> float:
        unique_values = set(column(data, x, filter_not_nan=True))
        if any(isnan(it) for it in column(data, x)):
            unique_values.add(float('nan'))

        return sum(
            len(rows_with_value(data, x, value)) / data_size * info(rows_with_value(data, x, value))
            for value in unique_values
        )

    def split_info_x(x: int) -> float:
        unique_values = set(column(data, x, filter_not_nan=True))
        if any(isnan(it) for it in column(data, x)):
            unique_values.add(float('nan'))

        return -sum(
            len(rows_with_value(data, x, value)) / data_size
            * log(len(rows_with_value(data, x, value)) / data_size, 2)
            for value in unique_values
        )

    def gain_ratio(x: int) -> float:
        i = info()
        i_x = info_x(x)
        s_i_x = split_info_x(x)
        gr = (i - i_x) / s_i_x
        # print(f'x = {x}, info = {i:.2f}, info_x = {i_x:.2f}, split_info_x = {s_i_x:.2f}, gr = {gr:.2f}')
        return gr

    return {idx: gain_ratio(idx) for idx in indices}


def extract_target_classes(data: list[list[float]]) -> list[tuple[float, float]]:
    cls = set((row[-2], row[-1]) for row in data)
    result = []

    for c in cls:
        if any(equal_classes(c, it) for it in result):
            continue

        result.append(c)

    return result


MAX = 10_000
MIN = -MAX

KGF_IDX = 30


def main() -> None:
    with open(CLEAN_DATA_FILE_NAME, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=';')
        rows = list(reader)
        headers, data = [f'({idx}) {item}' for idx, item in enumerate(rows[0][2:-1])], rows[1:]
        data = parse_data(data)
        data = [row[2:] for row in data]

        target = [len(headers) - 2, len(headers) - 1]

        continuous_idx = []
        categorical_idx = []
        not_enough_data_idx = []
        single_unique_value_idx = []

        # Поглядим на выбросы
        attribute_info = analyze_attributes(data, treat_as_continuous=True)
        # save_outliers(data, headers, attribute_info)
        remove_outliers_idx = [
            (3, 200, MAX),
            (4, None, None),
            (6, None, MAX),
            (7, 60, MAX),
            (9, MIN, 70),
            (10, 102, MAX),
            (12, MIN, None),
            (13, MIN, 200),
            (14, MIN, 7.1),
            (15, MIN, None),
            (17, MIN, 300),
            (18, MIN, None),
            (26, 700, MAX),
            (28, 0.62, MAX),
            (30, MIN, 325)
        ]

        data_len = len(data)

        for idx, low, high in remove_outliers_idx:
            chi_025 = attribute_info[idx]['chi_025']
            chi_075 = attribute_info[idx]['chi_075']

            if not low:
                low = chi_025 - 1.5 * (chi_075 - chi_025)

            if not high:
                high = chi_025 + 1.5 * (chi_075 - chi_025)

            data = remove_outliers(data, idx, low, high)

        print(f'removed {data_len - len(data)} rows of data')

        # Поглядим на пропуски, заполним где можно, удалим где нужно
        attribute_info = analyze_attributes(data)

        # Гистограммы распределения значений
        # save_histograms(data, headers, attribute_info)

        # Распределение признаков по всяким категориям
        for idx, header in enumerate(headers):
            info = attribute_info[idx]

            unique = info["unique"]
            t = info["type"]
            empty = info['empty %']

            if t == 'continuous':
                continuous_idx.append(idx)
            else:
                categorical_idx.append(idx)

            if unique == 1:
                single_unique_value_idx.append(idx)

            if empty > 60 and idx not in target:
                print(f"[not enough data] {header}: {empty:.2f}% missing data")
                not_enough_data_idx.append(idx)
            elif 0 < empty < 30:
                print(
                    f"[can impute] {header} ({info['type']}): {empty:.2f}% missing data")
                impute_data(data, idx, info)

        # for idx in not_enough_data_idx:
        #     if idx not in attribute_info:
        #         continue
        #
        #     print(f'[removed] {headers[idx]}: not enough data')
        #     attribute_info.pop(idx)

        for idx in single_unique_value_idx:
            if idx not in attribute_info:
                continue

            print(f'[removed] {headers[idx]}: single unique value')
            attribute_info.pop(idx)

        somewhat_useful_attributes = sorted(attribute_info.keys())
        categorical_idx = list(filter(lambda it: it in somewhat_useful_attributes, categorical_idx))
        continuous_idx = list(filter(lambda it: it in somewhat_useful_attributes, continuous_idx))

        print(len(somewhat_useful_attributes))

        print('Непрерывные признаки:')
        for idx in continuous_idx:
            info = attribute_info[idx]

            unique = info["unique"]
            total = info["total"]

            print(f'\t{headers[idx]}: {unique}/{total} = {unique / total * 100:.2f}%')

        print('\nКатегориальные признаки:')
        for idx in categorical_idx:
            info = attribute_info[idx]

            unique = info["unique"]
            total = info["total"]

            print(f'\t{headers[idx]}: {unique}/{total} = {unique / total * 100:.2f}%')

        print('\nМного пропусков:')
        for idx in not_enough_data_idx:
            print(f'\t{headers[idx]}')

        print('\nЕдинственное уникальное значение:')
        for idx in single_unique_value_idx:
            print(f'\t{headers[idx]}')

        categorical_idx.append(KGF_IDX)
        attribute_classes = {idx: classes(column(data, idx)) for idx in categorical_idx}
        for idx, cls in attribute_classes.items():
            apply_classes(data, idx, cls)

        # for row in data:
        #     print(row)

        # Корреляции
        highly_correlated = inspect_correlations(data, headers, target)

        target_classes = extract_target_classes(data)
        print(f'{len(target_classes)} target classes total')
        pp.pprint(target_classes)

        # nan = float('nan')

        # test_data = [
        #     [3, 1, nan, 1],
        #     [2, 2, nan, 1],
        #     [7, 3, 1, 2],
        #     [9, 4, 7, 2],
        #     [4, 1, 2, 3],
        #     [5, 2, 6, 3],
        #     [1, 3, 3, 1],
        #     [3, 4, 5, 1],
        #     [2, 1, 4, 2],
        #     [6, 2, nan, 2],
        #     [4, 3, nan, 3],
        # ]
        #
        # tg = extract_target_classes(test_data)
        # pp.pprint(tg)
        # h = ['attr 1', 'attr 2', 'target 1', 'target 2']
        #
        # gr = gain_ratios(test_data, list(range(len(h))), tg)
        # attribute_gain_ratio = sorted(
        #     [(h[idx], gain_ratio) for idx, gain_ratio in gr.items()],
        #     key=lambda item: -item[1])
        # pp.pprint(attribute_gain_ratio)

        gr = gain_ratios(data, somewhat_useful_attributes, target_classes)
        header_to_gain_ratio = {headers[idx]: gain_ratio for idx, gain_ratio in gr.items()}

        drop = []

        for (h1, h2) in highly_correlated:
            if h1 not in header_to_gain_ratio or h2 not in header_to_gain_ratio:
                continue
            h1_gain_ratio = header_to_gain_ratio[h1]
            h2_gain_ratio = header_to_gain_ratio[h2]

            if h1_gain_ratio > h2_gain_ratio:
                drop.append(h2)
                print(f'[drop] "{h2}": correlates with "{h1}" which has a higher gain ratio')
            if h2_gain_ratio > h1_gain_ratio:
                drop.append(h1)
                print(f'[drop] "{h1}": correlates with "{h2}" which has a higher gain ratio')

        for it in drop:
            if it not in header_to_gain_ratio:
                continue
            header_to_gain_ratio.pop(it)

        attribute_gain_ratio = sorted(
            [(header, gain_ratio) for header, gain_ratio in header_to_gain_ratio.items()],
            key=lambda item: -item[1])[2:]
        # pp.pprint(attribute_gain_ratio)

        gain_ratio_values = [gr for (_, gr) in attribute_gain_ratio]
        gain_ratio_headers = [name for (name, _) in attribute_gain_ratio]

        print('Attributes that survived the cleansing:')
        for header in sorted(gain_ratio_headers, key=lambda h: headers.index(h)):
            print(f'\t{header} (gain ratio = {header_to_gain_ratio[header]:.3f})')
            if headers.index(header) in attribute_classes:
                print(f'\t\tclasses = {attribute_classes[headers.index(header)]}')

        plt.clf()
        # plt.xticks(rotation='vertical')
        plt.barh(gain_ratio_headers, width=gain_ratio_values)
        plt.show()


if __name__ == '__main__':
    main()
