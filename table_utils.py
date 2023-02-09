# Code adapted from Ciwan Ceylan
from typing import Iterable
import numpy as np


def float_exponent_notation(float_number, precision_digits, format_type="g", std_number=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with `precision_digits` digits of
    mantissa precision, printing a normal decimal if an
    exponent isn't necessary.
    """
    e_float = "{0:.{1:d}{2:}}".format(float_number, precision_digits, format_type)
    if "e" not in e_float:
        return "{}".format(e_float)
    mantissa, exponent = e_float.split("e")
    cleaned_exponent = exponent.strip("+").lstrip("0")
    if std_number is None:
        return "{0} \\times 10^{{{1}}}".format(mantissa, cleaned_exponent)
    std_mantissa = str(std_number / 10 ** int(cleaned_exponent))[:precision_digits + 1]
    return "{0} \\times 10^{{{1}}} \\pm {2} \\times 10^{{{1}}}".format(mantissa, cleaned_exponent, std_mantissa)


def create_table_start(row_columns, other_cols):
    layout = "{" + f"{len(row_columns) * 'r' + len(other_cols) * 'c'}" + "}"
    out = "\\begin{{tabular}}{layout}".format(layout=layout)
    out = "\n".join([out, "\\toprule", " & ".join(row_columns + other_cols) + "\\\\", "\\midrule\n"])
    return out


def create_table_end():
    return "\\bottomrule\n\\end{tabular}"


def get_best_threshold(grouped, num_sigma=1):
    best_val_threshold = {}
    mean_cols = [c for c in grouped.columns if c[1] == "mean"]
    for row in grouped.index:
        vals = grouped.loc[row, mean_cols]
        col = vals.index[vals.argmin()][0]
        thres = grouped.loc[row, (col, "mean")] + num_sigma * grouped.loc[row, (col, "std")]
        best_val_threshold[row] = (thres, False)
    return best_val_threshold


def tbl_elm(value, std, is_best, num_decimal=5, color="black"):
    if abs(value) < 10 ** num_decimal:
        element = f"{('{' + ':.{}f'.format(num_decimal) + '}').format(round(value, num_decimal))} \\pm {('{' + ':.{}f'.format(num_decimal) + '}').format(round(std, num_decimal))}"
    else:
        element = float_exponent_notation(value, num_decimal, std_number=std)
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    element = "{ \color{" + color + "}" + element + "}"
    return element


def tbl_elm_no_std(value, is_best, num_decimal=2):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def create_tbl_row(grouped, row_index, columns, thres):
    elements = []
    for col in columns:
        val = grouped.loc[row_index, (col, "mean")]
        std = grouped.loc[row_index, (col, "std")]
        if col.lower() in {'time (s)', 'duration'}:
            elements.append(
                tbl_elm_no_std(val, val >= thres[row_index][0] if thres[row_index][1] else val <= thres[row_index][0]))
        else:
            elements.append(
                tbl_elm(val, std, val >= thres[row_index][0] if thres[row_index][1] else val <= thres[row_index][0]))
    if isinstance(row_index, str) or isinstance(row_index, int):
        row_index = [str(row_index)]
    elif isinstance(row_index, Iterable):
        row_index = list(str(r) for r in row_index)
    out = " & ".join(row_index + elements)
    out += "\\\\\n"
    return out


def make_table(df, row_columns, other_columns):
    if isinstance(row_columns, dict):
        df = df.rename(columns=row_columns)
        row_columns = list(row_columns.values())
    if isinstance(other_columns, dict):
        df = df.rename(columns=other_columns)
        other_columns = list(other_columns.values())

    grouped = df.loc[:, row_columns + other_columns].groupby(row_columns).agg(["mean", "std"])
    thres = get_best_threshold(grouped, num_sigma=0)
    print(thres)
    grouped = grouped.T
    out = create_table_start(row_columns, other_columns)

    for row in grouped.index:
        out += create_tbl_row(grouped, row, other_columns, thres)

    out += create_table_end()
    return grouped, out


def create_tbl_row_transpose(row, columns, thres, num_decimal):
    elements = []
    row_index = [row.index.get_level_values(0)[0]]
    for col in row.columns:
        color = "black" if col in row_index[0] else "gray"
        val = row[row.index.get_level_values(1) == "mean"][col].values[0]
        std = row[row.index.get_level_values(1) == "std"][col].values[0]
        elements.append(
            tbl_elm(val, std, val >= thres[col][0] if thres[col][1] else val <= thres[col][0], num_decimal=num_decimal,
                    color=color))
    out = " & ".join(row_index + elements)
    out += "\\\\\n"
    return out


def make_transposed_table(df, row_columns, other_columns, num_decimal):
    if isinstance(row_columns, dict):
        df = df.rename(columns=row_columns)
        row_columns = list(row_columns.values())
    if isinstance(other_columns, dict):
        df = df.rename(columns=other_columns)
        other_columns = list(other_columns.values())

    grouped = df.loc[:, row_columns + other_columns].groupby(row_columns, sort=False).agg(["mean", "std"])
    out = create_table_start([" "], grouped.index.values.tolist())
    thres = get_best_threshold(grouped, num_sigma=0)
    grouped = grouped.T

    vals = grouped.index.get_level_values(0).unique()
    for i in vals:
        row = grouped[grouped.index.get_level_values(0) == i]
        out += create_tbl_row_transpose(row, other_columns, thres, num_decimal)

    out += create_table_end()
    return grouped, out
