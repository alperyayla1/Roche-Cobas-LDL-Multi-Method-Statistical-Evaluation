import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy import stats
KLS = []  # 2
TGL = []  # 1
HDL = []  # 4
LDL = []  # 3

def convert_to_float(value):
    try:
        if isinstance(value, str):
            value = value.replace(',', '.')
        return float(value)
    except ValueError:
        return None

def filter_sequential_groups(df):
    df_copy = df.copy()
    df_copy = df_copy[~(df_copy['Sonuç'].isna() | (df_copy['Sonuç'].astype(str).str.lower() == 'nan'))].reset_index(drop=True)

    i = 0
    DeletingRows = []

    while i < len(df_copy) - 3:
        current_numune = df_copy['Numune No'].iloc[i]
        current_group = df_copy.iloc[i:i + 4]

        # Check if all 4 rows have the same Numune No
        if len(set(current_group['Numune No'])) == 1:
            # Check if the group contains all required tests
            tests = set(current_group['Test Adı'])
            required_tests = {'Trigliserit', 'Kolesterol, total', 'LDL-kolesterol', 'HDL-Kolesterol'}

            if tests == required_tests:
                # Get values in correct order
                group_dict = {row['Test Adı']: row['Sonuç'] for _, row in current_group.iterrows()}

                TGL.append(convert_to_float(group_dict['Trigliserit']))
                KLS.append(convert_to_float(group_dict['Kolesterol, total']))
                LDL.append(convert_to_float(group_dict['LDL-kolesterol']))
                HDL.append(convert_to_float(group_dict['HDL-Kolesterol']))
                i += 4
            else:
                DeletingRows.append(df_copy.index[i])
                i += 1
        else:
            DeletingRows.append(df_copy.index[i])
            i += 1

    # Add remaining rows to DeletingRows
    if i < len(df_copy):
        DeletingRows.extend(df_copy.index[i:])

    return df_copy.drop(DeletingRows).reset_index(drop=True)

def martin_constant(TGL_Value, HDL_Value):
    martin_path = "C:/Users/alper/OneDrive/Masaüstü/martindataset.xlsx"
    MartinData = pd.read_excel(martin_path, header=None)
    MartinData.iloc[0, 0] = None
    MartinData = MartinData.to_numpy().astype(float)

    row_number = 69
    column_number = 6

    for idx, row in enumerate(MartinData[1:, 0], start=1):
        if TGL_Value <= row:
            row_number = idx
            break

    for j, column in enumerate(MartinData[0, 1:]):
        if HDL_Value <= column:
            column_number = j + 1
            break

    return MartinData[row_number, column_number]

# Read and process data
datas = pd.read_excel("C:/Users/alper/OneDrive/Masaüstü/ROCHE (1).xls")
datas['Sonuç'] = datas['Sonuç'].astype("str").apply(convert_to_float)
datas = filter_sequential_groups(datas)

# Convert lists to numpy arrays
TGL = np.array([x for x in TGL if x is not None], dtype=float)
KLS = np.array([x for x in KLS if x is not None], dtype=float)
LDL = np.array([x for x in LDL if x is not None], dtype=float)
HDL = np.array([x for x in HDL if x is not None], dtype=float)

# Calculate all formulas
Friedewald = KLS - HDL - TGL / 5
Yayla = KLS - HDL - (np.sqrt(TGL) * KLS / 100)
Sampson = (KLS / 0.948) - (HDL / 0.971) - (TGL / 8.56 + TGL * (KLS - HDL) / 2140 - (TGL ** 2) / 16100) - 9.44
Martin = np.zeros(len(LDL))
for i in range(len(LDL)):
    Martin[i] = KLS[i] - HDL[i] - (TGL[i] / martin_constant(TGL[i], KLS[i] - HDL[i]))

# Calculate and print statistics
def calculate_statistics(measured, calculated, method_name):
    error = measured - calculated
    bias = np.mean(error)
    std_error = np.std(error)
    rmse = np.sqrt(mean_squared_error(measured, calculated))
    mae = mean_absolute_error(measured, calculated)
    r2 = r2_score(measured, calculated)
    correlation, p_value = pearsonr(measured, calculated)
    mape = np.mean(np.abs(error / measured)) * 100

    return {
        'Method': method_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Bias': bias,
        'Standard Error': std_error,
        'Correlation': correlation,
        'P-value': p_value,
        'MAPE (%)': mape
    }

methods = {
    'Friedewald': Friedewald,
    'Yayla': Yayla,
    'Sampson': Sampson,
    'Martin': Martin
}

# Create statistics DataFrame
stats_list = []
for method_name, calculated_values in methods.items():
    stats = calculate_statistics(LDL, calculated_values, method_name)
    stats_list.append(stats)

stats_df = pd.DataFrame(stats_list)

# Save results to Excel
with pd.ExcelWriter('ldl_analysis_results.xlsx') as writer:
    stats_df.to_excel(writer, sheet_name='Statistics', index=False)


def print_statistics_table(data_dict):
    print("\nSummary Statistics Table")
    print("-" * 60)
    print(f"{'Parameter':<15} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)

    for name, data in data_dict.items():
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        print(f"{name:<15} {mean_val:>10.2f} {min_val:>10.2f} {max_val:>10.2f}")
    print("-" * 60)


# Create dict of data
data_stats = {
    'Kolesterol': KLS,
    'LDL': LDL,
    'HDL': HDL,
    'Trigliserit': TGL
}

print_statistics_table(data_stats)


# Create LDL categories table
def print_ldl_categories():
    # Create LDL categories
    ldl_categories = {
        'LDL < 70': len([x for x in LDL if x < 70]),
        '70 ≤ LDL < 100': len([x for x in LDL if 70 <= x < 100]),
        '100 ≤ LDL < 130': len([x for x in LDL if 100 <= x < 130]),
        '130 ≤ LDL < 160': len([x for x in LDL if 130 <= x < 160]),
        '160 ≤ LDL < 190': len([x for x in LDL if 160 <= x < 190]),
        'LDL ≥ 190': len([x for x in LDL if x >= 190])
    }

    print("\nLDL Categories Distribution")
    print("-" * 50)
    print(f"{'Category':<20} {'Count':>10} {'Percentage':>15}")
    print("-" * 50)
    total = sum(ldl_categories.values())
    for category, count in ldl_categories.items():
        percentage = (count / total) * 100
        print(f"{category:<20} {count:>10} {percentage:>14.1f}%")
    print("-" * 50)
    print(f"{'Total':<20} {total:>10}")


# Create TGL categories table
def print_tgl_categories():
    # Create TGL categories
    tgl_categories = {
        'TGL < 100': len([x for x in TGL if x < 100]),
        '100 ≤ TGL < 150': len([x for x in TGL if 100 <= x < 150]),
        '150 ≤ TGL < 200': len([x for x in TGL if 150 <= x < 200]),
        '200 ≤ TGL < 400': len([x for x in TGL if 200 <= x < 400]),
        'TGL ≥ 400': len([x for x in TGL if x >= 400])
    }

    print("\nTGL Categories Distribution")
    print("-" * 50)
    print(f"{'Category':<20} {'Count':>10} {'Percentage':>15}")
    print("-" * 50)
    total = sum(tgl_categories.values())
    for category, count in tgl_categories.items():
        percentage = (count / total) * 100
        print(f"{category:<20} {count:>10} {percentage:>14.1f}%")
    print("-" * 50)
    print(f"{'Total':<20} {total:>10}")


# Create MSE comparison table
def print_mse_by_categories():
    print("\nMSE Analysis by Categories")
    print("-" * 85)
    print(f"{'Category':<20} {'Friedewald':>15} {'Sampson':>15} {'Yayla':>15} {'Martin':>15}")
    print("-" * 85)

    # LDL categories
    ldl_ranges = [
        (lambda x: x < 70, 'LDL < 70'),
        (lambda x: 70 <= x < 100, '70 ≤ LDL < 100'),
        (lambda x: 100 <= x < 130, '100 ≤ LDL < 130'),
        (lambda x: 130 <= x < 160, '130 ≤ LDL < 160'),
        (lambda x: 160 <= x < 190, '160 ≤ LDL < 190'),
        (lambda x: x >= 190, 'LDL ≥ 190')
    ]

    for condition, label in ldl_ranges:
        indices = [i for i, x in enumerate(LDL) if condition(x)]
        if indices:
            f_mse = mean_squared_error(LDL[indices], Friedewald[indices])
            s_mse = mean_squared_error(LDL[indices], Sampson[indices])
            y_mse = mean_squared_error(LDL[indices], Yayla[indices])
            m_mse = mean_squared_error(LDL[indices], Martin[indices])
            print(f"{label:<20} {f_mse:>15.2f} {s_mse:>15.2f} {y_mse:>15.2f} {m_mse:>15.2f}")
    print("-" * 85)


# Print all tables
print(f"\nTotal Number of Patients: {len(LDL)}")
print_statistics_table(data_stats)
print_ldl_categories()
print_tgl_categories()
print_mse_by_categories()


def print_comprehensive_statistics_by_categories():
    # First table - RMSE, MAE, R²
    print("\nStatistical Analysis by Categories - Part 1")
    print("-" * 100)
    print(f"{'Category':<15} {'Method':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE(%)':>10} {'Bias':>10}")
    print("-" * 100)

    # LDL categories
    ldl_ranges = [
        (lambda x: x < 70, 'LDL < 70'),
        (lambda x: 70 <= x < 100, '70 ≤ LDL < 100'),
        (lambda x: 100 <= x < 130, '100 ≤ LDL < 130'),
        (lambda x: 130 <= x < 160, '130 ≤ LDL < 160'),
        (lambda x: 160 <= x < 190, '160 ≤ LDL < 190'),
        (lambda x: x >= 190, 'LDL ≥ 190')
    ]

    methods = {
        'Friedewald': Friedewald,
        'Sampson': Sampson,
        'Yayla': Yayla,
        'Martin': Martin
    }

    for condition, label in ldl_ranges:
        indices = [i for i, x in enumerate(LDL) if condition(x)]
        if indices:
            for method_name, values in methods.items():
                stats = calculate_statistics(LDL[indices], values[indices], method_name)

                print(f"{label:<15} {method_name:<12} "
                      f"{stats['RMSE']:>10.2f} {stats['MAE']:>10.2f} {stats['R²']:>10.3f} "
                      f"{stats['MAPE (%)']:>10.1f} {stats['Bias']:>10.2f}")
            print("-" * 100)

    # Second table - Correlation, P-value, Standard Error
    print("\nStatistical Analysis by Categories - Part 2")
    print("-" * 90)
    print(f"{'Category':<15} {'Method':<12} {'Correlation':>12} {'P-value':>12} {'Std Error':>12}")
    print("-" * 90)

    for condition, label in ldl_ranges:
        indices = [i for i, x in enumerate(LDL) if condition(x)]
        if indices:
            for method_name, values in methods.items():
                stats = calculate_statistics(LDL[indices], values[indices], method_name)

                print(f"{label:<15} {method_name:<12} "
                      f"{stats['Correlation']:>12.3f} {stats['P-value']:>12.3e} "
                      f"{stats['Standard Error']:>12.2f}")
            print("-" * 90)


# Call the function
print_comprehensive_statistics_by_categories()