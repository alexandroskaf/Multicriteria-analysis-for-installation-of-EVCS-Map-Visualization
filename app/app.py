import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Read Data
data = pd.read_csv('municipalities_criteria_values.csv', delimiter=';')
for column in data.columns:
    if data[column].dtype == 'float64':
        data[column] = data[column].apply(lambda x: round(x, 2))

# PROMETHEE II


def calculate_score(municipalities):

    if len(municipalities) > 1:
        filtered_data = data[data['d_1'].isin(municipalities)].copy()
    else:
        filtered_data = data.copy()  # If not selected municipalities

    filtered_data['distance_from_sea'] = filtered_data['distance_from_sea'].apply(
        lambda x: x if x < 4000 else 4000)
    filtered_data = filtered_data.drop(['score'], axis=1)

    # Normalize Data
    columns_to_normalize = ['mun_category', 'Airports', 'Gas']
    for column in columns_to_normalize:
        min_val = filtered_data[column].min()
        max_val = filtered_data[column].max()
        filtered_data[column] = (
            filtered_data[column] - min_val) / (max_val - min_val)

    columns_to_square_normalize = ['Finance', 'Evcs']
    for column in columns_to_square_normalize:
        min_val = filtered_data[column].min()
        max_val = filtered_data[column].max()
        non_negative_values = filtered_data[column].apply(
            lambda x: x if x >= 0 else np.nan)

        filtered_data[column] = np.sqrt(non_negative_values) / np.sqrt(max_val)

    columns_to_log_normalize = ['Hotels', 'Companies',
                                'Houses', 'distance_from_sea', 'CO2(tons/km^2)']
    for column in columns_to_log_normalize:
        min_val = filtered_data[column].min()
        max_val = filtered_data[column].max()
        filtered_data[column] = np.log(
            filtered_data[column] - min_val + 1) / np.log(max_val - min_val + 1)

    # Distances
    columns_to_abstract = ['Hotels', 'distance_from_sea', 'Companies', 'Houses',
                           'mun_category', 'Airports', 'Finance', 'CO2(tons/km^2)', 'Evcs', 'Gas']
    result_df = pd.DataFrame()
    for column in columns_to_abstract:
        differences = []
        for i, val1 in enumerate(filtered_data[column]):
            for j, val2 in enumerate(filtered_data[column]):
                if i != j:
                    diff = val1 - val2
                    differences.append(diff)
        result_df[column] = differences

    result_df = result_df.applymap(lambda x: max(0, x))

    # Pj(a,b) preference function
    result_df['Companies'] = result_df['Companies'].apply(
        lambda x: 0 if x < 0.05 else 1)
    result_df['Hotels'] = result_df['Hotels'].apply(
        lambda x: 0 if x < 0.05 else 1)
    result_df['Houses'] = result_df['Houses'].apply(
        lambda x: 0 if x < 0.063 else 1)
    result_df['distance_from_sea'] = result_df['distance_from_sea'].apply(
        lambda x: 0 if x < 0.05 else 1)
    result_df['CO2(tons/km^2)'] = result_df['CO2(tons/km^2)'].apply(
        lambda x: 0 if x < 0.05 else 1)
    result_df['Finance'] = result_df['Finance'].apply(
        lambda x: 0 if x <= 0.02 else 1)
    result_df['Evcs'] = result_df['Evcs'].apply(lambda x: 0 if x < 0.05 else 1)
    result_df['Airports'] = result_df['Airports'].apply(
        lambda x: 0 if x == 0 else (0.5 if x == 0.5 else 1))
    result_df['Gas'] = result_df['Gas'].apply(lambda x: 0 if x == 0 else 1)

    return result_df

# Add Weights


def weight_calc(municipalities, weights):
    result_df = calculate_score(municipalities)
    if len(municipalities) > 1:  # Check if municipalities is not an empty string
        filtered_data = data[data['d_1'].isin(municipalities)].copy()
    else:
        filtered_data = data.copy()
    result_weights = result_df.multiply(weights)

    result_row_sums = result_weights.sum(axis=1)
    result_weights['RowSums'] = result_row_sums

    numbers = result_weights['RowSums'].tolist()

    # WEIGHTED PREFERENCE INDEX
    new_df = pd.DataFrame(index=filtered_data.index,
                          columns=filtered_data.index)
    counter = 0
    for i in range(len(new_df.index)):
        for j in range(len(new_df.columns)):
            if i != j:
                new_df.iloc[i, j] = numbers[counter]
                counter += 1

    row_sums = new_df.sum(axis=1)
    column_sums = new_df.sum(axis=0)

    result = pd.DataFrame(index=['Result'], columns=new_df.columns)
    for col in new_df.columns:
        result.loc['Result', col] = row_sums[col] - column_sums[col]

    df_transposed = result.transpose().reset_index()
    df_transposed.columns = ['index', 'Αποτέλεσμα']
    df_transposed['rank'] = df_transposed['Αποτέλεσμα'].rank(
        method='min', na_option='bottom', ascending=False).astype(int)
    df_transposed_not_sorted = df_transposed
    df_transposed = df_transposed.sort_values(by='rank', ascending=True)
    df_transposed['Αποτέλεσμα'] = df_transposed['Αποτέλεσμα'].apply(
        lambda x: round(x, 2))
    merged_df = pd.merge(df_transposed, data[['d_1', 'Hotels', 'distance_from_sea', 'Companies', 'Houses', 'mun_category', 'Airports',
                         'Finance', 'CO2(tons/km^2)', 'Evcs', 'Gas']], left_on='index', right_index=True, how='left').sort_values('rank', ascending=True)
    column_mapping = {
        'rank': 'Κατάταξη',
        'd_1': 'ΔΗΜΟΣ',
        'Hotels': 'Κλίνες Ξενοδοχείων(C1)',
        'distance_from_sea': 'Απόσταση από θάλασσα(μέτρα)(C2)',
        'Companies': 'Επιχειρήσεις(C3)',
        'Houses': 'Πολυκατοικίες(C4)',
        'mun_category': 'Κόστος Εγκατάστασης και Συντήρησης(C5)',
        'Airports': 'Λιμάνια/Αεροδρόμια(C6)',
        'Finance': 'Επιδοτήσεις(€)(C7)',
        'CO2(tons/km^2)': 'CO2(ετήσιοι τόνοι/τ.χλμ)(C8)',
        'Evcs': 'Σταθμοί Φόρτισης Η/Ο(C9)',
        'Gas': 'Τιμή Βενζίνης(€)(C10)'
    }
    # Make Αποτέλεσμα to be the last column
    score_column = merged_df['Αποτέλεσμα']
    merged_df = merged_df.drop(columns=['Αποτέλεσμα'])
    merged_df['Αποτέλεσμα'] = score_column
    # Rename the columns
    merged_df_renamed = merged_df.rename(columns=column_mapping)
    merged_df_renamed = merged_df_renamed.drop(['index'], axis=1)
    return {'rank': df_transposed_not_sorted['rank'].tolist(), 'merged_df': merged_df, 'w_defaults': weights, 'merged_df_renamed': merged_df_renamed}


def take_ranks(municipalities, weights):
    result = weight_calc(municipalities, weights)
    rank_list = result['rank']

    return rank_list

# Sensitivity Analysis


@app.route('/sensitivity_analysis', methods=['GET', 'POST'])
def sensitivity_analysis():

    # Read the Weights

    data_df = pd.read_csv('sensitivity_analysis_weights.csv', delimiter=';')

    # Get selected municipalities from the request
    municipalities = request.args.get('municipalities', '').split(',')
    filtered_data = data[data['d_1'].isin(municipalities)]

    mun = filtered_data['d_1'].tolist()
    weights_list = []
    for column in data_df.columns:
        column_values = data_df[column].apply(lambda x: float(
            x.replace(',', '.')) if isinstance(x, str) else float(x)).tolist()
        weights_list.append(column_values)

    mun = filtered_data['d_1'].tolist()

    # Calculate rank changes
    rank_changes = []
    for weights in weights_list:
        ranks = take_ranks(municipalities, weights)
        rank_changes.append(ranks)

    rank_changes_df = pd.DataFrame(rank_changes, columns=mun)

    # Generate box plots for selected municipalities
    fig = make_subplots(rows=1, cols=1)
    for municipality in municipalities:
        fig.add_trace(
            go.Box(y=rank_changes_df[municipality], name=municipality), row=1, col=1)
    if len(municipalities) < 10:
        height = 600
    else:
        height = 800

    fig.update_layout(
        title="Ανάλυση Ευαισθησίας",
        xaxis=dict(title="ΔΗΜΟΙ"),
        yaxis=dict(title="Rank", autorange="reversed"),
        width=1500,
        height=height
    )

    graph_json = fig.to_json()

    return render_template('sensitivity_analysis.html', graph_json=graph_json)

# Criteria Information and Ranks Page Handler


@app.route('/csv', methods=['GET', 'POST'])
def csv_page():

    municipalities = request.args.get('municipalities', '').split(',')

    w1 = float(request.args.get('w1', 0.11))
    w2 = float(request.args.get('w2', 0.035))
    w3 = float(request.args.get('w3', 0.175))
    w4 = float(request.args.get('w4', 0.145))
    w5 = float(request.args.get('w5', 0.1))
    w6 = float(request.args.get('w6', 0.04))
    w7 = float(request.args.get('w7', 0.165))
    w8 = float(request.args.get('w8', 0.05))
    w9 = float(request.args.get('w9', 0.15))
    w10 = float(request.args.get('w10', 0.03))

    result = weight_calc(
        municipalities, [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10])

    # Unpack the returned dictionary
    merged_df = result['merged_df_renamed']
    w_defaults = result['w_defaults']

    # Handle the case where no municipalities are provided
    if not municipalities:
        return "Error: No municipalities provided."

    # Handle the case where the DataFrame is empty after filtering
    if merged_df.empty:
        return "Error: No data found for the specified municipalities."

    rendered_table = merged_df.to_html(index=False)

    return render_template('csv.html', data=rendered_table, w_defaults=w_defaults, municipalities=municipalities)


@app.route('/score_calc', methods=['GET', 'POST'])
def score_calc():

    municipalities = request.args.get('municipalities', '').split(',')

    # Set default weights (you can change them as needed)
    default_weights = [0.11, 0.035, 0.175, 0.145,
                       0.1, 0.04, 0.165, 0.05, 0.15, 0.03]

    # Call the helper function with the required parameters
    result = weight_calc(municipalities, default_weights)

    # Unpack the returned dictionary
    merged_df = result['merged_df']

    # Read and merge geojson based on municipality name
    df = gpd.read_file('file.geojson')
    merged = df.merge(merged_df, on='d_1')
    merged_json = merged.to_json()
    merged_geojson = json.loads(merged_json)

    return jsonify(merged_geojson)

  # Criteria Information Page


@app.route('/Πληροφορίες_Κριτηρίων')
def new_page():
    return render_template('criteria_information.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000,debug=True)
