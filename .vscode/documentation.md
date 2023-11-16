# app.py
## Routes
- **/ (index)**: Renders the main page (index.html).
- **/sensitivity_analysis**: Handles sensitivity analysis, calculates rank changes for different weights, and displays box plots.
- **/csv**: Displays criteria information and ranks based on user-selected municipalities and weights.
- **/score_calc**: Calculates scores for municipalities and returns a GeoJSON representation for mapping.

## Functions
- **index()**: Renders the main page (index.html).
- **calculate_score(municipalities)**: Applies most steps of PROMETHEE II method (criteria normalization, distances, Pj(a,b) preference function) to calculate scores for selected municipalities based on predefined criteria. Returns result_df Dataframe.
- **weight_calc(municipalities, weights)**: Applies last steps of PROMETHE II method(weighted preference indices) after calling calculate_score(municipalities) function for the selected municipalities. Multiplies the values in result_df by the corresponding weights provided in the weights argument. Constructs a new DataFrame (new_df) based on the weighted scores and calculates the rank for each municipality based on the weighted preference index. Returns a dictionary containing 'rank': A list representing the ranks of the municipalities based on the calculated scores.
'merged_df': A DataFrame containing various columns, including the original and transformed features, scores, and rankings.
'w_defaults': The input weights provided to the function.
'merged_df_renamed': A DataFrame similar to 'merged_df', but with columns renamed according to a specified mapping.
- **take_ranks(municipalities, weights)**: Returns the rank list for a given set of municipalities and weights.
- **sensitivity_analysis()**: Reads the weights from sensitivity_analysis_weights.csv. Iterates through the columns of the weights DataFrame and converts the values to floats. The resulting lists of weights for each column are stored in weights_list. Calls the take_ranks(municipalities,weights) for each set of weights in weights_list and collects the resulting ranks. The rank changes are stored in a DataFrame (rank_changes_df), where each column corresponds to a municipality. Uses the plotly library to create boxplots for the rank changes of each municipality and converts the figure to JSON format.Renders an HTML template named 'sensitivity_analysis.html', passing the JSON representation of the figure.
- **csv_page()**: Retrieves parameters from the HTTP request, including a list of municipalities (municipalities) and weights (w1 to w10). These weights are used to calculate the weighted preference index. Calls the weight_calc function with the provided municipalities and the default weights. Extracts relevant information from the result of the weight_calc function, including the renamed DataFrame (merged_df) and default weights (w_defaults). Converts the DataFrame (merged_df) to an HTML table (rendered_table) without displaying the index. Renders an HTML template named 'csv.html', passing the rendered table (data), default weights (w_defaults), and the list of municipalities.
- **score_calc()**: Retrieves parameters from the HTTP request, including a list of municipalities. Gives default weights. Calls the weight_calc() function with the requested municipalities and the default weights and extracts the merged_df Dataframe. Read the file.geojson using GeoPandas and merged it with merged_df based on column 'd1' which includes municipalities names. Converts the merged GeoJSON data to JSON format and returns it as a JSON response using Flask's jsonify function.
- **new_page()**: Renders the criteria information page.


# index.html
- **var map = L.map('map').setView([37.983810, 23.727539], 7)**: Initializes a Leaflet map centered at latitude 37.983810 and longitude 23.727539 with an initial zoom level of 7.
- **fetch('/score_calc')**: Fetches GeoJSON data from the /score_calc endpoint and processes it when the data is received.
- **updateLegend(attribute)**: Dynamically updates the legend based on the selected attribute-criteria.
- **getColor(value, thresholds)**: Determines the color for a given attribute value based on predefined thresholds.
- **style(feature)**: Defines the style for GeoJSON features, specifying fill color, border color, and opacity.
- **layer.on('mouseover', function ())**: Displays a tooltip and highlights a feature when the mouse is over a GeoJSON feature.
- **layer.on('mouseout', function ())**: Resets the style and removes the tooltip when the mouse moves out of a GeoJSON feature.
- **hideSelectedItems()**: Hides checkboxes for municipalities that are in the selectedMunicipalities array.
- **generateCheckboxes(searchText)**: Generates checkboxes based on the provided searchText-input.
- **municipalityForm.addEventListener('change', function ())**:  Sets up an event listener for changes in the municipality form. It updates the disabled state of the submit button based on the number of checked checkboxes (disable if checked checkboxes less than 2).
- **submitButton.addEventListener('click', function ())**: Sets up an event listener for clicks on the submit button. It calls the hideSelectedItems(), updateChoropleth(), and updateSelectedMunicipalitiesList() functions when the button is clicked.
- **viewCSVPage()**:  Retrieves selected municipalities, constructs a CSV URL, and opens it in a new browser window. 
- **updateChoropleth()**: Updates the choropleth map based on the selected checkboxes. It filters GeoJSON features, creates a new GeoJSON layer, and updates the map with the filtered layer. It also manages the visibility of the CSV button.
- **updateSelectedMunicipalitiesList(selectedMunicipalities)**: Updates the list of selected municipalities, fetches rank data, sorts based on rank, and displays the municipalities with their ranks.

# csv.html

- **goToSensitivityAnalysis()**: Redirect to sensitivity analysis page.
- **resetWeights()**: Resets the weight input fields to their default values and calls the updateSum function.
- **validateForm()**: Validates the form by checking if the input values are valid numbers between 0 and 1 and if the sum of weights is equal to 1.
- **updateSum()**: Calculates the sum of weights, updates the displayed sum, and enables/disables the update button based on whether the sum is equal to 1.
- **downloadCSV()**: Creates a CSV file containing table data and initiates the download of the file.
- **criteriaInformations()**: Redirects the user to a page with criteria information.
