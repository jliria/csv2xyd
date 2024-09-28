# csv2xyd
csv2xyd is a Python-based software designed to process large CSV files and convert them into XYD format, ideal for endemism analysis with NDM/vNDM. It provides advanced functionalities for preprocessing, filtering, and combining biodiversity data while offering spatial analysis capabilities.

System Requirements:

•	Python 3.12.4 or higher.

•	Required libraries: 
Tkinter 0.1.0 
Pandas 2.2.2
Dask 2024.7.0
FuzzyWuzzy 0.18.0
Folium 0.17.0
NumPy 2.0.0
GeoPandas 1.0.1 
Shapely 2.0.5

•	Recommended hardware: 16 GB of RAM or more for handling large datasets.

Sample data are available at: https://doi.org/10.5281/zenodo.13851938

File Menu
1. Open Semi-Formatted CSV
•	Use this option to open a CSV file with essential columns like species (genus and species), latitude, and longitude. Additional columns for higher taxonomic levels (e.g., phylum, class, order) are also supported.
•	Example: Open the file "World_Arthropoda.csv", choose the delimiter (comma or tab), and the program will display the number of occurrences and estimated memory usage.
2. Open CSV from a Biodiversity Repository
•	Load CSV files from repositories like GBIF. The program allows you to choose the relevant columns (e.g., phylum, class, order, species) and apply filters such as valid coordinates or specific taxonomic groups.
•	Example: Open "0024888-240626123714530.csv", select columns, filter for "Amphibia", and save the filtered file as "Chordata_amphibia.csv".
3. Merge and Sort CSV Files
•	This feature allows merging multiple CSV files and sorting the combined dataset by specific fields.
•	Example: Merge "File_1.csv", "File_2.csv", etc., and sort them by genus and species, saving the result as, for example "File_merge.csv".
4. Reorder Columns
•	After merging CSVs, you can rearrange the column order to suit your needs.

Preprocessing Menu
1. Remove Duplicates
•	Removes duplicate species entries with the same coordinates.
2. Occurrence Filter
•	Set a minimum number of occurrences per species (default is 10), ensuring that only species with sufficient data are processed.
3. Detect Typographical Errors
•	This option detects species names with small differences, which may indicate typos. A similarity threshold (default: 90%) can be adjusted. The program generates a report (possible_typo_errors.csv) for review and correction.
4. Show Occurrence Map
•	Displays a random selection of occurrences on an interactive map using Folium. The default sample size is 10%, but this can be adjusted.
5. Detect Taxonomic Errors
•	Identifies species with inconsistent taxonomic assignments, generating a report for review.

Processing Menu
1. Process CSV for XYD
•	Convert a CSV file into the XYD format required for NDM/vNDM analysis. Select columns (e.g., genus and species), define grid limits (gridx, gridy), and fill values to generate the XYD file.
2. Export to TNT
•	The program also generates a TNT file that can be processed with gettaxo.run macro to include higher groups in the endemism analysis.

Geospatial Analysis Menu
1. Run Geospatial Analysis
•	Perform exploratory spatial analysis on species occurrences and polygons. Load a shapefile or generate a custom grid to match the XYD file’s boundaries.
•	Adjust the grid size and analyze richness, diversity, and species evenness across the spatial distribution of data.

