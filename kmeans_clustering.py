# _*_ Coding: UTF-8 _*_
# Script: kmeans_clustering.py
# Author: Paula Álvarez López, Adrián Enrico Baissero García y María José López Caro.
# Date: 30/1/2023

########################################################################################################################
#           This script presents a comprehensive implementation of the KMeans clustering algorithm, designed
#           to be both user-friendly and highly interactive. The script focuses on adapting to a broad range
#           of databases with a varying amount of features and visualizing every step the script takes, letting
#           the user understand the underlying process. It provides valuable tools for exploring and gaining
#           insights into the structure of complex datasets. The script is adapted from morenobcn's capstone
#           GitHub project where he analyses hotel price clustering.
########################################################################################################################

# IMPORTS

# System import
import os

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit learn & Others
import csv
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import inv_boxcox
import numpy as np
from scipy import stats
import sys
from IPython.display import display

# Rename the KMeans function.
kmeans = KMeans()

# Avoid multithreading.
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

# FUNCTIONS


def path_message(dir_search, fail_quit=False, is_file=True):
    """It checks if a file in the given directory exists, and prints a message accordingly. If fail_quit is
    set to True, it also may end the script if the file has not been found.

    Parameters:
        dir_search --> File/Directory being searched.
        fail_quit --> If is set to True, the script will end if the file is not found.
        is_file --> If is set to True, the script will specifically search for a file, not a directory."""

    import os
    if is_file:
        if os.path.isfile(dir_search):
            print(decor_mid + "\n" + "The file in the directory {0} has been found.".format(dir_search))
            path_found = True
        else:
            print(decor_mid + "\n" + "The file in the directory {0} has NOT been found.".format(dir_search))
            print("Please, check that the defined file exists in the given directory!.")
            path_found = False
            # If the file is absolutely necessary for the script to work, the script may be ended if
            # the search has failed.
            if fail_quit:
                quit(0)
    # When the element search is a directory path, and not a file.
    else:
        if os.path.exists(dir_search):
            print(decor_mid + "\n" + "The directory {0} has been found.".format(dir_search))
            path_found = True
        else:
            print(decor_mid + "\n" + "The directory {0} has NOT been found.".format(dir_search))
            print("Please, check that the given directory exists!.")
            path_found = False
            # If the directory is absolutely necessary for the script to work, the script may be ended if
            # the search has failed.
            if fail_quit:
                quit(0)
    # path_found is returned so that the script may continue accordingly.
    return path_found


def enumerate_list(list_dict, exclude=None, inc_dict=False):
    """It enumerates a list out of a dictionary, then lets the used select out of the printed list.
    The key in the dictionary is replaced for self-increasing numeric value starting on 1. You may include a list
    of the keys you may want to exclude when printing. It returns the user's choice as a string number.

    Parameters:
        list_dict --> The list/dictionary containing the data that will be enumerated.
        exclude --> A list containing the keys that need to be excluded from the dictionary.
        inc_dict --> If is set to True, the script will expect a dictionary."""

    if exclude is None:
        exclude = []

    left_number = 1
    input_numbers = "("

    if inc_dict:
        for key, value in list_dict.items():

            # Only prints the non-excluded values in the dictionary.
            if key not in exclude:

                # Prints the values in a '1. value' format.
                print(str(left_number) + ". " + value)

                # input_numbers stores how many values are printed, so that it can ask the user accordingly.
                input_numbers += (str(left_number) + "/")
                left_number += 1
    else:
        # Works with a list instead of a dictionary. Cannot exclude values.
        for value in list_dict:

            # Prints the values in a '1. value' format.
            print(str(left_number) + ". " + value)

            # input_numbers stores how many values are printed, so that it can ask the user accordingly.
            input_numbers += (str(left_number) + "/")
            left_number += 1

    # The last / is changed for a '): ' for aesthetic purposes.
    input_numbers = input_numbers[:-1] + "): "
    user_choice = input(input_numbers)

    return user_choice


def find_delimiter(file_path):

    """This function automatically detects the delimiter being used in a csv file and returns
    the character. If none of the delimiters work, returns None.

    Parameters:
        file_path --> The file it's delimiters will be checked."""

    # Open the file located at file_path for reading.
    with open(file_path, "r") as csv_read:
        # List of delimiters to try.
        delimiters = [",", ";", "\t", "|"]

        # Loop through the list of delimiters.
        for delim in delimiters:
            # Reset the file pointer to the beginning of the file.
            csv_read.seek(0)

            # Try to read the file using the current delimiter.
            try:
                reader = csv.reader(csv_read, delimiter=delim)
                header_del = next(reader)

                # If the header row contains more than one column, return the current delimiter.
                if len(header_del) > 1:
                    return delim
            # If an error occurs, move on to the next delimiter.
            except:
                pass

    # If none of the delimiters work, return None.
    return None


def corr_cleaning(df):

    """This function removes the duplicate or redundant pairs from all the correlation
    values of a pandas dataframe. Returns a set of the names of said pairs.

    Parameters:
        df --> A pandas dataframe."""

    # Initialize an empty set called "duplicate_pair".
    duplicate_pair = set()

    # Get the names of all columns in the dataframe "df" and assign it to the variable "column".
    column_in = df.columns

    # Loop over the number of columns in the dataframe.
    for i in range(0, df.shape[1]):

        # Loop over the columns from 0 to the current column (i).
        for j in range(0, i + 1):
            # Add a tuple of the column names(column[i], column[j]) to the set "duplicate_pair".
            duplicate_pair.add((column_in[i], column_in[j]))

    # Return the set "duplicate_pair".
    return duplicate_pair


def corr_max(df, n=5):

    """This function calculates the correlation values of a pandas dataframe
    and returns a sorted list of said values. Needs the corr_cleaning function to work.
    Returns any number of pairs (n="") with its correlation value in a serial object.

    Parameters:
        df --> A pandas dataframe.
        n  --> The number of pairs the function will return."""

    # Calculate the absolute correlation values between all columns of the dataframe.
    corr_values = df.corr().abs().unstack()

    # Remove duplicate pairs of columns.
    duplicate_pair = corr_cleaning(df)
    corr_values = corr_values.drop(labels=duplicate_pair)

    # Sort the correlations in descending order and return the top n values.
    corr_values = corr_values.sort_values(ascending=False)

    return corr_values[0:n]


def elbow_clustering(df):

    """This function calculates the inertia values from 1 to 15 clusters using kmeans algorithm.
    Returns both the inertia values and the list of posible numbers of clusters, so it may be used
    in a plot.

    Parameters:
        df --> A pandas dataframe."""

    # Initialize an empty list to store the inertia values for each number of clusters.
    inertia_in = []

    # Define a range of possible number of clusters from 1 to 15.
    pos_clust_in = range(1, 15)

    # Loop through each possible number of clusters.
    for clust in pos_clust_in:
        # Create a KMeans object with the specified number of clusters and 10 initializations.
        km = KMeans(n_clusters=clust, n_init=10)

        # Fit the KMeans model to the input data.
        km = km.fit(df)

        # Append the inertia value for the current number of clusters to the list.
        inertia_in.append(km.inertia_)

    # Return the list of possible number of clusters and the corresponding inertia values.
    return pos_clust_in, inertia_in


# Decorating variables.
decor_mid = (120 * "*")
decor_imp = (120 * "#")

# Try to find a directory path in the run parameters.
try:

    # Before asking the user, if the parameter does not exist, it jumps to the exception.
    dir_origen = sys.argv[1]

    # The user is asked as the first parameter may not contain a directory path.
    user_answer = input("Does the first run parameter contain the directory path? (Y/N): ")

    # if the user answer anything other than "Y"/"y", the IndexError is raised.
    if user_answer.lower() != "y":
        raise IndexError

# Ask the user for the directory name in the given parent directory.
except IndexError:

    # May be changed by the user to the working directory they choose.
    dir_parent = "D:/DatosProg"

    # The actual name of the directory.
    dir_child = input("Please, introduce the name of the directory you wish to read: ")

    # Both paths are joined.
    dir_origen = os.path.join(dir_parent, dir_child)

    # To avoid raising exceptions when opened, the given path is checked. The fail_quit parameter is set to True,
    # so it will end the script if the directory does not exist (Informs the user first).
    path_message(dir_origen, fail_quit=True, is_file=False)

# Try statement to catch any potential errors that may occur while importing the database.
try:
    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("DATABASE IMPORT")
    print("\n" + decor_mid)

    # Finding the delimiter of the csv file.
    delimiter = find_delimiter(dir_origen)

    # If delimiter is found.
    if delimiter:
        print(f"CSV Delimiter detected: {delimiter}")
        delimiter_answer = input("Is that correct? (Y/N): ")
    # If delimiter is not found.
    else:
        print("The Delimiter was not found. What is the separating character for the CSV?")
        delimiter_answer = "n"

    # If delimiter answer is 'n', user inputs desired delimiter.
    if delimiter_answer.lower() == "n":
        delimiter = input("--> ")

    # Reading csv file using pandas read_csv method.
    import_data = pd.read_csv(dir_origen, delimiter=delimiter)
    csv_data = dir_origen

    # Confirming successful import of csv data.
    print("The data has been red successfully.")
    print("\n" + decor_imp + "\n")

    # Printing separator line.
    print("K-MEANS CLUSTERING DATA PREPROCESSING")
    print("\n" + decor_imp)

    # First step in preprocessing.
    print("First Step:")
    print("The clustering variables requires the BOX-COX / LOG transformation for the algorithm to work correctly.")
    print("Do you wish to apply the BOX-COX / LOG transformations on the fly?")
    print("Select 'N' if the data is already transformed.")

    # Storing the answer for later.
    box_cox_answer = input("(Y/N): ")
    print(decor_mid)

    # Second step in preprocessing.
    print("Second Step:")
    print("K-means clustering cannot work when some data has null values. Removing any null rows from the dataframe...")
    print("This WILL NOT affect the original file.")

    # Removing all the null data from the dataframe.
    import_data = import_data.dropna()
    print(decor_mid)

    # Third step in preprocessing.
    print("Third Step:")
    print("K-means algorithm is very sensitive to outlier data.")
    print("Removing outliers will significantly improve the results.")
    print("Do you wish to remove data 3 standard deviations away from the mean value?")

    # Storing the answer for later.
    outlier_answer = input("(Y/N): ")

    # Reading the header of the csv file.
    with open(csv_data) as header_read:
        header_reader = csv.reader(header_read, delimiter=delimiter)
        header = next(header_reader)

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("DATABASE PREPARATION")
    print("\n" + decor_mid)
    print("Any non-numerical columns must be removed from the dataframe.")
    print("Does your database contain any non-numerical columns?")

    # Get the user's answer to the above prompt.
    non_num_answer = input("(Y/N): ")

    print(decor_mid)

    # If the user answered "y".
    if non_num_answer.lower() == "y":

        # Print a prompt asking the user to select the columns they want to drop.
        print("Select the columns you want to drop. Write 'stop' when all the non-numerical columns are removed.")

        # Use a "while" loop to repeat the prompt until the user enters "stop".
        while True:
            print(decor_mid)

            # Shows the header to the user.
            non_num_drop_answer = (enumerate_list(header))

            # If the user enters "stop", break out of the loop.
            if non_num_drop_answer.lower() == "stop":
                break

            # Convert the user's answer from a string to an integer.
            non_num_drop_answer = int(non_num_drop_answer)

            # Store the column's name the user wants to drop.
            dataframe_pop = header[non_num_drop_answer - 1]

            # Remove the column the user selected from the list.
            header.pop(non_num_drop_answer - 1)

            # Remove the column the user wants to drop from the "import_data" dataframe.
            import_data.pop(dataframe_pop)

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("DATA TYPE SEPARATION")
    print("\n" + decor_mid)

    # Message explaining why categorical data cannot be used in the K-means algorithm.
    print("K-means algorithm cannot use numeric-categorical data in the clustering.")
    print("Data correlation may still be studied, but using different formulas, in two separate dataframes.")
    print("Does your database contain any categorical columns?")

    # Make a copy of the "import_data" dataframe and "header" list for later use.
    import_data_copy = import_data.copy()
    header_copy = header[:]

    # Define the variable to avoid errors.
    category_dataframe = []
    # Ask the user if the database contains categorical values.
    categ_answer = input("(Y/N): ")
    if categ_answer.lower() == "y":

        # Print a prompt asking the user to select the columns they want to separate.
        print("Select the columns you want to separate. Write 'stop' when all the categorical columns are separated.")

        # Initialize an empty list to store the categorical columns.
        category_list = []

        # The loop keeps repeating.
        while True:
            print(decor_mid)

            # Shows the header to the user.
            categ_drop_answer = (enumerate_list(header))

            # If the user enters "stop", break out of the loop.
            if categ_drop_answer.lower() == "stop":
                break

            # Convert the user's answer from a string to an integer.
            categ_drop_answer = int(categ_drop_answer)

            # Appends the categorical columns names into a list.
            category_list.append(header[categ_drop_answer-1])

            # Remove the column the user selected from the list.
            header.pop(categ_drop_answer - 1)

        # Extracts from the dataframe the information of the selected columns.
        category_columns = import_data[category_list]

        # Copies the categorical columns into an empty dataframe.
        category_dataframe = category_columns.copy()

        # Then removes the selected categorical columns from the original dataframe.
        for column in category_list:
            import_data.pop(column)

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("BASE ATTRIBUTES SELECTION")
    print("\n" + decor_mid)

    # First necessary column: Latitude.
    print("Column that contains Latitude value:")

    # Convert the user's answer from a string to an integer.
    lat_answer = int(enumerate_list(header))

    # Store the column's name the user wants to drop.
    lat_data = header[lat_answer - 1]

    # Remove the column the user selected from the list.
    header.pop(lat_answer - 1)

    # Remove the latitude column from the original dataframe.
    import_data.pop(lat_data)
    print(decor_mid)

    # Second necessary column: Longitude.
    print("Column that contains Longitude value:")

    # Convert the user's answer from a string to an integer.
    long_answer = int(enumerate_list(header))

    # Store the column's name the user wants to drop.
    long_data = header[long_answer - 1]

    # Remove the column the user selected from the list.
    header.pop(long_answer - 1)

    # Remove the longitude column from the original dataframe.
    import_data.pop(long_data)
    print(decor_mid)

    # Third necessary column: Main variable.
    print("Column that contains the main variable of interest:")

    # Convert the user's answer from a string to an integer.
    varint_answer = int(enumerate_list(header))

    # Store the column's name the user wants to drop.
    varint_data = header[varint_answer - 1]
    print(decor_mid)

    # Printing separator line.
    print("\n" + decor_imp + "\n")
    print("PRE-CLUSTERING DATA ANALYSIS")
    print("\n" + decor_imp)

    # Data dispersion analysis.
    print("First Analysis: Data dispersion")

    # User may choose whether to print the dispersion using another columns as thematic value or plain.
    print("Data dispersion may be plotted using a thematic column:")
    print("1. Use a thematic column" + "\n" + "2. Print only data dispersion")
    user_answer = int(input("(1/2): "))

    # Setting the plot size.
    plt.figure(figsize=(12, 12))

    # If the user wants to add thematic value.
    if user_answer == 1:

        # Display the available columns from the header copy. (Contains the dropped columns, except non-numerical).
        thematic_answer = int(enumerate_list(header_copy))

        # Stores the data from the chosen column.
        thematic_data = header_copy[thematic_answer - 1]

        # Generates a scatter plot using the longitud and latitude as values, including the chosen thematic column.
        # legend='full' would show the whole set of values for that column, which is not convenient in soft coding.
        sns.scatterplot(x=long_data, y=lat_data, data=import_data_copy, hue=thematic_data, palette='BrBG')

    # If the user wants to print only the data dispersion.
    if user_answer == 2:

        # Generates a scatter plot using the longitud and latitude as values.
        sns.scatterplot(x=long_data, y=lat_data, data=import_data_copy)

    # Shows the plot to the user.
    plt.show()

    # Pearson correlation analysis.
    print("\n")
    print("Second Analysis: Data correlation")

    # From the continuous dataframe.
    print("One heatmap will be generated for continuous features.")

    # From the previously separated categorical dataframe, if any columns are present.
    print("The correlation value of the categorical columns to the main variable will be printed.")
    print("\n")

    # The categorical features return the p-value as well.
    print("Pearson Correlation of the categorical features (First number is correlation, second p-value:")

    # If any categorical column is present.
    if categ_answer.lower() == "y":

        # For each column in the dataframe.
        for c in category_dataframe:

            # Checks the size of the column's unique rows.
            size = int(len(category_dataframe[c].unique()))

            # If there are more than two unique values: polychotomous variable.
            if size == 2:

                # Pearson correlation function between the main variable of interest and the polychotomous variable.
                correlation = stats.pearsonr(category_dataframe[c], import_data[varint_data])

            # Dichotomous variables.
            else:

                # Pearson correlation function between the main variable of interest and the dichotomous variable.
                correlation = stats.pointbiserialr(category_dataframe[c], import_data[varint_data])

            # Prints the result for both types of categorical variables.
            print(f"Correlation of [{c}] to [{varint_data}] is [{correlation}]")

    # Sets the plot size and title.
    plt.figure(figsize=(12, 12))
    plt.title("Pearson Correlation of the continuous features")

    # Generates a seaborn heatmap to show the correlation between all continuous features.
    sns.heatmap(import_data.corr(), annot=False, linewidths=.5, cmap="YlGnBu", square=True)

    # Shows the plot to the user.
    plt.show()

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("GENERATING THE MULTIDIMENSIONAL ARRAY TO BEGIN THE CLUSTERING")
    print("\n" + decor_mid)

    # The user may choose additional variables.
    print("You may now choose up to three additional variables before beginning the clustering.")
    print("Keep in mind these must correlate to the main variable")
    print("1. Choose additional variables manually from the database" + "\n" +
          "2. Automatically choose the highest correlating variables" + "\n" +
          "3. DO NOT CHOOSE ANY ADDITIONAL VARIABLES")

    # Define additional variables to avoid errors.
    add_variable1 = ""
    add_variable2 = ""
    add_variable3 = ""
    # Stores the user's decision.
    corr_answer = int(input("(1/2/3): "))

    # If the user wants to manually choose the variables.
    if corr_answer == 1:
        print(decor_mid)

        # Copies the header into an empty variable.
        corr_header = header[:]

        # First variable.
        print("First additional variable:")

        # Display the available columns from the header's copy and convert answer to an integer.
        corr_var_answer = int(enumerate_list(corr_header))

        # Store the column's name the user wants to drop.
        add_variable1 = corr_header[corr_var_answer - 1]

        # Remove the column the user selected from the list.
        corr_header.pop(corr_var_answer - 1)
        print(decor_mid)

        # Lets the user choose to not include any additional variables, by appending it to the header's copy.
        corr_header.append("DO NOT CHOOSE ANY ADDITIONAL VARIABLES")

        # Display the available columns from the header's copy and convert answer to an integer.
        corr_var_answer = int(enumerate_list(corr_header))

        # Store the column's name the user wants to drop.
        add_variable2 = corr_header[corr_var_answer - 1]

        # If the user wants to choose a second variable.
        if add_variable2 != "DO NOT CHOOSE ANY ADDITIONAL VARIABLES":

            # Remove the column the user selected from the list.
            corr_header.pop(corr_var_answer - 1)
        print(decor_mid)

        # If the user chose a second variable.
        if add_variable2 != "DO NOT CHOOSE ANY ADDITIONAL VARIABLES":

            # Display the available columns from the header's copy and convert answer to an integer.
            corr_var_answer = int(enumerate_list(corr_header))

            # Store the column's name the user wants to drop.
            add_variable3 = corr_header[corr_var_answer - 1]

            # If the user wants to choose a third variable.
            if add_variable3 != "DO NOT CHOOSE ANY ADDITIONAL VARIABLES":

                # Remove the column the user selected from the list.
                corr_header.pop(corr_var_answer - 1)

    # If the user wants the variables chosen automatically.
    if corr_answer == 2:

        # Runs the corr_max function, getting a sorted list of all the correlation values.
        corr_max_varint = corr_max(df=import_data, n=-1)

        # Empty list to store the three chosen variables.
        cluster_add_variables = []

        # To count the amount of times the loop has run.
        counter = 0

        # For each label row in the serial object.
        for item_label in corr_max_varint.items():

            # If the main variable is in the serial object (We check correlation only to the main variable).
            if varint_data in item_label[0]:

                # Return the name of the additional variable.
                var_label = item_label[0]

                # Append the additional variable to the list.
                cluster_add_variables.append(var_label[0])

            # If the counter reaches 3, the loop breaks.
                counter += 1
            if counter == 3:
                break

        # Separate the list into three different variables.
        add_variable1 = cluster_add_variables[0]
        add_variable2 = cluster_add_variables[1]
        add_variable3 = cluster_add_variables[2]

        # Inform the user on the automatically chosen variables.
        print(f"The three highest correlating variables are [{add_variable1}, {add_variable2}, {add_variable3}]" + "\n")

    # Stores the amount of additional variables the user chose.
    var_num = 0

    # If the user didn't choose any additional variables.
    if corr_answer == 3:
        clust_list = [varint_data, lat_data, long_data]

    # If the user choose only one additional variable.
    elif add_variable2 == "DO NOT CHOOSE ANY ADDITIONAL VARIABLES":
        clust_list = [varint_data, lat_data, long_data, add_variable1]
        var_num = 1

    # If the user choose two additional variable.
    elif add_variable3 == "DO NOT CHOOSE ANY ADDITIONAL VARIABLES":
        clust_list = [varint_data, lat_data, long_data, add_variable1, add_variable2]
        var_num = 2

    # If the user choose three additional variable.
    else:
        clust_list = [varint_data, lat_data, long_data, add_variable1, add_variable2, add_variable3]
        var_num = 3

    # Extracts from the dataframe the information of the selected columns.
    data_clustering_columns = import_data_copy[clust_list]

    # Copies the categorical columns into an empty dataframe.
    data_clustering = data_clustering_columns.copy()

    # Printing separator line.
    print("\n" + decor_imp + "\n")
    print("BEGINNING THE CLUSTERING PROCESS")
    print("\n" + decor_imp)
    print("The dataframe was restricted to the following columns (With 5 reference features):" + "\n")

    # Print the dataframe's first five rows.
    display(data_clustering.head())

    # If the user wanted to transform the data: First step, remove negative values.
    if box_cox_answer.lower() == "y":

        # Returns the main variable lowest row.
        varint_min = data_clustering[varint_data].min(axis=0)

        # If said row is lower or equal to 0.
        if varint_min <= 0:

            # Add the lowest value + 0.1 to make sure every value is above 0.
            data_clustering[varint_data] = data_clustering[varint_data] + ((varint_min * -1) + 0.1)

        # If the additional variable is presente, returns the lowest row.
        if var_num >= 1:
            add1_min = data_clustering[add_variable1].min(axis=0)

            # If said row is lower or equal to 0.
            if add1_min <= 0:

                # Add the lowest value + 0.1 to make sure every value is above 0.
                data_clustering[add_variable1] = data_clustering[add_variable1] + ((add1_min * -1) + 0.1)

        # If the additional variable is presente, returns the lowest row.
        if var_num >= 2:
            add2_min = data_clustering[add_variable2].min(axis=0)

            # If said row is lower or equal to 0.
            if add2_min <= 0:

                # Add the lowest value + 0.1 to make sure every value is above 0
                data_clustering[add_variable2] = data_clustering[add_variable2] + ((add2_min * -1) + 0.1)

        # If the additional variable is presente, returns the lowest row.
        if var_num == 3:
            add3_min = data_clustering[add_variable3].min(axis=0)

            # If said row is lower or equal to 0.
            if add3_min <= 0:

                # Add the lowest value + 0.1 to make sure every value is above 0.
                data_clustering[add_variable3] = data_clustering[add_variable3] + ((add3_min * -1) + 0.1)

    # If the user wants to remove outliers.
    if outlier_answer.lower() == "y":

        # Keep only values inside three standard deviations away from the mean.
        data_clustering[(np.abs(stats.zscore(data_clustering)) < 3).all(axis=1)]

    # If the user wanted to transform the data: Second step, transform the values.
    if box_cox_answer.lower() == "y":

        # Prepare the subplots for all the posible variable combinations (Up to four variables).
        if var_num == 0:
            fig, axs = plt.subplots(2, 1)
        if var_num == 1:
            fig, axs = plt.subplots(2, 2)
        if var_num == 2:
            fig, axs = plt.subplots(2, 3)
        if var_num == 3:
            fig, axs = plt.subplots(2, 4)

        # Set the position and title for the un-transformed main variable.
        ax = axs[0, 0]
        ax.set_title(f"Un-transformed {varint_data}")

        # Generate the first plot.
        sns.histplot(data_clustering[varint_data], kde=True, ax=axs[0, 0], color="firebrick")

        # Make the box-cox transformation for the main variable.
        data_clustering[varint_data], varint_lambda = stats.boxcox(data_clustering[varint_data])

        # Set the position and title for the transformed main variable.
        ax = axs[1, 0]
        ax.set_title(f"Box_Cox {varint_data}")

        # Generate the second plot.
        sns.histplot(data_clustering[varint_data], kde=True, ax=axs[1, 0], color="firebrick")

        if var_num >= 1:

            # Set the position and title for the un-transformed first additional variable.
            ax = axs[0, 1]
            ax.set_title(f"Un-transformed {add_variable1}")

            # Generate the third plot.
            sns.histplot(data_clustering[add_variable1], kde=True, ax=axs[0, 1], color="darkturquoise")

            # Make the logarithmic transformation for the first additional variable.
            data_clustering[add_variable1] = np.log1p(data_clustering[add_variable1])

            # Set the position and title for the transformed first additional variable.
            ax = axs[1, 1]
            ax.set_title(f"Log {add_variable1}")

            # Generate the fourth plot.
            sns.histplot(data_clustering[add_variable1], kde=True, ax=axs[1, 1], color="darkturquoise")

        if var_num >= 2:

            # Set the position and title for the un-transformed second additional variable.
            ax = axs[0, 2]
            ax.set_title(f"Un-transformed {add_variable2}")

            # Generate the fifth plot.
            sns.histplot(data_clustering[add_variable2], kde=True, ax=axs[0, 2], color="chartreuse")

            # Make the logarithmic transformation for the second additional variable.
            data_clustering[add_variable2] = np.log1p(data_clustering[add_variable2])

            # Set the position and title for the transformed second additional variable.
            ax = axs[1, 2]
            ax.set_title(f"Log {add_variable2}")

            # Generate the sixth plot.
            sns.histplot(data_clustering[add_variable2], kde=True, ax=axs[1, 2], color="chartreuse")

        if var_num == 3:

            # Set the position and title for the un-transformed third additional variable.
            ax = axs[0, 3]
            ax.set_title(f"Un-transformed {add_variable3}")

            # Generate the seventh plot.
            sns.histplot(data_clustering[add_variable3], kde=True, ax=axs[0, 3], color="plum")

            # Make the logarithmic transformation for the third additional variable.
            data_clustering[add_variable3] = np.log1p(data_clustering[add_variable3])

            # Set the position and title for the transformed third additional variable.
            ax = axs[1, 3]
            ax.set_title(f"Log {add_variable3}")

            # Generate the eighth plot.
            sns.histplot(data_clustering[add_variable3], kde=True, ax=axs[1, 3], color="plum")

        # Show the plot to the user.
        plt.show()

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("OPTIMAL CLUSTER NUMBER SELECTION")
    print("\n" + decor_mid)

    # Ask the user if they want to generate the elbow method function.
    print("The optimal cluster number may be estimated visually using the cluster inter or intra inertia.")
    print("If the cluster number is not yet estimated, the script may start the elbow method.")
    elbow_answer = input("Start method? (Y/N): ")

    # If the user wants to start the elbow method.
    if elbow_answer.lower() == "y":

        # Explain how the user must interpret the graph.
        print("\n" + "The elbow method will calculate a graph showing the intrainertia value using 1 to 15 clusters.")
        print("The optimal number of clusters is wherever the curve stops drastically reducing")
        print("This process may take a few minutes. Please wait.")

        # Run the elbow clustering function, which returns the posible clusters and a list containing the inercia.
        pos_clust, inertia = elbow_clustering(data_clustering)

        # Plot a graph showing the inertia as the y and the clusters as the x.
        plt.plot(pos_clust, inertia, 'bx-')

        # Label each axis and add the title.
        plt.xlabel("Posible clusters")
        plt.ylabel("Intracluster Inertia")
        plt.title("Elbow Method using Inertia")

        # Show the plot to the user.
        plt.show()

    # Printing separator line.
    print("\n" + decor_mid + "\n")
    print("RUNNING CLUSTERING ALGORITHM")
    print("\n" + decor_mid)

    # Set the cluster parameter for the kmeans algorithm.
    print("How many clusters do you wish to use in the algorithm? ")
    cluster_num = int(input("(1/15): "))

    # Set the max iteration parameter for the kmeans algorithm.
    print("\n" + "How many maximum iterations do you want the algorithm to run? [400] iterations is a good start.")
    iter_num = int(input("(200/1000): "))

    # Set the algorithm parameters.
    print("\n" + "The clustering has began. Please wait.")
    kmeans.set_params(n_clusters=cluster_num, max_iter=iter_num, n_init=10)

    # Begin the clustering algorithm and add the results into an additional column in the dataframe.
    data_clustering["KMEANS"] = kmeans.fit_predict(data_clustering)

    # Sets the plot size.
    plt.figure(figsize=(12, 12))

    # Generates a seaborn scatter plot showing the cluster distribution.
    sns.scatterplot(x=long_data, y=lat_data, data=import_data_copy, hue=kmeans.labels_, palette="Accent", legend="full")

    # Show the plot to the user.
    plt.show()

    # Add each cluster center into an empty dataframe.
    cluster_center = pd.DataFrame(kmeans.cluster_centers_, columns=clust_list)

    if box_cox_answer.lower() == "y":
        # Undo the Box-cox transformation of the main variable using the lambda value generated previously.
        data_clustering[varint_data] = inv_boxcox(data_clustering[varint_data], varint_lambda)

        # Remove the previously added value to shift the plot into positives.
        if varint_min <= 0:
            data_clustering[varint_data] = data_clustering[varint_data] + (varint_min - 0.1)

        # Undo the logarithmic transformation of the first additional value using the expm1 function.
        if var_num >= 1:
            data_clustering[add_variable1] = np.expm1(data_clustering[add_variable1])

            # Remove the previously added value to shift the plot into positives.
            if add1_min <= 0:
                data_clustering[add_variable1] = data_clustering[add_variable1] + (add1_min - 0.1)

        # Undo the logarithmic transformation of the second additional value using the expm1 function.
        elif var_num >= 2:
            data_clustering[add_variable2] = np.expm1(data_clustering[add_variable2])

            # Remove the previously added value to shift the plot into positives.
            if add2_min <= 0:
                data_clustering[add_variable2] = data_clustering[add_variable2] + (add2_min - 0.1)

        # Undo the logarithmic transformation of the third additional value using the expm1 function.
        elif var_num == 3:
            data_clustering[add_variable3] = np.expm1(data_clustering[add_variable3])

            # Remove the previously added value to shift the plot into positives.
            if add3_min <= 0:
                data_clustering[add_variable3] = data_clustering[add_variable3] + (add3_min - 0.1)

    # Printing separator line.
    print("\n" + decor_imp + "\n")
    print("CLUSTERING RESULTS")
    print("\n" + decor_imp)

    # Display the cluster center of every variable used in the clustering.
    print("First display: Cluster center values" + "\n")
    display(cluster_center)

    # Transformed clustering results.
    print("\n" + "Second display: KMeans cluster for each feature" + "\n")

    # Returns the number of rows of the dataframe.
    num_rows = data_clustering.shape[0]

    # If there are more than 500 rows, the user may only export the data into a csv file.
    if num_rows >= 500:
        print("The database contains more than 500 rows, so the data is not available for printing.")
        print("Do you want to export the information into a .csv file?")
        export_answer = input("(Y/N): ")

    # If there are less than 500 rows.
    else:
        print("Would you like to:")
        print("1. Print the information")
        print("2. Export the information")
        export_answer = int(input("(1/2): "))

        # The user may print the information on screen.
        if export_answer == 1:

            # Setting pandas parameters to be able to print any amount of rows and columns.
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            # Print the dataframe.
            print(data_clustering)

        # The user may export the information into a csv file.
        if export_answer == 2:
            export_answer = "y"

    # Export the information.
    if export_answer.lower() == "y":

        # Get the folder where the input file is found.
        export_dir = os.path.dirname(dir_origen)
        export_dir = os.path.join(export_dir, "kmeans_clustering.csv")
        # Export the dataframe directly into a csv file. Index=False removes the index generated by pandas.
        data_clustering.to_csv(export_dir, index=False)

    # End message.
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("The Clustering script has ended! Thank you for using it.")
    print("\n" + decor_imp + "\n" + decor_mid + "\n")

# Except errors to catch most of the possible errors on the script and let the user know the problem.
except IndexError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("You introduced a value out of the list range!")
    print("Please, be careful and only input the number/letters you see between parenthesis.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except TypeError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("Some non-numeric data type made its way into an operation!")
    print("Remember, the algorithm cannot work with any non-numeric data. Check twice and re-run the script.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except KeyError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("Some non-numeric data type made its way into an operation!")
    print("Remember, the algorithm cannot work with any non-numeric data. Check twice and re-run the script.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except ModuleNotFoundError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("You seem to miss some packages!")
    print("Try to install any missing modules and re-run the script.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except ImportError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("You seem to miss some packages!")
    print("Try to install any missing modules and re-run the script.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except FileNotFoundError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("The directory you are trying to find could not be located!")
    print("Please, make sure the file is in the given directory and make sure your folders don't contain blank spaces")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except PermissionError:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("The file could not be exported due to insufficient permissions in the output folder!")
    print("Please, make sure the folder is not private and that it actually exists.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")

except:
    print("\n" + decor_mid + "\n" + decor_imp + "\n")
    print("Something went wrong!")
    print("Please, check everything is in order and re-run the script.")
    print("If the problem persist use the debug tool to find where the code is failing.")
    print("\n" + decor_mid + "\n" + decor_imp + "\n")