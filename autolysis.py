# /// script
# dependencies = [
#   "requests<3",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "chardet",
#   "python-dotenv"
# ]
# ///


import os
import requests
import pandas as pd
import sys
from io import StringIO
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import chardet
from dotenv import load_dotenv

load_dotenv()

# Configuration
CONFIG = {
    "OPENAI_API_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "OUTPUT_DIR": os.getcwd()  # Current working directory
}

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json",
}


# Function to call OpenAI API
def ask_openai(messages: list[dict], functions: list[dict] | None = None) -> dict:
    """
    Communicate with the OpenAI API using provided messages and functions.

    Args:
        messages (list[dict]): List of messages for the OpenAI API.
        functions (list[dict] | None): Optional list of function definitions.

    Returns:
        dict: Response from the OpenAI API or an error message.
    """
    # Create the payload to send to the OpenAI API
    payload = {
        "model": "gpt-4o-mini",  # Specify the model to use for the API request
        "messages": messages,    # Pass the list of messages for the conversation
    }
    
    # If functions are provided, include them in the payload
    if functions:
        payload["functions"] = functions

    try:
        # Send the request to the OpenAI API
        response = requests.post(CONFIG["OPENAI_API_URL"], headers=HEADERS, json=payload)
        response.raise_for_status()  # Raise an exception for any HTTP errors
        return response.json()  # Return the response from the API as a JSON object
    except requests.exceptions.RequestException as e:
        # Handle errors during the API request
        print(f"Error communicating with OpenAI API: {e}")
        return {"error": str(e)}  # Return the error as a dictionary


# Function to describe the dataset
def describe_dataset(df: pd.DataFrame) -> tuple[str, str]:
    """
    Generate descriptive statistics and dataset information.

    Args:
        df (pd.DataFrame): The dataset to describe.

    Returns:
        tuple:
            - description (str): Descriptive statistics of the dataset.
            - info (str): Detailed dataset structure and data types.
    """
    try:
        # Generate descriptive statistics for all columns in the dataset
        description = df.describe(include="all").to_string()

        # Capture detailed information about the DataFrame (e.g., data types, non-null counts)
        info_buf = StringIO()  # Buffer to hold the info string
        df.info(buf=info_buf)  # Store DataFrame info into the buffer
        info = info_buf.getvalue()  # Retrieve the string from the buffer

        return description, info  # Return both descriptions and DataFrame info
    except Exception as e:
        # Handle any errors encountered during the description process
        print(f"Error describing dataset: {e}")
        return "Error in description", str(e)  # Return an error message


# Function to calculate missing values
def calculate_missing_values(df: pd.DataFrame) -> dict:
    """
    Calculate the number of missing values for each column in the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        dict: Dictionary of missing values count for each column.
    """
    try:
        # Calculate the number of missing values (NaN) in each column
        return df.isna().sum().to_dict()
    except Exception as e:
        # Handle any errors encountered during the calculation
        print(f"Error calculating missing values: {e}")
        return {"error": str(e)}  # Return the error as a dictionary

# Function to calculate unique values in each column
def calculate_unique_values(df: pd.DataFrame) -> dict:
    """
    Calculate the number of unique values for each column in the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        dict: Dictionary of unique values count for each column.
    """
    try:
        # Calculate the number of unique values in each column
        return df.nunique().to_dict()
    except Exception as e:
        # Handle any errors encountered during the calculation
        print(f"Error calculating unique values: {e}")
        return {"error": str(e)}  # Return the error as a dictionary

# Function to save plots to file
def save_plot_to_file(plot_name: str, plt_obj: plt) -> str:
    """
    Save the given plot object to a file.

    Args:
        plot_name (str): The name of the plot file.
        plt_obj (plt): The Matplotlib plot object.

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Define the full path to save the plot (using OUTPUT_DIR from the config)
        plot_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{plot_name}.png")

        # Save the plot as a PNG file with tight bounding box to avoid clipping
        plt_obj.savefig(plot_path, bbox_inches="tight")
        
        # Return the relative path of the saved plot
        return f"./{plot_name}.png"
    except Exception as e:
        # Handle any errors encountered while saving the plot
        print(f"Error saving plot: {e}")
    finally:
        # Ensure that the plot is closed after saving
        plt_obj.close()

# Function to generate a correlation heatmap
def generate_correlation_heatmap(df: pd.DataFrame, plot_name="correlation_heatmap") -> str:
    """
    Generate and save a correlation heatmap for numerical columns in the dataset.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        plot_name (str, optional): Name of the plot file. Defaults to "correlation_heatmap".

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Select numerical columns only
        numerical_df = df.select_dtypes(include=["number"])
        if numerical_df.empty:
            return "No numerical columns found to generate correlation heatmap."

        # Calculate the correlation matrix
        correlation_matrix = numerical_df.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            linewidths=0.5, linecolor='black', vmin=-1, vmax=1
        )
        plt.title("Correlation Heatmap", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45, va="top")

        # Save the plot
        path = save_plot_to_file(plot_name, plt)
        return f"Plot saved to path: {path}"
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return "Plot could not be generated."

# Function to generate a box plot
def generate_box_plot(df: pd.DataFrame , columns: str, plot_name="box_plot") -> str:
    """
    Generate and save a box plot for outlier analysis.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        plot_name (str, optional): Name of the plot file. Defaults to "box_plot".
        columns (str):  columns to include.

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Select the specified columns or all numeric columns
        data = df[columns] if columns else df.select_dtypes(include=["number"])

        # Plot the box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, orient="h")
        plt.title(f"Box Plot for {columns}", fontsize=16)
        plt.xlabel("Values", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(["Box Plot"], loc="upper right")  # Added legend

        # Save the plot
        path = save_plot_to_file(plot_name, plt)

        return f"Plot saved to path: {path}"
    except Exception as e:
        print(f"Error generating box plot: {e}")
        return "Plot could not be generated."

# Function to generate a bar plot
def generate_bar_plot(df: pd.DataFrame, columns: str, plot_name="bar_plot") -> str:
    """
    Generate and save a bar plot for categorical data analysis.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        plot_name (str, optional): Name of the plot file. Defaults to "bar_plot".
        columns (str): Specific categorical column to include.

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Select the specified columns or all categorical columns
        data = df[columns] if columns else df.select_dtypes(include=["object"])

        # Plot the bar plot
        plt.figure(figsize=(12, 6))
        data_count = data.value_counts()
        bar_plot = data_count.plot(kind="bar", stacked=True, figsize=(10, 8))

        plt.title("Bar Plot for Categorical Data", fontsize=16)
        plt.xlabel("Categories", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        # Add legend for the bar plot
        bar_plot.legend(title="Category Values", loc="upper left", bbox_to_anchor=(1, 1))

        # Save the plot
        path = save_plot_to_file(plot_name, plt)
        return f"Plot saved to path: {path}"
    except Exception as e:
        print(f"Error generating bar plot: {e}")
        return "Plot could not be generated."

# Function to generate a scatter plot
def generate_scatter_plot(df: pd.DataFrame, x_column: str, y_column: str, plot_name="scatter_plot") -> str:
    """
    Generate and save a scatter plot for numerical data analysis between two variables.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        plot_name (str, optional): Name of the plot file. Defaults to "scatter_plot".
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis.

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Plot the scatter plot
        plt.figure(figsize=(10, 6))
        scatter_plot = sns.scatterplot(data=df, x=x_column, y=y_column, color='blue', marker='o', label="Data Points")
        plt.title(f"Scatter Plot between {x_column} and {y_column}", fontsize=16)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add legend for the scatter plot
        scatter_plot.legend(title="Points", loc="upper right")

        # Save the plot
        path = save_plot_to_file(plot_name, plt)
        return f"Plot saved to path: {path}"
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
        return "Plot could not be generated."


# Analyze the dataset and interact with OpenAI
def analyze_data(file_path):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # Step 0: Read the CSV file
        try:
            # Detect file encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
            print(f"Detected file encoding: {encoding}")

            # Read the CSV file with the detected encoding
            df = pd.read_csv(file_path, encoding=encoding)

            print("File read successfully!")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
        
        
        # Step 1: Initial prompt generation
        try:
            description, info = describe_dataset(df)

            prompt = f"""
            Here is a summary of the dataset:
            File name: {file_name}
            Data Frame Varible name: df

            **Description:**
            {description}

            **Info:**
            {info}

            Provide insights about the dataset. You can call the appropriate functions for analysis. 
            Your analysis must include:
            1. General description of the dataset.
            2. Descriptive statistics of the dataset.
            3. Missing values in the dataset.
            4. Plots Related to the dataset.
            5. Potential next steps for analysis.

            **Instructions:**
            1. You must call functions to generate plots and include them in the analysis.
            2. Your final response should be in markdown format with plots linked, and the analysis should flow logically and engage the user.
            3. Be sure to explain your thought process in a narrative, as if telling a story, and break down your analysis clearly.
            4. There are may plotting functions available to you, make sure you use then as per the need of analysis.
            """ 

            # Step 2: Initial prompt to OpenAI
            messages = [
                {"role": "system", "content": "You are a data analysis assistant agent, you have your own function calling capablities. use them to generate the analysis by returning a function call in response."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            # Step 3: Define functions for various analyses and plots
            functions = [
                {
                    "name": "calculate_missing_values",
                    "description": "Calculate the number of missing values for each column in the dataset. This helps identify columns with incomplete data that may need cleaning or preprocessing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                        },
                        "required": ["df"],
                    },
                },
                {
                    "name": "generate_correlation_heatmap",
                    "description": "Generate and save a correlation heatmap plot for the dataset. This function creates a visual representation of the correlation matrix for numeric columns in the dataset, highlighting relationships between variables. The plot includes annotations for correlation coefficients, axis labels, and a color bar.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                            "plot_name": {
                                "type": "string",
                                "description": "The name of the saved plot file. Defaults to 'correlation_heatmap'.",
                            },
                        },
                        "required": ["df"],
                    },
                },
                {
                    "name": "generate_box_plot",
                    "description": "Generate and save a box plot for outlier analysis. This function can analyze specific columns or all numeric columns in the dataset to identify potential outliers. The plot includes axis labels and annotations for better clarity, and a legend indicating the analysis type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                            "columns": {
                                "type": "string",
                                "description": "A specific columns to include in the box plot.",
                            },
                            "plot_name": {
                                "type": "string",
                                "description": "The name of the saved plot file. Defaults to 'box_plot'.",
                            },
                        },
                        "required": ["df","columns"],
                    },
                },
                {
                    "name": "generate_bar_plot",
                    "description": "Generate and save a bar plot for categorical data analysis. This function creates a bar plot to visualize the count of each category in the dataset, and includes legends for category values. The plot allows for easy comparison between different categories in the dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                            "columns": {
                                "type": "string",
                                "description": "A specific categorical columns to include.",
                            },
                            "plot_name": {
                                "type": "string",
                                "description": "The name of the saved plot file. Defaults to 'bar_plot'.",
                            },
                        },
                        "required": ["df","columns"],
                    },
                },
                {
                    "name": "generate_scatter_plot",
                    "description": "Generate and save a scatter plot to analyze the relationship between two numerical columns in the dataset. The plot includes a legend for data points and annotations for better understanding.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                            "x_column": {
                                "type": "string",
                                "description": "The column name for the x-axis.",
                            },
                            "y_column": {
                                "type": "string",
                                "description": "The column name for the y-axis.",
                            },
                            "plot_name": {
                                "type": "string",
                                "description": "The name of the saved plot file. Defaults to 'scatter_plot'.",
                            },
                        },
                        "required": ["df", "x_column", "y_column"],
                    },
                },
                {
                    "name": "calculate_unique_values",
                    "description": "Calculate the number of unique values for each column in the dataset. This helps identify columns with repeated values, categorical data, or potential issues like duplicates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset."
                            }
                        },
                        "required": ["df"]
                    }
                }
            ]

            # Available functions dictionary for OpenAI interaction
            available_functions = {
                "calculate_missing_values": calculate_missing_values,
                "generate_correlation_heatmap": generate_correlation_heatmap,
                "generate_box_plot": generate_box_plot,
                "generate_bar_plot": generate_bar_plot,
                "generate_scatter_plot": generate_scatter_plot,
                "calculate_unique_values":calculate_unique_values
            }
            print("Initial prompt generated successfully!")
        except Exception as e:
            print(f"Error generating initial prompt: {e}")
            return

        # Step 4: Process OpenAI's responses and handle function calls
        print("Calling model...")
        try:
            insights = ""
            while True:
                response = ask_openai(messages, functions=functions)
                if not response:
                    break

                response_message = response["choices"][0]["message"]
                if "function_call" in response_message:
                    # Extract the function name and arguments (if any)
                    function_name = response_message["function_call"]["name"]
                    function_args = response_message["function_call"].get("arguments", "{}")

                    try:
                        # Parse arguments string into a dictionary
                        function_args = json.loads(function_args)
                        print(function_args)
                    except json.JSONDecodeError:
                        print(f"Error decoding function arguments: {function_args}")
                        function_args = {}

                    # Look up the function in the dictionary
                    if function_name in available_functions:                            
                        print(f"Calling function: {function_name} \n")
                        try:
                            # Call the corresponding function
                            function_result = available_functions[function_name](df, **function_args)
                            # Append the function result back to the messages
                            messages.append(
                                {
                                    "role": "function",
                                    "name": function_name,
                                    "content": str(function_result),
                                }
                            )
                        except Exception as e:
                            # Log the error and append an error message to the conversation
                            error_message = f"Error executing function '{function_name}': {e}"
                            print(error_message)
                            messages.append(
                                {
                                    "role": "function",
                                    "name": function_name,
                                    "content": error_message,
                                }
                            )
                    else:
                        # Handle unknown function call
                        unknown_function_message = f"Unknown function call: {function_name}"
                        print(unknown_function_message)
                        messages.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": unknown_function_message,
                            }
                        )

                else:
                    # Final response from the model
                    insights = response_message["content"]
                    break
        except Exception as e:
            print(response)
            print(f"Error processing OpenAI response: {e}")
            insights = "Error in processing OpenAI response"


        # Step 5: Generate Markdown content
        try:
            markdown_content = (
                f"{insights} \n"
            )
        except Exception as e:
            print(f"Error generating Markdown content: {e}")
            markdown_content = insights

        # Step 6: Save to Markdown file
        try:
            output_file = os.path.join(CONFIG['OUTPUT_DIR'], "README.md")

            with open(output_file, "w") as file:
                file.write(markdown_content)
            print(f"Analysis complete! Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to Markdown file: {e}")

    except Exception as e:
        print(f"Error analyzing dataset: {e}")


# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    analyze_data(dataset_path)