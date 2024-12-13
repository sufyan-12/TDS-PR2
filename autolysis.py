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
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
    }
    if functions:
        payload["functions"] = functions

    try:
        response = requests.post(CONFIG["OPENAI_API_URL"], headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with OpenAI API: {e}")
        return {"error": str(e)}

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
        description = df.describe(include="all").to_string()
        info_buf = StringIO()
        df.info(buf=info_buf)
        info = info_buf.getvalue()
        return description, info
    except Exception as e:
        print(f"Error describing dataset: {e}")
        return "Error in description", str(e)

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
        return df.isna().sum().to_dict()
    except Exception as e:
        print(f"Error calculating missing values: {e}")
        return {"error": str(e)}

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
        plot_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{plot_name}.png")
        plt_obj.savefig(plot_path, bbox_inches="tight")
        return f"./{plot_name}.png"
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt_obj.close()

# Function to generate correlation heatmap
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
def generate_box_plot(df: pd.DataFrame, plot_name="box_plot", columns: list[str] | None = None) -> str:
    """
    Generate and save a box plot for outlier analysis.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        plot_name (str, optional): Name of the plot file. Defaults to "box_plot".
        columns (list[str] | None, optional): Specific columns to include. Defaults to None (all numeric columns).

    Returns:
        str: Path to the saved plot.
    """
    try:
        # Select the specified columns or all numeric columns
        data = df[columns] if columns else df.select_dtypes(include=["number"])

        # Plot the box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, orient="h", palette="Set2")
        plt.title("Box Plot for Outlier Analysis", fontsize=16)
        plt.xlabel("Values", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        path = save_plot_to_file(plot_name, plt)

        return f"Plot saved to path: {path}"
    except Exception as e:
        print(f"Error generating box plot: {e}")
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
            Make sure your analysis includes the following:
            1. General description of the dataset.
            2. Descriptive statistics of the dataset.
            3. Missing values in the dataset.
            4. At least 2 plots.
            5. Potential next steps for analysis.

            **Instructions:**
            2. You must call functions to generate plots, such as correlation heatmap and box plot and include them in the analysis.
            3. Your final response should be in markdown format with plots linked, and the analysis should flow logically and engage the user.
            5. Be sure to explain your thought process in a narrative, as if telling a story, and break down your analysis clearly.

            """ 

            # Step 2: Initial prompt to OpenAI
            messages = [
                {"role": "system", "content": "You are a data analysis assistant."},
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
                    "description": "Generate and save a correlation heatmap plot for the dataset. This function creates a visual representation of the correlation matrix for numeric columns in the dataset, highlighting relationships between variables.",
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
                    "name": "generate_box_plot",
                    "description": "Generate and save a box plot for outlier analysis. This function can analyze specific columns or all numeric columns in the dataset to identify potential outliers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df": {
                                "type": "object",
                                "description": "The DataFrame object containing the dataset.",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of specific columns to include in the box plot. If not provided, the function will default to analyzing all numeric columns.",
                            },
                        },
                        "required": ["df"],
                    },
                },
            ]


            # Available functions dictionary for OpenAI interaction
            available_functions = {
                "calculate_missing_values": calculate_missing_values,
                "generate_correlation_heatmap": generate_correlation_heatmap,
                "generate_box_plot": generate_box_plot
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