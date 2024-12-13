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

PLOT_CONST = {
    'df':  None,
    'file_name': ''
}


# Function to call OpenAI API
def ask_openai(messages, functions=None):
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
        return response

# Function to describe the dataset
def describe_dataset(df):
    try:
        # Get the descriptive statistics
        description = df.describe(include="all").to_string()

        # Capture the output of df.info()
        info_buf = StringIO()
        df.info(buf=info_buf)
        info = info_buf.getvalue()
        return description, info

    except Exception as e:
        # Handle unexpected errors and return meaningful error message
        error_message = f"Error describing dataset: {str(e)}"
        print(error_message)
        return {"error": error_message}, None

# Function to calculate missing values
def calculate_missing_values(df, **kwargs):
    try:
        # Calculate missing values for each column
        missing_values = df.isna().sum()

        # Convert to dictionary for better JSON compatibility
        return missing_values.to_dict()

    except Exception as e:
        # Handle unexpected errors and return meaningful error message
        error_message = f"Error calculating missing values: {str(e)}"
        print(error_message)
        return {"error": error_message}
# Function to save plots to file
def save_plot_to_file(plot_name, plt_obj):
    """Save the plot to the file_name/plot_name.png directory."""
    try:
        #output_dir = os.path.join(CONFIG["OUTPUT_DIR"], PLOT_CONST['file_name'])
        #os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        #plot_path = os.path.join(output_dir, f"{plot_name}.png")
        plot_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{plot_name}.png")
        
        # Save the plot
        plt_obj.savefig(plot_path, bbox_inches="tight")
        return f"./{plot_name}.png"
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt_obj.close()

# Function to generate correlation heatmap plot
def generate_correlation_heatmap(df=PLOT_CONST['df'], file_name=PLOT_CONST['file_name'], plot_name="correlation_heatmap"):
    """Generate and save a correlation heatmap plot for numerical columns only."""
    try:
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=["number"])
        
        # Check if there are any numerical columns
        if numerical_df.empty:
            print("No numerical columns found in the dataset to generate a correlation heatmap.")
            return None
        
        # Calculate the correlation matrix
        correlation_matrix = numerical_df.corr()
        
        # Generate the heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True
        )
        plt.title("Correlation Heatmap")
        
        # Save the plot
        plot_path = save_plot_to_file(plot_name, plt)
        return f"plot generated and saved at {plot_path}, use this path in the mardown file"
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return "Plot could not be generated"
   
# Function to generate box plot for outlier analysis
def generate_box_plot(df=PLOT_CONST['df'], file_name=PLOT_CONST['file_name'], plot_name="box_plot", columns=None):
    """Generate and save box plots for outlier analysis, either for specified columns or all numeric columns."""
    try:
        # If columns are specified, use them; otherwise, use all numeric columns
        if columns:
            df = df[columns]
        else:
            df = df.select_dtypes(include=["number"])  # Select numeric columns

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, orient="h", palette="Set2")
        plt.title("Box Plot for Outlier Analysis")
        plot_path = save_plot_to_file(plot_name, plt)
        return f"plot generated and saved at {plot_path}, use this path in the mardown file"
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return "Plot could not be generated"

# Function to generate cluster analysis plot
def generate_cluster_plot(df=PLOT_CONST['df'], file_name=PLOT_CONST['file_name'], plot_name="cluster_analysis", n_clusters=3):
    """Generate and save a cluster analysis plot using PCA for visualization."""
    try:
        # Remove non-numeric data
        numeric_df = df.select_dtypes(include=["number"]).dropna()

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(numeric_df)
        labels = kmeans.labels_

        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df)

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            hue=labels,
            palette="Set2",
            legend="full"
        )
        plt.title(f"Cluster Analysis (n_clusters={n_clusters})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plot_path = save_plot_to_file(plot_name, plt)
        return f"plot generated and saved at {plot_path}, use this path in the mardown file"
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return "Plot could not be generated"
    
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
            global PLOT_CONST
            PLOT_CONST['df'] = df
            PLOT_CONST['file_name'] = file_name

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
                    "description": "Calculate the number of missing values for each column in the dataset.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "generate_correlation_heatmap",
                    "description": "Generate and save a correlation heatmap plot for the dataset. This function creates a heatmap of the correlation matrix of numeric columns in the dataset.",
                    "parameters": {
                        "type": "object",
                         "properties": {}
                    }
                },
                {
                    "name": "generate_box_plot",
                    "description": "Generate and save a box plot for outlier analysis. This function can analyze specific columns or all numeric columns in the dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific columns to include in the box plot. If not provided, all numeric columns will be used.",
                            },
                        },
                        "required": ["columns"],
                    },
                },
                {
                    "name": "generate_cluster_plot",
                    "description": "Generate and save a cluster analysis plot using PCA for visualization. The dataset is clustered using KMeans, and PCA is applied to reduce dimensions for plotting.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_clusters": {
                                "type": "integer",
                                "default": 3,
                                "minimum": 2,
                                "maximum": 10,
                                "description": "Number of clusters for KMeans clustering.",
                            },
                        }
                    },
                },
            ]

            # Available functions dictionary for OpenAI interaction
            available_functions = {
                "calculate_missing_values": calculate_missing_values,
                "generate_correlation_heatmap": generate_correlation_heatmap,
                "generate_box_plot": generate_box_plot,
                "generate_cluster_plot": generate_cluster_plot,
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
            #output_dir = os.path.join(CONFIG["OUTPUT_DIR"], file_name)
            #os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            #output_file = os.path.join(output_dir, "README.md")
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