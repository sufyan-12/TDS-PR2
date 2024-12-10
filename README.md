# Data Analysis and Visualization Tool

This Python script allows for in-depth analysis and visualization of CSV datasets. It integrates OpenAI's GPT-4 for intelligent insights, generates various types of plots, and produces a comprehensive report in markdown format. The analysis includes descriptive statistics, missing value analysis, outlier detection, and clustering, with results saved in an easily accessible directory.

## Features

- **Dataset Description**: Summarizes the dataset with descriptive statistics and basic metadata (such as data types, missing values, etc.).
- **Missing Values Analysis**: Identifies columns with missing data and provides a summary.
- **Visualization**:
  - **Correlation Heatmap**: Visualizes correlations between numerical features.
  - **Box Plot**: Helps identify outliers in numerical columns.
  - **Cluster Analysis Plot**: Uses KMeans clustering and PCA to group similar data points and visualize them in a 2D space.
- **Markdown Report**: Generates a markdown report with insights, plots, and suggestions for further analysis.
  
## Dependencies

This script requires the following Python libraries:

- `requests`: For making HTTP requests to the OpenAI API.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For generating plots.
- `seaborn`: For enhanced data visualization (heatmaps, box plots).
- `scikit-learn`: For clustering (KMeans) and dimensionality reduction (PCA).
- `chardet`: For detecting file encoding when reading CSV files.
- `python-dotenv`: For managing environment variables securely.

## Prerequisites

Before running the script, make sure to:

1. Install the necessary Python libraries by running:

    ```bash
    pip install -r requirements.txt
    ```

2. Set up the environment variable for OpenAI API access. Create a `.env` file in the same directory as the script and add your `AIPROXY_TOKEN`:

    ```
    AIPROXY_TOKEN=your_api_token_here
    ```

## Usage

### Running the Script

To run the script, use the following command:

```bash
python autolysis.py <dataset.csv>
```

Where `<dataset.csv>` is the path to the CSV file you want to analyze.

### Process Flow

1. **Read Dataset**: The script reads the CSV file, detects the file encoding, and loads it into a pandas DataFrame.
2. **Initial Analysis**: It provides a summary of the dataset including descriptive statistics and metadata.
3. **OpenAI Interaction**: The script sends a detailed dataset description to OpenAI's GPT-4, asking for insights and suggesting further analysis steps.
4. **Function Calls**: Based on the response from GPT-4, the script automatically calls functions to generate missing value reports, correlation heatmaps, box plots for outlier detection, and KMeans clustering plots.
5. **Plot Generation**: The relevant plots are saved to the output directory.
6. **Markdown Report**: A comprehensive markdown report is created with analysis results, visualizations, and insights, which is saved in a folder named after the dataset.

### Example

Running the script on a sample dataset:

```bash
python autolysis.py sample_data.csv
```

After running the script, a `README.md` file will be generated in the folder named `sample_data`. This markdown file will contain:

- A summary of the dataset's descriptive statistics.
- Insights from GPT-4 about the dataset, including any suggested analyses.
- Links to the generated plots, including:
  - Correlation heatmap.
  - Box plot for outlier analysis.
  - Cluster analysis plot.

## Next Steps

- Explore additional preprocessing steps (e.g., handling missing data).
- Dive deeper into any identified clusters for segmentation analysis.
- Perform advanced analyses such as predictive modeling based on the dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project and instructions on how to use the script, ensuring that users can easily understand how to execute the analysis and interpret the results.