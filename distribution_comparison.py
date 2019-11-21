import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_distribution_histogram(dataframe, 
                                    column_name, 
                                    title, x_axis_label, y_axis_label,
                                    label_name,
                                    number_bins = 15):
    """
    This function generates a histogram.
    Args:
        dataframe:
        column_name: String. Name of the column whose distribution we
        want to visualize.
        title: String. Title of the histogram.
        x_axis_label: String. X-axis label.
        y_axis_label: String. Y-axis label.
    Outputs:
        Histogram containing distribution for specific column column_name.
    """
    plt.hist(dataframe[column_name], bins = number_bins, label = label_name)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(loc='upper right')
    
def mann_whitney_u_test(distribution_1, distribution_2):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    Args:
       distribution_1: List. First distribution that we want to compare.
       distribution_2: List. Second distribution that we want to compare.
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float. p-value for the test.
    """
    u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
    #Print the results
    print('U-Statistic: ', u_statistic)
    print('p-value: ', p_value)
    return u_statistic, p_value

if __name__ == "__main__" :
    #Read in the cancer data set
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                     header = None)
    #Declare the column names of the data set
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                  'native-country', 'salary']
    generate_distribution_histogram(df, 'age',
                                    title = 'Age Distribution: US Population',
                                    x_axis_label = 'Age (years)',
                                    y_axis_label = 'Frequency',
                                    label_name = 'Age')
    #Generate histograms based on attributes
    df_less_than_50k = df[df['salary'] == ' <=50K']
    df_greater_than_50k = df[df['salary'] == ' >50K']
    generate_distribution_histogram(df_less_than_50k, 'age',
                                    title = 'Age Distribution: US Population',
                                    x_axis_label = 'Age (years)',
                                    y_axis_label = 'Frequency',
                                    label_name = '<=$50K')
    generate_distribution_histogram(df_greater_than_50k, 'age',
                                    title = 'Age Distribution: US Population',
                                    x_axis_label = 'Age (years)',
                                    y_axis_label = 'Frequency',
                                    label_name = '>$50K')
    #Perform the Mann-Whitney U Test on the two distributions
    mann_whitney_u_test(list(df_greater_than_50k['age']), list(df_less_than_50k['age']))