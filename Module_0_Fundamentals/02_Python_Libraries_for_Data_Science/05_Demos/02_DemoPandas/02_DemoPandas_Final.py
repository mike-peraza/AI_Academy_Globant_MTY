
############################################################################################
# Python code example for reading a CSV file with 20 Students data representing theirs
# scores per subject (Math, Science, English). Doing some statistics and calculate their
# average per subject and finally plotting it to generate a visual graph.
############################################################################################
# Step 1. We first need to import the libraries this demo is going to use
# pip install pandas
import pandas as pd
# pip install matplotlib 
import matplotlib.pyplot as plt

# Step 2. Load the dataset
students_data = pd.read_csv('Module_0_Fundamentals/02_Python_Libraries_for_Data_Science/05_Demos/02_DemoPandas/students_data.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(students_data.head())

# Basic statistics
print("\nBasic statistics:")
print(students_data.describe())

# Step 3. Calculate average score for each subject
average_scores = students_data[['Math', 'Science', 'English']].mean(axis=1)

# Step 4. Plot average scores
plt.figure(figsize=(10, 6))
average_scores.plot(kind='bar', color='skyblue')
plt.title('Average Scores of Students')
plt.xlabel('Subjects')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
