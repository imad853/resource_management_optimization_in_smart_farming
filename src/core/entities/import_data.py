import csv
from statistics import mean

def find_min(csv_file, column_name):
    """
    Find the minimum value in a CSV column.
    Returns None if column isn't found or has no numeric values.
    """
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        numeric_values = []
        for row in reader:
            try:
                numeric_values.append(float(row[column_name]))
            except (ValueError, KeyError):
                continue
        return min(numeric_values) if numeric_values else None

def find_max(csv_file, column_name):
    """
    Find the maximum value in a CSV column.
    Returns None if column isn't found or has no numeric values.
    """
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        numeric_values = []
        for row in reader:
            try:
                numeric_values.append(float(row[column_name]))
            except (ValueError, KeyError):
                continue
        return max(numeric_values) if numeric_values else None

def calculate_average(csv_file, column_name):
    """
    Calculate the average of values in a CSV column.
    Returns None if column isn't found or has no numeric values.
    """
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        numeric_values = []
        for row in reader:
            try:
                numeric_values.append(float(row[column_name]))
            except (ValueError, KeyError):
                continue
        return mean(numeric_values) if numeric_values else None

# Example usage
if __name__ == "__main__":
    file_path = "Crop_recommendationV2.csv"  # Replace with your file
    column = "N"       # Replace with your column name
    
    print(f"Minimum: {find_min(file_path, column)}")
    print(f"Maximum: {find_max(file_path, column)}")
    print(f"Average: {calculate_average(file_path, column)}")