import json

# Specify the path to your JSON file
json_file_path = 'D:\Python Project\export_onnx/test\witness.json'

# Open and read the JSON file
try:
    with open(json_file_path, 'r') as file:
        # Load the JSON data
        data = json.load(file)
        print("JSON data loaded successfully:")


except FileNotFoundError:
    print("The specified JSON file was not found.")
except json.JSONDecodeError:
    print("Error decoding JSON from the file.")
except Exception as e:
    print(f"An error occurred: {e}")
print(data.keys())
print(len(data['inputs'][0]))
print(len(data['pretty_elements']['outputs'][0]))