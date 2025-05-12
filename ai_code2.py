import json

# Your list of strings
my_list = ["Hello world", "This is the second string", "And a third one.", "String with numbers 123"]
filename = "my_list_data.json"

# --- SAVING the list ---
try:
    with open(filename, 'w', encoding='utf-8') as f:
        # Use json.dump to write the list to the file
        # indent=4 makes the file nicely formatted and readable
        json.dump(my_list, f, indent=4)
    print(f"List successfully saved to {filename}")
except IOError as e:
    print(f"Error saving file: {e}")

# --- LOADING the list later ---
loaded_list = []
try:
    with open(filename, 'r', encoding='utf-8') as f:
        # Use json.load to read the data back into a Python list
        loaded_list = json.load(f)
    print(f"List successfully loaded from {filename}:")
    print(loaded_list)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except json.JSONDecodeError:
    print(f"Error: File '{filename}' does not contain valid JSON.")
except IOError as e:
    print(f"Error loading file: {e}")

# Verify it's the same
# assert my_list == loaded_list # This will be true if loading worked