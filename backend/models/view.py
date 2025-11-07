import numpy as np

try:
    # Load the .npy file (trusted source)
    data = np.load('species_labels.npy', allow_pickle=True)

    print("✅ File loaded successfully!\n")

    # If the data is an object (like list, dict, or tuple), handle separately
    if isinstance(data, np.ndarray) and data.shape != ():
        print("Data contents:\n", data)
        print("\n------------------------------")
        print(f"Shape       : {data.shape}")
        print(f"Data type   : {data.dtype}")
        print(f"Total items : {len(data)}")
        print("------------------------------")

    else:
        # Handle object or scalar content
        print("Data is a Python object, not a numeric array.\n")
        obj = data.item() if hasattr(data, "item") else data
        print("Object contents:\n", obj)
        print("\n------------------------------")
        print(f"Type        : {type(obj)}")
        if hasattr(obj, "__len__"):
            print(f"Total items : {len(obj)}")
        print("------------------------------")

except FileNotFoundError:
    print("❌ Error: File 'species_labels.npy' not found. Please check the file path.")
except ValueError as e:
    print(f"❌ ValueError: {e}")
except Exception as e:
    print(f"⚠️ Unexpected error: {e}")
