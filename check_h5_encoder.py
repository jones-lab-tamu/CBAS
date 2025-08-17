import os
import sys
import h5py

def inspect_h5_file(h5_path: str):
    """
    Opens a CBAS-generated HDF5 file and inspects its metadata.
    """
    print("\n" + "="*50)
    print(f"--- Inspecting Metadata for: {os.path.basename(h5_path)} ---")
    print("="*50)

    if not os.path.exists(h5_path):
        print(f"ERROR: File not found at the provided path.")
        print(f"Path: {h5_path}")
        return

    try:
        with h5py.File(h5_path, 'r') as h5f:
            print("File opened successfully. Reading attributes...")
            
            # Check for the encoder model identifier attribute
            if 'encoder_model_identifier' in h5f.attrs:
                encoder_id = h5f.attrs['encoder_model_identifier']
                print(f"\n[SUCCESS] Found Encoder Stamp!")
                print(f"  Encoder Used: {encoder_id}")
            else:
                print(f"\n[WARNING] Encoder stamp not found.")
                print("  This .h5 file is likely from an older version of CBAS or the stamping process failed.")

            # You can add checks for other potential metadata here in the future
            # For example:
            # if 'creation_date' in h5f.attrs:
            #     print(f"  Creation Date: {h5f.attrs['creation_date']}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred while reading the file.")
        print(f"  The file may be corrupt or not a valid HDF5 file.")
        print(f"  Details: {e}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    print("CBAS HDF5 File Inspector")
    print("This script reads the metadata from a single _cls.h5 file to identify the encoder model used to create it.")
    
    # Check if a path was provided as a command-line argument
    if len(sys.argv) > 1:
        filepath_to_check = sys.argv[1]
        inspect_h5_file(filepath_to_check)
    else:
        # If not, run in a loop asking for input
        while True:
            filepath_to_check = input("\nPlease enter the FULL path to a _cls.h5 file (or type 'exit' to quit):\n> ").strip()
            
            if filepath_to_check.lower() == 'exit':
                break
            
            inspect_h5_file(filepath_to_check)

    print("\nInspection complete.")