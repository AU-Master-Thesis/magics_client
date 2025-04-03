#!/usr/bin/env python3

"""
Functional test script for the MagicsClient.start_sim() method.

This script attempts to:
1. Instantiate the MagicsClient with the specified Magics root directory.
2. Call start_sim() to ensure the simulator starts and the API becomes active.
3. Perform a basic API call (is_api_active) after starting.
4. Print success or error messages.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow importing adaptive_gbp
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent # Go up two levels: test -> magics_client -> adaptive_gbp
sys.path.insert(0, str(project_root.parent)) # Add the directory containing adaptive_gbp

try:
    from adaptive_gbp.magics_client.magics_client import MagicsClient, MagicsError
except ImportError as e:
    print(f"Error importing MagicsClient: {e}")
    print("Ensure the script is run from the correct directory or the project is installed.")
    sys.exit(1)

# --- Configuration ---
# !!! IMPORTANT: Set this to the correct root path of your Magics Rust project installation !!!
MAGICS_ROOT_DIRECTORY = "/home/zartris/code/rust/magics_api/"
# Specify a scenario to test loading
TEST_SCENARIO = "JunctionTwoway" 
# --- End Configuration ---

def main():
    print("--- Testing MagicsClient.start_sim() ---")

    if not Path(MAGICS_ROOT_DIRECTORY).is_dir():
        print(f"ERROR: Magics root directory not found or invalid: {MAGICS_ROOT_DIRECTORY}")
        print("Please update the MAGICS_ROOT_DIRECTORY variable in this script.")
        sys.exit(1)

    client = None
    try:
        print(f"Instantiating MagicsClient with magics_root_dir='{MAGICS_ROOT_DIRECTORY}'...")
        # Increase timeout slightly for potentially slower startup
        client = MagicsClient(magics_root_dir=MAGICS_ROOT_DIRECTORY, timeout=10000) 

        print(f"Calling client.start_sim(initial_scenario='{TEST_SCENARIO}')...")
        # Use a longer timeout for the functional test start
        start_successful = client.start_sim(
            timeout_seconds=45, 
            check_interval=1.0,
            initial_scenario=TEST_SCENARIO 
        ) 

        if start_successful:
            print(f"\nstart_sim(initial_scenario='{TEST_SCENARIO}') reported success.")
            print("Verifying API status again...")
            if client.is_api_active():
                print("Verification successful: is_api_active() returned True.")
                print("\n--- Test Passed ---")
            else:
                print("ERROR: Verification failed: is_api_active() returned False after start_sim() succeeded.")
                print("\n--- Test Failed ---")
                sys.exit(1)
        else:
            # start_sim should raise an exception on failure, so this path might not be hit
            # unless the return value logic changes. Included for completeness.
            print("ERROR: start_sim() returned False.")
            print("\n--- Test Failed ---")
            sys.exit(1)

    except (MagicsError, ValueError, FileNotFoundError, TimeoutError) as e:
        print(f"\nERROR during test: {type(e).__name__}: {e}")
        print("\n--- Test Failed ---")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during test: {type(e).__name__}: {e}")
        print("\n--- Test Failed ---")
        sys.exit(1)
    finally:
        if client:
            print("Closing client connection...")
            client.close()

if __name__ == "__main__":
    main()
