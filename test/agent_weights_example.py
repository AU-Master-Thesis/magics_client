#!/usr/bin/env python3
"""
Example script demonstrating how to set per-agent factor weights in Magics.

This script allows you to set different factor weights for different agents
in the simulation, which can be useful for testing different behaviors
or optimizing performance for specific scenarios.
"""

import time
import json
from adaptive_gbp.magics_client.magics_client import MagicsClient
import numpy as np

# Connect to the simulation
client = MagicsClient(timeout=10_000)  # 10 second timeout

if not client.is_api_active():
    print("Activating API...")
    client.set_api_active(True)

# Get the current state to see all agents
agent_states = client.get_agent_state()
print(f"Found {len(agent_states)} agents in the simulation")

for agent_id, state in agent_states.items():
    print(f"Agent {agent_id}: Position {state['position']}")
    # Print current factor weights for each agent
    if 'factor_graph_state' in state and 'weights' in state['factor_graph_state']:
        weights = state['factor_graph_state']['weights']
        print(f"  Current weights: dynamic={weights['dynamic']}, obstacle={weights['obstacle']}, "
              f"interrobot={weights['interrobot']}, tracking={weights['tracking']}")

# Ask which mode to use
print("\nHow would you like to set weights?")
print("1. Single agent or system-wide")
print("2. Multiple agents at once (batch mode)")
mode_choice = input("Enter your choice (1 or 2): ")

if mode_choice == "1":
    # Single agent or system-wide mode
    agent_choice = input("\nEnter agent ID to modify (leave empty for system-wide change): ")
    agent_id_to_modify = None if agent_choice == "" else int(agent_choice)

    # Ask which factor weights to modify
    print("\nEnter new weights (leave empty to keep current value):")
    dynamic_input = input("Dynamic factor weight: ")
    obstacle_input = input("Obstacle factor weight: ")
    interrobot_input = input("Interrobot factor weight: ")
    tracking_input = input("Tracking factor weight: ")

    # Prepare weights dictionary
    weights = {}
    if dynamic_input:
        weights["dynamic"] = float(dynamic_input)
    if obstacle_input:
        weights["obstacle"] = float(obstacle_input)
    if interrobot_input:
        weights["interrobot"] = float(interrobot_input)
    if tracking_input:
        weights["tracking"] = float(tracking_input)

    if weights:
        # Apply the weight update
        if agent_id_to_modify is None:
            print("\nUpdating system-wide factor weights...")
        else:
            print(f"\nUpdating factor weights for agent {agent_id_to_modify}...")
        
        client.set_factor_weights(weights, agent_id_to_modify)
        print("Weight update sent successfully")
    else:
        print("No weights specified, nothing to update")

else:
    # Batch mode - update multiple agents at once
    agent_weights = {}
    
    # Ask how many agents to modify
    num_agents = int(input("\nHow many agents do you want to modify? "))
    
    for i in range(num_agents):
        agent_id = input(f"\nEnter agent ID #{i+1}: ")
        agent_id = int(agent_id)
        
        print(f"Enter weights for agent {agent_id}:")
        dynamic = float(input("Dynamic factor weight: "))
        obstacle = float(input("Obstacle factor weight: "))
        interrobot = float(input("Interrobot factor weight: "))
        tracking = float(input("Tracking factor weight: "))
        
        agent_weights[agent_id] = {
            "dynamic": dynamic,
            "obstacle": obstacle,
            "interrobot": interrobot,
            "tracking": tracking
        }
    
    # Ask about default weights
    print("\nDo you want to set default weights for all other agents? (y/n)")
    default_choice = input().lower()
    
    default_weights = None
    if default_choice == 'y':
        print("\nEnter default weights for all other agents:")
        dynamic = float(input("Dynamic factor weight: "))
        obstacle = float(input("Obstacle factor weight: "))
        interrobot = float(input("Interrobot factor weight: "))
        tracking = float(input("Tracking factor weight: "))
        
        default_weights = {
            "dynamic": dynamic,
            "obstacle": obstacle,
            "interrobot": interrobot,
            "tracking": tracking
        }
    
    # Apply the batch weight update
    print("\nUpdating weights for multiple agents...")
    client.set_batch_factor_weights(agent_weights, default_weights)
    print("Batch weight update sent successfully")

# Step the simulation to apply changes
print("\nStepping simulation to apply changes...")
client.step()
print("Step complete")

# Get updated state
updated_states = client.get_agent_state()
print("\nUpdated weights for all agents:")
for agent_id, state in updated_states.items():
    if 'factor_graph_state' in state and 'weights' in state['factor_graph_state']:
        weights = state['factor_graph_state']['weights']
        print(f"Agent {agent_id}: dynamic={weights['dynamic']}, obstacle={weights['obstacle']}, "
              f"interrobot={weights['interrobot']}, tracking={weights['tracking']}")

# Clean up
client.close()
print("Connection closed")
