"""
Magics ZeroMQ Client

This module provides a client for communicating with the Magics simulation
using ZeroMQ.
"""

import time
import uuid
import json
import zmq
import numpy as np
import enum
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, TypedDict

# For compatibility with Python < 3.11 where NotRequired isn't available
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class PlanningStrategy(enum.Enum):
    """Planning strategy used by an agent."""

    ONLY_LOCAL = "OnlyLocal"
    RRT_STAR = "RrtStar"


class MissionState(enum.Enum):
    """Mission state for an agent."""

    IDLE = "Idle"
    ACTIVE = "Active"
    COMPLETED = "Completed"


# TypedDict definitions to document the structure of returned data
class CollisionInfoDict(TypedDict, total=False):
    """Information about collisions for an agent."""

    robot_collisions_total: int  # Total number of collisions with other robots
    robot_collisions_delta: (
        int  # Change in robot collisions since last state extraction
    )
    environment_collisions_total: int  # Total number of collisions with the environment
    environment_collisions_delta: (
        int  # Change in environment collisions since last state extraction
    )


class StateVectorInfoDict(TypedDict, total=False):
    """Information about a state vector (position and velocity)."""

    position: np.ndarray  # Position component of the state vector [x, y]
    velocity: np.ndarray  # Velocity component of the state vector [vx, vy]


class VariableInfoDict(TypedDict, total=False):
    """Information about a variable in the factor graph."""

    index: int  # Index of the variable
    factorgraph_id: int  # ID of the factor graph this variable belongs to
    mean: List[float]  # Mean vector of the variable [x, y, vx, vy]
    covariance: List[float]  # Covariance matrix of the variable (4x4, flattened)
    estimated_position: List[float]  # Estimated position from the variable [x, y]
    estimated_velocity: List[float]  # Estimated velocity from the variable [vx, vy]


class ObstacleFactorInfoDict(TypedDict, total=False):
    """Information about an obstacle factor in the factor graph."""

    variable_index: int  # Index of the variable this factor is connected to
    sdf_value: float  # SDF value at the position, ranges from 0.0 (free space) to 1.0 (obstacle)
    position: List[float]  # Position where the SDF value was measured [x, y]


class InterRobotFactorInfoDict(TypedDict, total=False):
    """Information about an inter-robot factor in the factor graph."""

    variable_index: int  # Index of the variable this factor is connected to
    external_robot_id: int  # ID of the external robot this factor connects to
    external_factorgraph_id: int  # ID of the external factor graph
    external_variable_index: (
        int  # Index of the variable in the external robot's factor graph
    )
    safety_distance: float  # Safety distance for collision avoidance
    distance_between_variables: float  # Current distance between estimated positions
    active: bool  # Whether the factor is active (called skip in Rust)


class TrackingFactorInfoDict(TypedDict, total=False):
    """Information about a tracking factor in the factor graph."""

    variable_index: int  # Index of the variable this factor is connected to
    tracking_path: List[List[float]]  # Path that the robot is tracking
    tracking_index: int  # Current index in the tracking path
    projected_position: List[float]  # Projected position on the path [x, y]
    path_deviation: float  # Path deviation measurement (normalized distance)
    distance_to_path: (
        float  # Distance from robot to projected point on path (in world units)
    )


class DynamicFactorInfoDict(TypedDict, total=False):
    """Information about a dynamic factor in the factor graph."""

    from_variable_index: int  # Index of the source variable
    to_variable_index: int  # Index of the destination variable
    delta_t: float  # Time step between the variables


class FactorDetailsDict(TypedDict, total=False):
    """Detailed information about factor graph components."""

    variables: List[VariableInfoDict]  # Information about variables in the factor graph
    obstacle_factors: List[ObstacleFactorInfoDict]  # Information about obstacle factors
    interrobot_factors: List[
        InterRobotFactorInfoDict
    ]  # Information about inter-robot factors
    tracking_factors: List[TrackingFactorInfoDict]  # Information about tracking factors
    dynamic_factors: List[DynamicFactorInfoDict]  # Information about dynamic factors


class MessageStatsDict(TypedDict, total=False):
    """Statistics about messages in the factor graph."""

    internal: int  # Number of internal messages
    external: int  # Number of external messages


class FactorCountsDict(TypedDict, total=False):
    """Counts of different factor types in the factor graph."""

    obstacle: int  # Number of obstacle factors
    interrobot: int  # Number of interrobot factors
    dynamic: int  # Number of dynamic factors
    tracking: int  # Number of tracking factors


class FactorWeightsDict(TypedDict, total=False):
    """Weights for different factor types in the factor graph."""

    dynamic: float  # Weight for dynamic factors (lower values, means more important)
    obstacle: float  # Weight for obstacle factors (lower values, means more important)
    interrobot: (
        float  # Weight for interrobot factors (lower values, means more important)
    )
    tracking: float  # Weight for tracking factors (lower values, means more important)


class FactorGraphStateDict(TypedDict, total=False):
    """State of a factor graph."""

    weights: FactorWeightsDict  # Current weights of the factor graph
    variable_count: int  # Number of variables in the factor graph
    factor_count: int  # Number of factors in the factor graph
    messages_sent: MessageStatsDict  # Statistics about messages sent
    messages_received: MessageStatsDict  # Statistics about messages received
    factor_counts: FactorCountsDict  # Counts of different factor types
    factor_details: FactorDetailsDict  # All factors and variables in the factor graph


class MissionProgressDict(TypedDict, total=False):
    """Mission progress information."""

    started_at: float  # Time when the mission started
    finished_at: Optional[float]  # Time when the mission finished, if completed
    active_route: int  # Index of the active route
    total_routes: int  # Total number of routes
    total_waypoints: int  # Total number of waypoints
    remaining_waypoints: int  # Number of remaining waypoints


class MissionStateDict(TypedDict, total=False):
    """Current mission state of an agent."""

    type: MissionState  # The type of mission state (IDLE, ACTIVE, COMPLETED)
    waiting_for_waypoints: NotRequired[
        bool
    ]  # Whether the agent is waiting for waypoints (only for IDLE state)


class AgentStateDict(TypedDict, total=False):
    """State of an agent in the simulation."""

    agent_id: int  # ID of the agent
    factorgraph_id: int  # ID of the factor graph
    position: np.ndarray  # Position of the agent [x, y]
    velocity: np.ndarray  # Velocity of the agent [vx, vy]
    factor_graph_state: FactorGraphStateDict  # Factor graph state of the agent
    connected_neighbors: List[int]  # IDs of connected neighbors
    mission_state: MissionStateDict  # Current mission state of the agent
    planning_strategy: PlanningStrategy  # Planning strategy used by the agent
    radius: float  # Radius of the agent
    communication_active: bool  # Whether the agent's communication is active
    communication_radius: float  # Communication radius of the agent
    target_speed: float  # Target speed of the agent
    current_waypoint_index: Optional[int]  # Index of the current waypoint
    next_waypoint: Optional[StateVectorInfoDict]  # Information about the next waypoint
    goal_point: Optional[np.ndarray]  # Position of the goal point [x, y]
    mission_progress: MissionProgressDict  # Mission progress information
    factor_details: (
        FactorDetailsDict  # Detailed information about factor graph components
    )
    collision_info: CollisionInfoDict  # Information about collisions


class EnvironmentStateDict(TypedDict, total=False):
    """State of the environment in the simulation."""

    obstacles: np.ndarray  # Positions of obstacles in the environment
    boundaries: Dict[
        str, np.ndarray
    ]  # Boundaries of the environment {'min': [x,y], 'max': [x,y]}
    total_agents: int  # Total number of agents in the environment
    agent_density_map: Optional[List[float]]  # Optional agent density map
    sdf_resolution: Optional[Tuple[int, int]]  # Optional SDF resolution (width, height)
    world_size: Optional[Tuple[float, float]]  # Optional world size (width, height)


class MagicsError(Exception):
    """Exception raised for errors from the Magics API."""

    pass


class MagicsClient:
    """
    Client for communicating with the Magics simulation using ZeroMQ.

    This client uses the ZeroMQ REQ-REP pattern to send commands to the
    simulation and receive responses.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: int = 5000,
        magics_root_dir: Optional[str] = None,
        connect_on_init: bool = True,  # Allow delaying connection
    ):
        """
        Initialize the Magics client.

        Args:
            host: Hostname or IP address of the Magics server.
            port: Port number of the Magics server.
            timeout: Timeout in milliseconds for receiving responses.
            magics_root_dir: Optional path to the root directory of the Magics Rust project installation.
                             Required if using start_sim().
            connect_on_init: If True, connect the ZMQ socket during initialization.
                             Set to False if you want to manage connection manually or delay it.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.magics_root_dir = magics_root_dir
        self.connection_string = f"tcp://{self.host}:{self.port}"
        self._connected = False
        self.verbose = False
        if connect_on_init:
            self._connect_socket()

    def _connect_socket(self):
        """Connects or reconnects the ZMQ socket, ensuring a clean state."""
        if self._connected:
            # print("Debug: Closing existing socket before reconnecting...") # Optional debug
            # Ensure the socket is properly closed before recreating
            self.socket.setsockopt(
                zmq.LINGER, 0
            )  # Discard pending messages immediately
            self.socket.close()
            self._connected = False  # Mark as disconnected

        # Always recreate the socket to ensure a fresh state
        # print("Debug: Recreating socket...") # Optional debug
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        # print(f"Debug: Attempting to connect to {self.connection_string}") # Optional debug
        try:
            self.socket.connect(self.connection_string)
            self._connected = True
            # print(f"Debug: Socket connected to {self.connection_string}") # Optional debug
        except zmq.ZMQError as e:
            self._connected = False
            # Close the newly created socket if connection fails
            self.socket.close()
            raise MagicsError(
                f"Failed to connect socket to {self.connection_string}: {e}"
            ) from e

    def _find_latest_executable(self, project_root: Path) -> Optional[Path]:
        """Finds the most recently compiled magics executable within the project root."""
        debug_path = project_root / "target" / "debug" / "magics"
        release_path = project_root / "target" / "release" / "magics"

        executables = []
        if debug_path.exists():
            executables.append((debug_path, os.path.getmtime(debug_path)))
        if release_path.exists():
            executables.append((release_path, os.path.getmtime(release_path)))

        if not executables:
            return None

        # Sort by modification time, newest first
        executables.sort(key=lambda x: x[1], reverse=True)
        return executables[0][0]  # Return the path of the newest executable

    def _send_request(self, command: str, **parameters) -> Any:
        """
        Send a request to the server and return the response.

        Args:
            command: Command to execute
            **parameters: Command parameters

        Returns:
            The response data from the server

        Raises:
            MagicsError: If the server returns an error
            zmq.ZMQError: If there's a ZMQ error
        """
        # Create request with nested command structure
        request = {"command": {"command": command}, "request_id": str(uuid.uuid4())}
        if self.verbose:
            print(f"Request: {command}")

        # Add parameters if provided
        if parameters:
            request["command"]["parameters"] = parameters

        # Convert request to JSON
        request_json = json.dumps(request)

        # debugging.
        # if command == "SetFactorWeights":
        #     print(request_json)

        # Send request
        self.socket.send_string(request_json)

        # Receive response
        try:
            response_str = self.socket.recv_string()
            response = json.loads(response_str)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise MagicsError("Timeout while waiting for response")
            raise

        # Check for errors
        if response.get("status") != "Success":
            raise MagicsError(response.get("error", "Unknown error"))

        # Return data
        return response.get("data")

    def get_agent_state(self) -> Dict[str, AgentStateDict]:
        """
        Get the state of all agents in the simulation.

        Returns:
            Dictionary mapping agent IDs to agent states with the following structure:
            {
                "agent_id": 123,                         # ID of the agent
                "factorgraph_id": 456,                   # ID of the factor graph
                "position": array([x, y]),               # 2D position of the agent
                "velocity": array([vx, vy]),             # 2D velocity of the agent
                "factor_graph_state": {                  # State of the agent's factor graph
                    "weights": {                         # Current weights of the factor graph
                        "dynamic": 1.0,                  # Weight for dynamic factors
                        "obstacle": 1.0,                 # Weight for obstacle factors
                        "interrobot": 1.0,               # Weight for inter-robot factors
                        "tracking": 1.0                  # Weight for tracking factors
                    },
                    "variable_count": 10,                # Number of variables in the factor graph
                    "factor_count": 30,                  # Number of factors in the factor graph
                    "messages_sent": {                   # Statistics about messages sent
                        "internal": 100,                 # Number of internal messages
                        "external": 50                   # Number of external messages
                    },
                    "messages_received": {               # Statistics about messages received
                        "internal": 100,                 # Number of internal messages
                        "external": 50                   # Number of external messages
                    },
                    "factor_counts": {                   # Counts of different factor types
                        "obstacle": 5,                   # Number of obstacle factors
                        "interrobot": 5,                 # Number of interrobot factors
                        "dynamic": 10,                   # Number of dynamic factors
                        "tracking": 5                    # Number of tracking factors
                    },
                    "factor_details": {...}              # Detailed information about factors (see FactorDetailsDict)
                },
                "connected_neighbors": [124, 125],       # IDs of connected neighbors
                "mission_state": {                       # Current mission state of the agent
                    "type": MissionState.IDLE,           # The mission state type (enum)
                    "waiting_for_waypoints": True        # Only for IDLE state
                },
                "planning_strategy": PlanningStrategy.ONLY_LOCAL,  # Planning strategy used
                "radius": 0.5,                          # Radius of the agent
                "communication_active": True,            # Whether communication is active
                "communication_radius": 5.0,             # Communication radius of the agent
                "target_speed": 1.0,                     # Target speed of the agent
                "current_waypoint_index": 2,             # Index of the current waypoint
                "next_waypoint": {                       # Information about the next waypoint
                    "position": array([x, y]),           # Position component
                    "velocity": array([vx, vy])          # Velocity component
                },
                "goal_point": array([x, y]),             # Position of the goal point
                "mission_progress": {                    # Mission progress information
                    "started_at": 1234.5,                # Time when the mission started
                    "finished_at": None,                 # Time when the mission finished (if completed)
                    "active_route": 0,                   # Index of the active route
                    "total_routes": 1,                   # Total number of routes
                    "total_waypoints": 10,               # Total number of waypoints
                    "remaining_waypoints": 8             # Number of remaining waypoints
                },
                "factor_details": {...},                 # Detailed information about factor graph components
                "collision_info": {                      # Information about collisions
                    "robot_collisions_total": 0,         # Total number of collisions with other robots
                    "robot_collisions_delta": 0,         # Change in robot collisions since last state extraction
                    "environment_collisions_total": 0,   # Total number of collisions with the environment
                    "environment_collisions_delta": 0    # Change in environment collisions since last extraction
                }
            }
        """
        data = self._send_request("GetAgentState")

        # Check data format
        if not data:
            raise MagicsError("No data returned from get_agent_state")

        if "type" not in data or data["type"] != "AgentStates":
            raise MagicsError(f"Invalid response format for get_agent_state: {data}")

        # Process agent states
        result = {}
        for agent_id, state in data["content"].items():
            # Convert positions and velocities to numpy arrays
            if "position" in state:
                state["position"] = np.array(state["position"], dtype=np.float32)
            if "velocity" in state:
                state["velocity"] = np.array(state["velocity"], dtype=np.float32)

            # Process planning strategy
            if "planning_strategy" in state:
                strategy_str = state["planning_strategy"]
                try:
                    state["planning_strategy"] = PlanningStrategy(strategy_str)
                except ValueError:
                    # If the value doesn't match an enum, keep the original string
                    print(f"Invalid planning strategy: {strategy_str}")

            # Process mission state
            if "mission_state" in state:
                mission_state = state["mission_state"]
                if isinstance(mission_state, dict) and "Idle" in mission_state:
                    # Handle Idle state with waiting_for_waypoints field
                    state["mission_state"] = {
                        "type": MissionState.IDLE,
                        "waiting_for_waypoints": mission_state["Idle"].get(
                            "waiting_for_waypoints", False
                        ),
                    }
                else:
                    # Handle Active and Completed states
                    try:
                        mission_type = mission_state["type"]
                        state["mission_state"] = {"type": MissionState(mission_type)}
                    except ValueError:
                        # If the value doesn't match an enum, keep the original
                        print(f"Invalid mission state: {mission_state}")

            # Convert goal_point to numpy array if present
            if "goal_point" in state and state["goal_point"] is not None:
                state["goal_point"] = np.array(state["goal_point"], dtype=np.float32)

            # Convert next_waypoint to numpy arrays if present
            if "next_waypoint" in state and state["next_waypoint"] is not None:
                if "position" in state["next_waypoint"]:
                    state["next_waypoint"]["position"] = np.array(
                        state["next_waypoint"]["position"], dtype=np.float32
                    )
                if "velocity" in state["next_waypoint"]:
                    state["next_waypoint"]["velocity"] = np.array(
                        state["next_waypoint"]["velocity"], dtype=np.float32
                    )

            result[agent_id] = state

        return result

    def get_environment_state(self) -> EnvironmentStateDict:
        """
        Get the state of the environment.

        Returns:
            Dictionary containing environment state with the following structure:
            {
                "obstacles": array([[x1, y1], [x2, y2], ...]),   # Positions of obstacles in the environment
                "boundaries": {                                   # Boundaries of the environment
                    "min": array([min_x, min_y]),                 # Minimum boundary coordinates
                    "max": array([max_x, max_y])                  # Maximum boundary coordinates
                },
                "total_agents": 10,                               # Total number of agents in the environment
                "agent_density_map": [0.1, 0.2, ...],             # Optional agent density map (if available)
                "sdf_resolution": (100, 100),                     # Optional SDF resolution (width, height) (if available)
                "world_size": (50.0, 50.0)                        # Optional world size (width, height) (if available)
            }
        """
        data = self._send_request("GetEnvironmentState")

        # Check data format
        if not data:
            raise MagicsError("No data returned from get_environment_state")

        if "type" not in data or data["type"] != "EnvironmentState":
            raise MagicsError(
                f"Invalid response format for get_environment_state: {data}"
            )

        # Process environment state
        result = data["content"]

        # Convert obstacles to numpy arrays
        if "obstacles" in result:
            result["obstacles"] = np.array(result["obstacles"], dtype=np.float32)

        # Convert boundaries to numpy arrays
        if "boundaries" in result:
            result["boundaries"] = {
                "min": np.array(result["boundaries"][0], dtype=np.float32),
                "max": np.array(result["boundaries"][1], dtype=np.float32),
            }

        return result

    def set_factor_weights(
        self, weights: FactorWeightsDict, agent_id: Optional[int] = None
    ) -> None:
        """
        Set factor graph weights.

        Args:
            weights: Dictionary mapping factor names to weights with the following structure:
                {
                    "dynamic": 1.0,      # Weight for dynamic factors
                    "obstacle": 1.0,     # Weight for obstacle factors
                    "interrobot": 1.0,   # Weight for inter-robot factors
                    "tracking": 1.0      # Weight for tracking factors
                }
            agent_id: Optional agent ID for per-agent weights. If None, the weights are applied system-wide.
                     If provided, the weights are only applied to the specified agent.
        """
        self._send_request("SetFactorWeights", weights=weights, agent_id=agent_id)

    def step(self) -> None:
        """
        Step the simulation forward by one frame.
        """
        self._send_request("Step")

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the simulation.

        This reloads the current environment, resetting all robots and the simulation state.

        Args:
            seed (Optional[int]): An optional seed for the random number generator
                                  to ensure reproducibility. If None, the simulation
                                  will reset with random initialization.
        """
        # Ensure seed is within u64 range if provided
        if seed is not None and not (0 <= seed <= 18446744073709551615):
             raise ValueError("Seed must be a valid unsigned 64-bit integer (0 to 2^64 - 1)")

        # Always include the 'seed' key in parameters, even if None,
        # to ensure the 'parameters' field exists in the JSON for this command.
        params = {"seed": seed}
        self._send_request("Reset", **params)

    def load_environment(self, name: str) -> None:
        """
        Load a specific environment by name.

        This loads a different environment configuration, replacing the current one.
        The environment must exist in the config/scenarios directory.

        Args:
            name: The name of the environment to load (e.g., "CircleExperiment", "JunctionTwoway")

        Raises:
            MagicsError: If the environment with the specified name is not found
        """
        if name == self.get_current_scenario():
            print(f"{name} environment is already loaded, reseting")
            self.reset()
        else:
            self._send_request("LoadEnvironment", name=name)

    def is_api_active(self) -> bool:
        """
        Check if the API is active.

        Returns:
            True if the API is active, False otherwise
        """
        data = self._send_request("IsApiActive")

        # Check data format
        if not data:
            raise MagicsError("No data returned from is_api_active")

        if "type" not in data or data["type"] != "Boolean":
            raise MagicsError(f"Invalid response format for is_api_active: {data}")

        return data["content"]

    def set_api_active(self, active: bool) -> None:
        """
        Set the API active state.

        Args:
            active: Whether to activate the API
        """
        self._send_request("SetApiActive", active=active)

    def send_command(self, command: str, parameters: Dict[str, Any]) -> Any:
        """
        Send a custom command to the server.

        Args:
            command: Command name
            parameters: Command parameters

        Returns:
            The response data from the server
        """
        return self._send_request(command, **parameters)

    def set_iterations_per_step(self, iterations: int) -> None:
        """
        Set the number of iterations per step.

        Controls how many simulation iterations are performed per API step command.
        Higher values result in more computation per step but may improve convergence.

        Args:
            iterations: The number of iterations per step (must be greater than 0)
                       A typical range is 1-10, with 2-5 being common for most applications.
        """
        self._send_request("SetIterationsPerStep", iterations=iterations)

    def get_simulation_hz(self) -> float:
        """
        Get the simulation Hz (frequency).

        The simulation Hz controls how frequently the physics is updated in the simulation.
        Higher values result in more frequent updates and potentially more accurate physics,
        but require more computational resources.

        Returns:
            The current simulation Hz (typically between 30 and 120)
        """
        data = self._send_request("GetSimulationHz")

        # Check data format
        if not data:
            raise MagicsError("No data returned from get_simulation_hz")

        if "type" not in data or data["type"] != "Number":
            raise MagicsError(f"Invalid response format for get_simulation_hz: {data}")

        return data["content"]

    def set_simulation_hz(self, hz: float) -> None:
        """
        Set the simulation Hz (frequency).

        The simulation Hz controls how frequently the physics is updated in the simulation.
        Higher values result in more frequent updates and potentially more accurate physics,
        but require more computational resources.

        Args:
            hz: The new Hz value (must be greater than 0)
                Typical values range from 30 (for slower but less resource-intensive simulation)
                to 120 (for more accurate physics but higher computational load)
        """
        if hz <= 0:
            raise ValueError("Hz must be greater than 0")

        self._send_request("SetSimulationHz", hz=hz)

    def get_current_scenario(self) -> Optional[str]:
        """
        Get the name of the currently loaded scenario.

        Returns:
            The name of the current scenario as a string, or None if no scenario is loaded
            or the information is unavailable.
        """
        data = self._send_request("GetCurrentScenario")

        if not data:
            # Consider returning None or raising a specific error if no data is expected
            return None
            # raise MagicsError("No data returned from get_current_scenario")

        if "type" not in data or data["type"] != "CurrentScenario":
            raise MagicsError(
                f"Invalid response format for get_current_scenario: {data}"
            )

        # The content itself is Option<String> from Rust, which becomes None or str here
        return data.get("content")

    def spawn_agent(
        self,
        initial_position: List[float],
        goal_position: List[float],
        initial_velocity: Optional[List[float]] = None,
        radius: Optional[float] = None,
        planning_strategy: Optional[str] = None,
        target_speed: Optional[float] = None,
        weights: Optional[FactorWeightsDict] = None,
    ) -> int:
        """
        Spawn a new agent in the simulation.

        Args:
            initial_position: Initial position [x, z]
            goal_position: Goal position [x, z]
            initial_velocity: Optional initial velocity [x, z] (defaults to zero)
            radius: Optional radius (defaults to config value)
            planning_strategy: Optional planning strategy ("OnlyLocal" or "RrtStar", defaults to "OnlyLocal")
            target_speed: Optional target speed (defaults to config value)
            weights: Optional custom factor weights (defaults to config values)

        Returns:
            The ID (u32) of the newly spawned agent.

        Raises:
            MagicsError: If the server returns an error or times out.
        """
        params = {
            "initial_position": initial_position,
            "goal_position": goal_position,
        }
        if initial_velocity is not None:
            params["initial_velocity"] = initial_velocity
        if radius is not None:
            params["radius"] = radius
        if planning_strategy is not None:
            params["planning_strategy"] = planning_strategy
        if target_speed is not None:
            params["target_speed"] = target_speed
        if weights is not None:
            params["weights"] = weights

        data = self._send_request("SpawnAgent", **params)

        if not data:
            raise MagicsError("No data returned from spawn_agent")
        if "type" not in data or data["type"] != "SpawnedAgentId":
            raise MagicsError(f"Invalid response format for spawn_agent: {data}")

        return data["content"]

    def remove_agent(self, agent_id: int) -> None:
        """
        Remove an agent from the simulation.

        Args:
            agent_id: The ID (u32) of the agent to remove.

        Raises:
            MagicsError: If the server returns an error.
        """
        self._send_request("RemoveAgent", agent_id=agent_id)

    def get_available_squares(self) -> List[Dict[str, Any]]:
        """
        Get all available squares defined in the current formation configuration.

        Returns:
            List of square definitions, where each square is a dictionary:
            {
                "id": str,          # Unique identifier for the square (e.g., "initial_0", "waypoint_0_1")
                "square_type": str, # "InitialPosition" or "Waypoint"
                "min": [float, float], # Minimum corner [x, y]
                "max": [float, float], # Maximum corner [x, y]
                "min_distance": Optional[float] # Minimum distance between points (if specified)
            }
        """
        data = self._send_request("GetAvailableSquares")

        if not data:
            raise MagicsError("No data returned from get_available_squares")
        if "type" not in data or data["type"] != "AvailableSquares":
            raise MagicsError(f"Invalid response format for get_available_squares: {data}")

        return data.get("content", [])

    def replan_completed_agents(
        self,
        strategy: str = "CompleteRandom",
        square_id: Optional[str] = None,
        avoid_current_square: bool = True,
    ) -> None:
        """
        Replan all completed agents with new goals based on the specified strategy.

        Args:
            strategy: The strategy to use for generating new goals.
                      Options: "CompleteRandom", "RandomSquares".
            square_id: The ID of a specific square to use when strategy is "RandomSquares".
                       If None, a random eligible square will be chosen.
            avoid_current_square: If True and using "RandomSquares" without a specific
                                  square_id, the agent's current target square will be
                                  avoided when selecting a new random square.

        Raises:
            MagicsError: If the server returns an error.
            ValueError: If the strategy is invalid.
        """
        if strategy not in ["CompleteRandom", "RandomSquares"]:
            raise ValueError(f"Invalid replan strategy: {strategy}")

        params = {
            "strategy": strategy,
            "square_id": square_id,
            "avoid_current_square": avoid_current_square,
        }
        self._send_request("ReplanCompletedAgents", **params)


    def close(self) -> None:
        """
        Close the connection cleanly, handling potential ZMQ state issues.
        """
        # print("Debug: Closing client...") # Optional debug
        socket_closed = False
        if hasattr(self, "socket") and self.socket:
            if not self.socket.closed:
                try:
                    # print("Debug: Setting LINGER to 0 and closing socket...") # Optional debug
                    self.socket.setsockopt(
                        zmq.LINGER, 0
                    )  # Ensure quick close, discard pending
                    self.socket.close()
                    socket_closed = True
                    # print("Debug: Socket closed.") # Optional debug
                except zmq.ZMQError as e:
                    # Log error but proceed to context termination if possible
                    print(
                        f"Warning: Error closing socket: {e} (errno: {e.errno}). Attempting context termination."
                    )
                    # If error is EFSM, the socket might already be unusable/closed implicitly
                    if e.errno == zmq.EFSM:
                        socket_closed = True  # Treat as closed if in wrong state
            else:
                # print("Debug: Socket already closed.") # Optional debug
                socket_closed = True
        else:
            # print("Debug: No socket attribute or socket is None.") # Optional debug
            socket_closed = True  # Nothing to close

        self._connected = False  # Mark as disconnected regardless of close success

        context_terminated = False
        if hasattr(self, "context") and self.context:
            if not self.context.closed:
                try:
                    # print("Debug: Terminating context...") # Optional debug
                    # Add a very small delay, sometimes helps ZMQ cleanup
                    # time.sleep(0.01)
                    self.context.term()
                    context_terminated = True
                    # print("Debug: Context terminated.") # Optional debug
                except zmq.ZMQError as e:
                    print(f"Warning: Error terminating context: {e}")  # Log error
            else:
                # print("Debug: Context already terminated.") # Optional debug
                context_terminated = True
        else:
            # print("Debug: No context attribute or context is None.") # Optional debug
            context_terminated = True  # Nothing to terminate

    def start_sim(
        self,
        timeout_seconds: int = 30,
        check_interval: float = 0.5,
        initial_scenario: Optional[str] = None,
    ) -> bool:
        """
        Starts the Magics simulator if it's not already running and waits for the API to become active.

        Args:
            timeout_seconds: Maximum time in seconds to wait for the API to become active.
            check_interval: Time in seconds between checks for API activity.
            initial_scenario: Optional name of the scenario to load immediately upon simulator start
                              (e.g., "JunctionTwoway", "CircleExperiment").

        Returns:
            True if the simulator API is active (either initially or after starting).

        Raises:
            ValueError: If magics_root_dir was not provided during initialization.
            FileNotFoundError: If the configured magics_root_dir doesn't exist or the executable cannot be found.
            TimeoutError: If the simulator API does not become active within the timeout period.
            MagicsError: For other API communication errors during the check.
        """
        initial_check_passed = False
        reset_socket_needed = False
        try:
            # Ensure socket is connected before the check
            if not self._connected:
                self._connect_socket()

            if self.is_api_active():
                print("Simulator API is already active.")
                initial_check_passed = True
                return True  # Already active, nothing more to do
        except MagicsError as e:
            if "Timeout" in str(e):
                # This is expected if the server isn't running.
                print(
                    f"API not active (Timeout), attempting to connect to an already started simulator..."
                )
                # Mark that we need to reset the socket after the timeout
                reset_socket_needed = True
            else:
                # Different API error during check
                raise MagicsError(f"API error checking status: {e}") from e
        except Exception as e:
            # Handle unexpected errors (e.g., ZMQ connection issues)
            raise MagicsError(f"Unexpected error checking API status: {e}") from e

        # If the initial check timed out, reset the socket connection
        if reset_socket_needed:
            print("Resetting socket connection after initial timeout...")
            self._connect_socket()  # Reconnect to clear potential bad state

        # --- Start Simulator ---
        if self.magics_root_dir is None:
            raise ValueError(
                "Magics root directory not configured. Please provide 'magics_root_dir' when creating MagicsClient."
            )

        magics_root_path = Path(self.magics_root_dir)
        if not magics_root_path.is_dir():
            raise FileNotFoundError(
                f"Configured Magics root directory does not exist or is not a directory: {self.magics_root_dir}"
            )

        latest_executable = self._find_latest_executable(magics_root_path)
        if latest_executable is None:
            raise FileNotFoundError(
                f"No compiled Magics executable found in target/debug or target/release within {self.magics_root_dir}"
            )

        print(f"Starting Magics simulator from: {latest_executable}")
        # Construct command arguments
        cmd_args = [str(latest_executable)]
        if initial_scenario:
            print(f"  with initial scenario: {initial_scenario}")
            cmd_args.extend(["--initial-scenario", initial_scenario])

        try:
            # Use Popen for non-blocking execution, redirecting stdout and stderr (no message from the sim)
            subprocess.Popen(
                cmd_args,
                cwd=magics_root_path,
                stdout=subprocess.DEVNULL,  # Redirect standard output
                stderr=subprocess.DEVNULL,  # Redirect standard error
            )
            # Give the process a moment to start up
            time.sleep(1.0)  # Increased sleep slightly to allow scenario loading
        except Exception as e:
            raise MagicsError(f"Failed to start simulator process: {e}") from e

        # --- Poll for API Activation ---
        start_time = time.monotonic()
        api_now_active = False
        while time.monotonic() - start_time < timeout_seconds:
            time.sleep(check_interval)
            try:
                if self.is_api_active():
                    print("Simulator API activated successfully.")
                    api_now_active = True
                    break  # Exit the loop immediately on success
            except MagicsError as e:
                # Server might still be initializing, or temporarily unreachable.
                if "Timeout" in str(e):
                    # If a poll attempt times out, try resetting the connection for the next try
                    # print("Debug: Poll timed out, resetting socket...") # Optional debug
                    try:
                        self._connect_socket()  # Reset connection before next poll
                    except MagicsError as conn_e:
                        print(
                            f"Warning: Failed to reconnect socket during polling after timeout: {conn_e}"
                        )
                        # Continue polling, maybe it recovers later
                else:
                    # Log other MagicsErrors during polling but continue
                    print(f"Warning: MagicsError during polling: {e}")
                pass  # Continue polling after handling MagicsError
            except zmq.ZMQError as e:
                # Catch potential ZMQ errors during polling (e.g., if server crashes)
                print(
                    f"Warning: ZMQError during polling: {e}. Attempting to reset connection."
                )
                try:
                    self._connect_socket()  # Attempt to reset connection
                except MagicsError as conn_e:
                    print(
                        f"Warning: Failed to reconnect socket after ZMQError: {conn_e}"
                    )
                    # Continue polling after failed reset attempt
                pass  # Continue polling after handling ZMQError

        if not api_now_active:
            raise TimeoutError(
                f"Simulator API did not become active within {timeout_seconds} seconds."
            )
        time.sleep(5)  # let it load
        return True  # Return True only if api_now_active is True
