import matplotlib.pyplot as plt
import time
from Nodes.ball import BallNode
from Nodes.table import TableNode

BALL_START_POINT = 5

def table_collision_callback(ball_state, table_position):
    """
    Collision callback to modify the ball's velocity if it hits the table.
    
    :param ball_state: The state dictionary of the BallNode.
    :param table_position: The height of the table.
    """
    if ball_state["position"] < table_position + 1e-6:  # Small buffer for precision issues
        ball_state["position"] = table_position
        
        # Only reverse velocity if still moving significantly
        if abs(ball_state["velocity"]) > 1e-4:  
            ball_state["velocity"] = -ball_state["velocity"] * ball_state["elasticity"]
        else:
            ball_state["velocity"] = 0  # Stop if velocity is too small


if __name__ == "__main__":
    # Initialize nodes
    ball = BallNode(name="ball", timestep=0.01)
    table = TableNode(name="table", timestep=0.01)

    ball.state['elasticity'] = 0.7
    ball.state['position'] = BALL_START_POINT

    # Register the collision callback
    ball.register_boundary_condition(lambda state: table_collision_callback(state, table.state["position"]))

    # Prepare visualization
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)  # Static x-axis
    ax.set_ylim(-1, BALL_START_POINT)  # Dynamic y-axis based on ball motion
    ball_marker, = ax.plot([0.5], [ball.state["position"]], 'ro', markersize=20)  # Single point

    plt.xlabel("Time Step")
    plt.ylabel("Ball Position")

    # Run the simulation
    for step in range(1000):
        ball.step()
        
        # Update the ball marker position (keeping x constant)
        ball_marker.set_xdata([0.5])  
        ball_marker.set_ydata([ball.state["position"]])  

        table_line, = ax.plot([0, 1], [table.state["position"], table.state["position"]], 'k-', linewidth=2)

        plt.pause(0.001)  # Pause for visualization update
        print(f"Time step {step}: Ball position = {ball.state['position']}, velocity = {ball.state['velocity']}")

    plt.show()
