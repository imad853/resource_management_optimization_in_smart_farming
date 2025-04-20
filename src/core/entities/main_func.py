
def main():
    # === Initial State ===
    print("Initializing the initial state with soil moisture, nutrients, and other factors...\n")
    
    # Instantiate the optimization problem with the initial state
    farm_problem = optimazition_problem(initial_state)
    
    # Creating a minimal Node class to represent a state and its associated costs
    print("Defining a simple Node class to represent a state and its associated costs...\n")
    class Node:
        def __init__(self, state, g=0):
            self.state = state  # The state of the farm
            self.g = g  # Cost to reach this state (g represents the cost)
            self.f = 0  # Total cost (f = g + h, where h is heuristic)
        
        # Function to create a copy of the current Node (important for tree search)
        def copy(self):
            return Node(self.state.copy(), self.g)

    # === 1. Create initial node ===
    print("1. Testing initial state:")
    root = Node(initial_state)  # Creating the root node with the initial state
    print(f"Initial state: {root.state}\n")

    # === 2. Test heuristic ===
    print("2. Heuristic value of the state:")
    h = farm_problem.heuristic(root.state)  # Calculate heuristic for the initial state
    print(f"Heuristic h(s): {h}\n")

    # === .. Test cost ===
    print("3. Cost value of the state:")
    g = farm_problem.cost(root.state)  # Calculate cost for the initial state
    print(f"Cost g(s): {g}\n")

    # === 4. Testing if the action is valid ===
    print("4. Testing if the action is valid:")
    
    # Assuming you have a method to get valid actions.
    valid_actions = farm_problem.get_valid_actions(root.state)

    # If no valid actions exist, print an appropriate message.
    if valid_actions:
        print("Valid actions are:")
        for action in valid_actions:
            print(f"Action: {action}")
    else:
        print("No valid actions available.\n")

    # === 5. Test apply_action ===
    print("5. Applying one action:")
    test_action = (0.5, 0.2)  # water amount = 0.5, fertilizer amount = 0.2
    new_state = farm_problem.apply_action(root.state.copy(), test_action)  # Apply the action to the state
    print(f"New state after action {test_action}: {new_state}\n")

    # === 6. Test expand_node ===
    print("6. Expanding node:")
    children = farm_problem.expand_node(root)  # Expand the node to generate child nodes
    for idx, child in enumerate(children):
        print(f"Child {idx+1}:")
        print(f"State: {child.state}")
        print(f"g: {child.g}, f: {child.f}")
        print()

    # === 7. Test goal ===
    print("7. Testing if it is a goal state:")

    # Perform goal test
    goal_state = farm_problem.goal_test(root.state)

    # Print whether the current state is a goal state
    if goal_state:
        print(f"The current state {root.state} is a goal state.")
    else:
        print(f"The current state {root.state} is NOT a goal state.\n")


# Check if this is the main program being executed
if __name__ == "__main__":
    main()  # Run the main function
