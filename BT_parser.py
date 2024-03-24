import xml.etree.ElementTree as ET

# class Node:
#     def __init__(self):
#         pass

#     def run(self, agent):
#         raise NotImplementedError()

# class SequenceNode(Node):
#     def __init__(self, children):
#         self.children = children

#     def run(self, agent):
#         for child in self.children:
#             if not child.run(agent):
#                 return False
#         return True

# class SelectorNode(Node):
#     def __init__(self, children):
#         self.children = children

#     def run(self, agent):
#         for child in self.children:
#             if child.run(agent):
#                 return True
#         return False

# class ConditionNode(Node):
#     def __init__(self, condition_name):
#         self.condition_name = condition_name

#     def run(self, agent):
#         condition_func = getattr(agent, self.condition_name, None)
#         if not condition_func:
#             raise ValueError(f"Condition function '{self.condition_name}' not found in the agent class.")
#         return condition_func()

# class ActionNode(Node):
#     def __init__(self, action_name):
#         self.action_name = action_name

#     def run(self, agent):
#         action_func = getattr(agent, self.action_name, None)
#         if not action_func:
#             raise ValueError(f"Action function '{self.action_name}' not found in the agent class.")
#         action_func()
#         return True

class SequenceNode():
    def __init__(self, children):
        self.children = children

    def run(self, agent):
        for child in self.children:
            if not child.run(agent):
                return False
        return True

class SelectorNode():
    def __init__(self, children):
        self.children = children

    def run(self, agent):
        for child in self.children:
            if child.run(agent):
                return True
        return False



class ConditionNode():
    def __init__(self, condition_name):
        self.condition_name = condition_name

    def run(self, agent):
        condition_func = getattr(agent, self.condition_name, None)
        if not condition_func:
            raise ValueError(f"Condition function '{self.condition_name}' not found in the agent class.")
        return condition_func()

class ActionNode():
    def __init__(self, action_name):
        self.action_name = action_name

    def run(self, agent):
        action_func = getattr(agent, self.action_name, None)
        if not action_func:
            raise ValueError(f"Action function '{self.action_name}' not found in the agent class.")
        # action_func()
        return action_func()
    



def parse_behavior_tree(file_path):
    # # Local import to avoid circular dependency
    # from simulator_vi import SwarmAgent

    tree = ET.parse(file_path)
    root = tree.getroot()
    return parse_node(root)
    

def parse_node(node):
    node_type = node.tag.lower()

    if node_type == "condition":
        condition_name = node.text.strip()
        return ConditionNode(condition_name)
    elif node_type == "action":
        action_name = node.text.strip()
        return ActionNode(action_name)
    elif node_type == "sequence":
        children = [parse_node(child) for child in node]
        return SequenceNode(children)
    elif node_type == "selector":
        children = [parse_node(child) for child in node]
        return SelectorNode(children)
    else:
        raise ValueError(f"Unknown node type: {node_type}")
    

def print_nodes(node, indent=0):
    """
    Recursively prints all nodes in the behavior tree.
    """
    indent_str = "  " * indent
    
    if isinstance(node, SequenceNode):
        print(f"{indent_str}SequenceNode:")
        for child in node.children:
            print_nodes(child, indent + 1)
    elif isinstance(node, SelectorNode):
        print(f"{indent_str}SelectorNode:")
        for child in node.children:
            print_nodes(child, indent + 1)
    elif isinstance(node, ConditionNode):
        print(f"{indent_str}ConditionNode: {node.condition_name}")
    elif isinstance(node, ActionNode):
        print(f"{indent_str}ActionNode: {node.action_name}")
    else:
        print(f"{indent_str}Unknown node type: {type(node).__name__}")

if __name__=='__main__':
    path= "BT.xml"
    root_node = parse_behavior_tree(path)
    print_nodes(root_node)
    print("Behavior Tree:")

# # import robot_behavior  # Import behavior tree or specific behaviors
# # import simulator_vi  # Import behavior tree or specific behaviors

# import xml.etree.ElementTree as ET



# class Node:
#     def run(self, agent):
#         raise NotImplementedError("Must be implemented by subclass.")


# class ConditionNode(Node):
#     def __init__(self, condition_func):
#         self.condition_func = condition_func

#     def run(self):
#         return self.condition_func(self)


# class ActionNode(Node):
#     def __init__(self, action_func):
#         self.action_func = action_func

#     def run(self):
#         return self.action_func(self)
    
# # class ConditionNode(Node):
# #     def __init__(self, condition_name):
# #         self.condition_name = condition_name

# #     def run(self, agent):
# #         condition_func = getattr(agent, self.condition_name, None)
# #         if condition_func is None:
# #             raise ValueError(f"Condition function '{self.condition_name}' not found.")
# #         return condition_func()

# # class ActionNode(Node):
# #     def __init__(self, action_name):
# #         self.action_name = action_name

# #     def run(self, agent):
# #         action_func = getattr(agent, self.action_name, None)
# #         if action_func is None:
# #             raise ValueError(f"Action function '{self.action_name}' not found.")
# #         action_func()


# class SequenceNode(Node):
#     def __init__(self, children):
#         self.children = children

#     def run(self,agent):
#         for child in self.children:
#             result = child.run(agent)
#             if not result:
#                 return False
#         return True

# # class SequenceNode(Node):
# #     def __init__(self, children):
# #         self.children = children
# #         self.current_child_index = 0  # Track the current child index

# #     def run(self, agent):
# #         # Check if we've already completed all children
# #         if self.current_child_index >= len(self.children):
# #             return True  # Indicates BT completion
        
# #         # Run the current child
# #         child_completed = self.children[self.current_child_index].run(agent)
# #         if child_completed:
# #             self.current_child_index += 1  # Move to the next child
        
# #         return False  # Indicates BT is still running


# class SelectorNode(Node):
#     def __init__(self, children):
#         self.children = children

#     def run(self,agent):
#         for child in self.children:
#             result = child.run(agent)
#             if result:
#                 return True
#         return False



# #########
    

# def parse_behavior_tree(file_path):
#     # Local import to avoid circular dependency
#     from simulator_vi import SwarmAgent
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     return parse_node(root)


# def parse_node(node):
#     node_type = node.tag.lower()

#     # if node_type == "condition":
#     #     condition_name = node.text.strip()
#     #     # Attempt to retrieve the function from the robot_behavior module
#     #     condition_func = getattr(simulator_vi.SwarmAgent, condition_name, None)
#     #     if not condition_func:
#     #         raise ValueError(f"Condition function '{condition_name}' not found in robot_behavior.")
#     #     return ConditionNode(condition_func)
#     # elif node_type == "action":
#     #     action_name = node.text.strip()
#     #     # Attempt to retrieve the function from the robot_behavior module
#     #     action_func = getattr(simulator_vi.SwarmAgent, action_name, None)
#     #     if not action_func:
#     #         raise ValueError(f"Action function '{action_name}' not found in robot_behavior.")
#     #     return ActionNode(action_func)
#     # In parse_node function
#     if node_type == "condition":
#         condition_name = node.text.strip()
#         return ConditionNode(condition_name)
#     elif node_type == "action":
#         action_name = node.text.strip()
#         return ActionNode(action_name)
#     elif node_type == "sequence":
#         children = [parse_node(child) for child in node]
#         return SequenceNode(children)
#     elif node_type == "selector":
#         children = [parse_node(child) for child in node]
#         return SelectorNode(children)
#     else:
#         raise ValueError(f"Unknown node type: {node_type}")






# if __name__=='__main__':
#     path= "BT.xml"
#     root_node = parse_behavior_tree(path)
#     # root_node.run()
