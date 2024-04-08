import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation, Config, Window
from BT_parser import parse_behavior_tree  # Make sure this correctly parses your BT.xml
from LLAMA2_GGML import generate_BT
import math

class SwarmAgent(Agent):
    def __init__(self, images, simulation, pos, nest_pos, target_pos):
        super().__init__(images=images, simulation=simulation)
        self.root_node = parse_behavior_tree(xml_path)  # Adjust the path if necessary
        self.pos = pos
        self.nest_pos = nest_pos
        self.target_pos = target_pos  # Directly use the passed target position
        self.target_detected_flag = False
        self.obstacle_radius = 5
        self.state = "seeking"

    def update(self):
        self.root_node.run(self)


    def nothing(self):
        """
        does nothing
        """
        pass
    
    def obstacle(self):
        """
        Check for obstacle intersections
        """
        for intersection in self.obstacle_intersections(scale=self.obstacle_radius):
            return True
        return False

    def obstacle_detected(self):
        """
        Check if an obstacle is detected
        """
        if self.obstacle():
            return True
        else:
            return False

    def avoid_obstacle(self):
        """
        avoid obstacle
        """
        return True

    def search_for_target(self):
        """
        Search for target
        """
        if not self.target_detected_flag:
            return True
        else:
            return False
        
    def target_detected(self):
        """
        Check if the target is detected
        """
        distance = math.dist(self.target_pos, self.pos)
        if distance <= 10:
            self.target_detected_flag = True
            return True
        return False
    
    def target_reached(self):
        """
        Check if the target is reached
        """
        if self.target_detected_flag:
            return True
        else:
            return False

    def change_color(self):
        """
        Change color based on target detection
        """
        self.change_image(1)  # Change to the second image (green)
        return True

    def return_to_nest(self):
        """
        Return to the nest
        """
        distance = math.dist(self.nest_pos, self.pos)
        if distance <= 15:
            self.state = "completed"
            self.freeze_movement()
            return True
        return False

    def looking_for_nest(self):
        """
        Look for the nest
        """
        if self.target_detected_flag:
            return True
        else:
            return False

    def wander(self):
        """
        Implement wandering behavior
        """
        super().change_position()
        return True

    def path_clear(self):
        """
        Check if the path is clear of obstacles
        """
        return not self.obstacle()

    def form_line(self):
        """
        Form a line towards the center of the window
        """
        center_x = self.config.window.width / 2
        direction = Vector2(center_x, self.pos.y) - self.pos
        if direction.length() > 0:
            direction.scale_to_length(self.config.movement_speed)
            self.pos += direction
        return True
    
    # def nothing(self):
    #     pass

    # def obstacle(self):
    #     # Check for obstacle intersections (this method is correct as-is)
    #     for intersection in self.obstacle_intersections(scale=self.obstacle_radius):
    #         return True
    #     return False

    # def obstacle_detected(self):
    #     # Implement logic to detect obstacles
    #     if self.obstacle():
    #         return True
    #     else:
    #         return False

    # def avoid_obstacle(self):
    #     # Implement obstacle avoidance (assuming this method needs no changes)
    #     return True

    # def search_for_target(self):
    #     if not self.target_detected_flag:
    #         # self.wander()
    #         # print('looking_for_target')
    #         return True
    #     else:
    #         # self.move_to_position(self.target_pos)
    #         return False
        
    # def target_detected(self):        
    #     distance = math.dist(self.target_pos, self.pos)
    #     if distance <= 10:
    #         # self.change_color()
    #         self.target_detected_flag = True
    #         return True
    #     return False
    

    # # def move_to_position(self, target_pos=[0,0]):
    # #     # Move towards the target position
    # #     direction = target_pos - self.pos
    # #     if direction.length() > 0:
    # #         direction.scale_to_length(self.config.movement_speed)
    # #         self.pos += direction
    # #     return True


    # def target_reached(self):
    #     if self.target_detected_flag:
    #         return True
    #     else:
    #         return False

    #     # if self.target_detected_flag:
    #     #     # print(self.on_site_id())
    #     #     target_distance = self.pos.distance_to(self.target_pos)
    #     #     if target_distance <= self.config.radius:
    #     #         return True
    #     # return False

    # def change_color(self):
    #     # player = self.in_proximity_accuracy().without_distance().filter_kind(SwarmAgent).first()
    #     # if player is not None:
    #     # if self.target_detected_flag:
    #     self.change_image(1)  # Change to the second image (green)
    #     return True
    #     # else:
    #     #     self.change_image(0)  # Change to the first image (white)
    #     #     return True



    # def return_to_nest(self):
    #     distance = math.dist(self.nest_pos, self.pos)
    #     if distance <= 15:
    #         self.state = "completed"
    #         self.freeze_movement()
    #         return True
    #     return False

    # def looking_for_nest(self):
    #     # print('looking_for_nest')
    #     if self.target_detected_flag:
    #         # return self.move_to_position(self.nest_pos)
    #         # print('looking_for_nest = True')
    #         return True
    #     else:
    #         return False
    #         # return True

    # def wander(self):
    #     super().change_position()
    #     return True


    # def path_clear(self):
    #     return not self.obstacle()



    # def form_line(self):
    #     # print('form_line')
    #     # Simple line formation towards the center of the window
    #     center_x = self.config.window.width / 2
    #     direction = Vector2(center_x, self.pos.y) - self.pos
    #     if direction.length() > 0:
    #         direction.scale_to_length(self.config.movement_speed)
    #         self.pos += direction
    #     # print("form_line")
    #     return True
    

# Remaining simulation setup and execution code...

def draw_obstacle():
    x = 350
    y = 100
    simulation.spawn_obstacle("examples/images/rect_obst.png", x, y)
    simulation.spawn_obstacle("examples/images/rect_obst (1).png", y, x)

def draw_target(simulation):
    x = target_x
    y = target_y
    simulation.spawn_site("examples/images/rect.png", x, y)

def nest():
    x = nest_x
    y = nest_y
    simulation.spawn_site("examples/images/nest.png", x, y)

def load_images(image_paths):
    return [pg.image.load(path).convert_alpha() for path in image_paths]

if __name__ == '__main__':
    xml_path = "behavior_tree.xml"
    # prompt = input('What behavior would you like to generate: ')
    # generate_BT(prompt=prompt, file_name=xml_path)

    config = Config(radius=50, visualise_chunks=True, window=Window.square(500))
    simulation = Simulation(config)

    # Nest position as a Vector2 object
    nest_x, nest_y = 450, 400  # Define the nest's position
    nest_pos = Vector2(nest_x, nest_y)

    # Define the target's position
    target_x, target_y = 200, 100
    target_pos = Vector2(target_x, target_y)

    # Ensure this path is correct and load images
    agent_images_paths = ["examples/images/white.png", "examples/images/green.png"]
    loaded_agent_images = load_images(agent_images_paths)  # Load images into Pygame surfaces

    # Initialize agents with loaded images and a starting position
    for _ in range(50):
        agent = SwarmAgent(images=loaded_agent_images, simulation=simulation, pos=Vector2(nest_x, nest_y),
                           nest_pos=nest_pos, target_pos=target_pos)
        simulation._agents.add(agent)
        simulation._all.add(agent)

    draw_obstacle()
    draw_target(simulation)
    nest()

    simulation.run()