import numpy as np

class Node:
    def __init__(self, x, y, cost, parent_key):
        self.x, self.y = x, y
        self.cost = cost
        self.parent_key = parent_key

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_key)

class A_star():
    def __init__(self, obstacles, boundary, resolution):
        '''
        obstacles: [[minx,maxx,miny,maxy], ...]
        boundary: [minx,maxx,miny,maxy]
        resolution: grid size. Distance between two points
        '''
        xmin = boundary[0]
        xmax = boundary[1]
        ymin = boundary[2]
        ymax = boundary[3]
        self.x_indices=np.arange(xmin, xmax + resolution, resolution)
        self.y_indices=np.arange(ymin, ymax + resolution, resolution)
        self.resolution = resolution
        self.obstacles = []
        
        obsx, obsy = [], []
        for obs in obstacles:
            ox, oy = self.create_obstacle_boundary(obs, resolution)
            obsx, obsy = obsx + ox, obsy + oy

        for i in range(len(obsx)):
            self.obstacles.append([obsx[i], obsy[i]])

        obsx, obsy = self.create_obstacle_boundary(boundary, resolution)
        for i in range(len(obsx)):
            self.obstacles.append([obsx[i], obsy[i]])
        
    def point_x(self, x):
        return self.x_indices[np.argmin(np.power(self.x_indices-x,2))]

    def point_y(self, y):
        return self.y_indices[np.argmin(np.power(self.y_indices-y,2))]

    def create_obstacle_boundary(self, obstacle, resolution):
        x_min, x_max, y_min, y_max = self.point_x(obstacle[0]), self.point_x(obstacle[1]), self.point_y(obstacle[2]), self.point_y(obstacle[3])
        ox, oy = [], []
        
        #left side
        ox.append(x_min)
        oy.append(y_min)
        while oy[-1]+resolution <= y_max:
            ox.append(x_min)
            oy.append(self.point_y(oy[-1]+resolution))
        
        if oy[-1] < y_max:
            ox.append(x_min)
            oy.append(y_max)

        #up and down sides
        while ox[-1]+resolution <= x_max:
            ox.append(self.point_x(ox[-1]+resolution))
            oy.append(y_max)
            ox.append(ox[-1])
            oy.append(y_min)
        
        #right side
        ox.append(x_max)
        oy.append(y_min)
        while oy[-1]+resolution <= y_max:
            ox.append(x_max)
            oy.append(self.point_y(oy[-1]+resolution))
        
        if oy[-1] < y_max:
            ox.append(x_max)
            oy.append(y_max)
        
        return ox, oy

    def collision(self, node):
        if [node.x, node.y] in self.obstacles:
            return True
        else:
            False

    def best_node(self, open_nodes, goal):
        best = object
        min_cost = np.finfo(float).max

        for key in open_nodes.keys():
            curr = np.array([open_nodes[key].x, open_nodes[key].y])
            total_cost = open_nodes[key].cost + np.linalg.norm(curr-goal)
            if total_cost < min_cost:
                min_cost = total_cost
                best = open_nodes[key]
        return best, min_cost

    def move(self, node):
        nodes = []

        x = self.point_x(node.x + self.resolution)
        y = self.point_y(node.y)
        nodes.append(Node(x,y, node.cost + self.resolution, (node.x, node.y)))

        x = self.point_x(node.x - self.resolution)
        y = self.point_y(node.y)
        nodes.append(Node(x,y, node.cost + self.resolution, (node.x, node.y)))

        x = self.point_x(node.x)
        y = self.point_y(node.y + self.resolution)
        nodes.append(Node(x,y, node.cost + self.resolution, (node.x, node.y)))

        x = self.point_x(node.x)
        y = self.point_y(node.y - self.resolution)
        nodes.append(Node(x,y, node.cost + self.resolution, (node.x, node.y)))

        x = self.point_x(node.x + self.resolution)
        y = self.point_y(node.y + self.resolution)
        nodes.append(Node(x,y, node.cost + np.sqrt(2 * self.resolution**2), (node.x, node.y)))

        x = self.point_x(node.x - self.resolution)
        y = self.point_y(node.y - self.resolution)
        nodes.append(Node(x,y, node.cost + np.sqrt(2 * self.resolution**2), (node.x, node.y)))

        x = self.point_x(node.x + self.resolution)
        y = self.point_y(node.y - self.resolution)
        nodes.append(Node(x,y, node.cost + np.sqrt(2 * self.resolution**2), (node.x, node.y)))

        x = self.point_x(node.x - self.resolution)
        y = self.point_y(node.y + self.resolution)
        nodes.append(Node(x,y, node.cost + np.sqrt(2 * self.resolution**2), (node.x, node.y)))

        return nodes
        
    def plan(self, start, goal):
        sx = start[0]
        sy = start[1]
        self.open_nodes = dict()
        self.closed_nodes = dict()
        self.ordered_nodes = []
        self.open_nodes[(self.point_x(sx), self.point_y(sy))] = Node(self.point_x(sx), self.point_y(sy), 0, None)
        goal_key = (self.point_x(goal[0]),self.point_y(goal[1]))
        solved = False

        while not solved:
            best_node, min_cost = self.best_node(self.open_nodes, goal)
            best_key = (best_node.x, best_node.y)
            # Add the expanded node to closed list and remove from open list
            self.closed_nodes[best_key] = best_node
            del self.open_nodes[best_key]
            # Stopping criterion: Check if best node is the goal 
            if best_key == goal_key:
                solved = True
                break
            #Generate new nodes and attemt to add or replace 
            new_nodes = self.move(best_node)
            for node in new_nodes:
                key = (self.point_x(node.x), self.point_y(node.y))                
                if not self.collision(node):
                    # If key is in the closed list then attempt to replace
                    # Else attemp to insert or replace in open list      
                    if key in self.closed_nodes:
                        if self.closed_nodes[key].cost > node.cost:
                            self.closed_nodes[key] = node

                    elif key in self.open_nodes and self.open_nodes[key].cost > node.cost:
                        self.open_nodes[key] = node
                        self.ordered_nodes.append(node)
                        
                    elif not key in self.open_nodes:
                        self.open_nodes[key] = node
                        self.ordered_nodes.append(node)
        #traceback: 
        path = []
        key = (self.point_x(goal[0]), self.point_y(goal[1]))
        while key is not None:
            path = [[self.closed_nodes[key].x, self.closed_nodes[key].y]] + path
            key = self.closed_nodes[key].parent_key
        return path

    #Debug purpose
    def get_all_nodes(self):
        return self.ordered_nodes

def view_planning():
    import matplotlib.pyplot as plt
    obstacles = [[-4, -1, -3, 10], [2.5, 5.0,-8,5]] #Rectangular obstackles [xmin, xmax, y min, ymax]
    boundary = [-10, 10, -10, 10] # [xmin, xmax, ymin, ymax]
    planner = A_star(obstacles, boundary, resolution=0.8)
    start, goal = [-7,0], [7,5]
    path = planner.plan(start, goal)
    print ("Planned path: ", path)
    all_nodes = planner.get_all_nodes()

    plt.plot([d[0] for d in planner.obstacles], [d[1] for d in planner.obstacles], 'Dk')
    plt.plot(start[0], start[1], '*b', alpha=1.0, label='start', markersize=12)
    plt.plot(goal[0], goal[1], '*r', alpha=1.0, label='goal', markersize=12)

    x = []
    y = []
    for node in all_nodes:
        x.append(node.x)
        y.append(node.y)
        plt.plot(node.x, node.y, '.b')
        plt.pause(0.000000001)
 
    plt.plot([d[0] for d in path], [d[1] for d in path], '-', linewidth=3)
    plt.pause(0.000000001)
    plt.show()

if __name__=='__main__':
    view_planning()