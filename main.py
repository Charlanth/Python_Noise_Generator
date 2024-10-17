# Noise Generator
# Name: Charles-Émile Lanthier
# Version: alpha 0.0.0
# Last revision: 2024-10-17

# TODO: Add mountain range map
# !FIX: Fix chunk border artifacts
# TODO: Add biomes (maybe)
# TODO: Add interface to customize generation parameters (maybe)

import pygame, sys, random
from pygame.locals import QUIT

def initialize_graphics() -> pygame.Surface:
    """
    Initializes the graphical display for the game.

    This function initializes the pygame library, sets up the display window with
    dimensions based on world_size_px, and fills the screen with a black background.

    Returns:
        screen (pygame.Surface): The initialized screen surface where game elements will be rendered.
    """

    pygame.init()
    screen = pygame.display.set_mode((world_size_px, world_size_px))
    screen.fill((0, 0, 0))
    return screen

class Tile:
    """
    Represents a single tile in the game world, including its position, noise value, 
    continent association, type, color, and biome.

    Attributes:
        - x (int): X-coordinate of the tile.
        - y (int): Y-coordinate of the tile.
        - noise_val (int): Represents the noise value, indicating terrain type (e.g., land or water).
        - continent_val (int): Indicates if the tile belongs to a continent (1) or not (0).
        - tile_type (tuple): Stores the type of tile after classification (e.g., forest, mountain).
        - colour (tuple): The RGB color of the tile based on its type and biome.
        - biome (str): The specific biome type for this tile, if applicable.
    
    Methods:
        - load_tile(screen): Determines the tile type and color, and renders the tile on the screen.
        - get_tile_colour(): Returns the color associated with the tile based on predefined constants.
        - unload_tile(): Placeholder for unloading tile data (not yet implemented).
    """

    COLOR_CONSTANTS = {
        (1, 'd') : {1 : (248, 220, 172)},
        (1, 'f') : {0 : (0, 245, 0), 1 : (0, 245, 0), 2 : (0, 245, 0), 3 : (0, 245, 0), 4 : (90, 90, 150)},
        (1, 't') : {1 : (0, 180, 0)}, #Trees
        (0, 'd') : {1 : (0, 0, 255)},
        (0, 'f') : {0 : (0, 0, 200), 1 : (0, 0, 200), 2 : (0, 0, 200), 3 : (0, 0, 180), 4 : (0, 0, 180)}
    }

    def __init__(self, x, y, noise_val, continent_val):
        self.x = x
        self.y = y
        self.noise_val = noise_val
        self.continent_val = continent_val
        self.tile_type = None
        self.color = None
        self.biome = None

    def load_tile(self, screen) -> None:
        """
        Determines the tile type based on its attributes and draws the tile on the screen.
        """

        self.tile_type = get_tile_type(self)

        # Randomly generates trees on tiles with noise values of 1 (Land) 
        # that aren't considered mountains
        self.tile_type = (1, 't') if (self.tile_type[0] in range(3) and random.random() > 0.3 and 
                                    self.noise_val != 0 and self.tile_type[1] == 'f') else (self.tile_type)
        self.color = self.get_tile_colour()
        pygame.draw.polygon(screen, self.color, [(self.x, self.y),(self.x + tile_size, self.y),
                                                (self.x + tile_size, self.y - tile_size),
                                                (self.x, self.y - tile_size)])
    
    def get_tile_colour(self) -> tuple:
        """
        Retrieves the appropriate color for the tile based on its noise and type.
        """

        return Tile.COLOR_CONSTANTS[(self.noise_val, self.tile_type[1])][self.tile_type[0]]

    def unload_tile(self) -> None:
        """
        Placeholder for future functionality to unload tile data.
        """

        pass
    
class Chunk:
    """
    Represents a chunk of the game world, containing tiles and associated metadata.

    Attributes:
        x (int): The x-coordinate of the chunk in the world.
        y (int): The y-coordinate of the chunk in the world.
        ID (int): The unique identifier for the chunk.
        name (str): A descriptive name for the chunk.
        tiles (dict): A dictionary of tiles within the chunk, indexed by their coordinates.

    Methods:
        __init__(x, y, chunkID, tiles): Initializes a new Chunk instance.
        load_chunk(screen): Loads and renders all tiles in the chunk onto the given screen.
        unload_chunk(): Clears tile data when the chunk is no longer needed.
    """
    
    def __init__(self, x : int, y : int, chunkID : int, tiles : dict) -> None:
        """
        Initializes a new Chunk instance.

        Args:
            x (int): The x-coordinate of the chunk.
            y (int): The y-coordinate of the chunk.
            chunkID (int): The unique identifier for the chunk.
            tiles (dict): A dictionary of tiles contained within the chunk.
        """
        
        self.x = x
        self.y = y
        self.ID = chunkID
        self.name = f'chunk {self.ID}'
        self.tiles = tiles

    def load_chunk(self, screen : pygame.Surface) -> None:
        """
        Loads and renders all tiles in this chunk onto the specified screen.

        Args:
            screen (pygame.Surface): The surface on which to render the tiles.
        """
        
        for (x, y), tile in self.tiles.items():
            tile.load_tile(screen)

    def unload_chunk(self) -> None:
        """
        Clears the tile data from the chunk when it is no longer needed.
        """
        
        self.tiles.clear()

class World:
    """
    Represents the game world, managing chunks and maps for continents and mountains.

    Attributes:
        chunks (dict): A dictionary storing the chunks of the world, indexed by their coordinates.
        continent_map (dict): A dictionary mapping coordinates to continent tiles.
        mountain_map (dict): A dictionary mapping coordinates to mountain tiles.

    Methods:
        __init__(): Initializes a new instance of the World class, setting up empty maps for chunks and terrain features.
    """
    
    def __init__(self) -> None:
        """
        Initializes a new World instance.

        Sets up the following attributes:
        - chunks: A dictionary to store the world's chunks.
        - continent_map: A dictionary to store the continent tiles.
        - mountain_map: A dictionary to store the mountain tiles.
        """
        
        self.chunks = {}
        self.continent_map = {}
        self.mountain_map = {} 

# Constants
world = World()
tile_size = 2 #The size of a tile in pixels, needs to be atleast 1 or bigger

# The world is square shaped this is the lenght of it's sides in tiles
# Bigger worlds require more processing
world_size = 256 
chunk_size = 32 * tile_size # The size of a chunk side in pixels, by default 32 tiles or 64 pixels
world_size_px = world_size * tile_size # The size of the world sides in pixels

def compare_tile_neighbors(tile : Tile, neighbor_tile_dist=1) -> tuple:
    """
    Compares the noise value of a tile with its neighboring tiles and returns the count of matching neighbors.

    This function checks the adjacent tiles (up, down, left, right) around a given tile within the same chunk and 
    neighboring chunks, if necessary. It compares the noise value of each neighboring tile to the current tile's 
    noise value. 

    Args:
        tile (Tile): The tile whose neighbors will be compared. The tile must have 'x', 'y', and 'noise_val' attributes.
        neighbor_tile_dist (int, optional): The distance between the tile and its neighbors. 
        Defaults to 1, which means directly adjacent tiles.

    Returns:
        tuple: A tuple containing:
        - neighbor_ammount (int): The number of neighboring tiles with the same noise value as the given tile.
        - combined_noise_value (int): The noise value of the surounding land.
        - average_noise_value (int): The average noise value of the surounding noise.
    """
    
    neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0)] # Directions for neighbor Tiles (Up, Left, Down, Right)
    current_chunk_x = tile.x - (tile.x % chunk_size)
    current_chunk_y = tile.y - (tile.y % chunk_size)
    current_chunk = world.chunks.get((current_chunk_x, current_chunk_y))

    neighbor_ammount = 0
    combined_noise_value = 0
    average_noise_value = 0
    for neighbor_x, neighbor_y in neighbors:
        neighbor_tile_x = tile.x + neighbor_x * tile_size * neighbor_tile_dist
        neighbor_tile_y = tile.y + neighbor_y * tile_size * neighbor_tile_dist

        # Check if the neighbor tile is in the current chunk
        if (neighbor_tile_x // chunk_size == current_chunk_x // chunk_size and
            neighbor_tile_y // chunk_size == current_chunk_y // chunk_size):
            neighbor_tile = current_chunk.tiles.get((neighbor_tile_x, neighbor_tile_y))
        else:
            # Fetch the neighboring chunk only if the neighbor is in a different chunk
            neighbor_chunk_x = neighbor_tile_x - (neighbor_tile_x % chunk_size)
            neighbor_chunk_y = neighbor_tile_y - (neighbor_tile_y % chunk_size)
            neighbor_chunk = world.chunks.get((neighbor_chunk_x, neighbor_chunk_y))
            if neighbor_chunk:
                neighbor_tile = neighbor_chunk.tiles.get((neighbor_tile_x, neighbor_tile_y))
            else: # Ignores world border adds a neighbor to the count
                neighbor_tile = None
                neighbor_ammount += 1
                combined_noise_value += 1

        if neighbor_tile:
            combined_noise_value += neighbor_tile.noise_val
            if neighbor_tile.noise_val == tile.noise_val:
                neighbor_ammount += 1 
    if neighbor_ammount > 0:
        average_noise_value = combined_noise_value // neighbor_ammount
    return neighbor_ammount, combined_noise_value, average_noise_value

def get_tile_type(tile : Tile) -> tuple:
    """
    Determines the type of a given tile based on its neighboring tiles' noise values.

    Args:
        tile (Tile): The tile to evaluate, which must have a 'noise_val' attribute.

    Returns:
    tuple: A tuple containing:
        - An integer representing the amount of neighboring tiles of interest (1 for direct neighbors, 
            or a count of further neighbors if applicable).
        - A string indicating the type ('d' for direct, 'f' for further).
    """
    
    direct_neighbor_ammount = compare_tile_neighbors(tile)[0]
    further_neighbor_ammount = compare_tile_neighbors(tile, 2)[0]

    if further_neighbor_ammount == 4 and tile.noise_val == 1 and random.random() >= 0.5:
        further_neighbor_ammount -= 1
    
    if direct_neighbor_ammount < 4:
        return (1, 'd')
    else:
        return (further_neighbor_ammount, 'f')

def chunk_smoothen_noise(current_chunk : Chunk, repetition : int) -> None:
    """
    Smoothens the noise values of tiles in the current chunk by averaging with their neighbors.

    This function iterates over all tiles in the specified chunk and updates each tile's noise value based on 
    the noise values of its neighboring tiles, repeating the process for a given number of iterations.

    Args:
        current_chunk (Chunk): The chunk containing the tiles to be smoothed. Must have 'x', 'y', and 'tiles' attributes.
        repetition (int): The number of times to repeat the smoothing process.
    """
    
    current_chunk_x, current_chunk_y = current_chunk.x, current_chunk.y

    for _ in range(repetition):
        for y in range(current_chunk_y, current_chunk_y + chunk_size, tile_size):
            for x in range(current_chunk_x, current_chunk_x + chunk_size, tile_size):
                # Count how many neighbors are land (noise value of 1)
                neighbor_ammount, combined_noise_value, average_noise_value = compare_tile_neighbors(current_chunk.tiles.get((x, y)), 1)

                # Update the tile's noise value based on neighbors noise values
                if current_chunk.tiles.get((x, y)):
                    if (neighbor_ammount > 2 and average_noise_value == 1) or combined_noise_value > 2:
                        current_chunk.tiles[(x, y)].noise_val = 1
                    elif (neighbor_ammount > 2 and average_noise_value == 0) or combined_noise_value < 2:
                        current_chunk.tiles[(x, y)].noise_val = 0
                else:
                    print("Failed to update tile")

def generate_continent_map(continent_resolution : int, continent_ammount=3, gen_chance :float=1) -> None:
    '''
    Generates the continent map for the world, using the recursive function spread_continent.

    Args:
        continent_resolution (int): Resolution of the continent map, subdivisions of chunks reserved for continents
        continent_ammount (int): Ammount of continents to generate, they can generate stacked together
        gen_chance (int): Chance of an extra continent tile to generate, bigger values generate larger continents
    '''

    # Checks if continent was lower equal to 0 and set it back to default
    if continent_ammount <= 0:
        continent_ammount = 3
    
    keys = list(world.continent_map.keys())
    random.shuffle(keys)

    for x, y in keys:
        if world.continent_map[(y, x)].noise_val != 1:
            spread_continent(continent_resolution, y, x, gen_chance)
            continent_ammount -= 1
            if continent_ammount == 0:
                break

def spread_continent(continent_resolution : int, y : int, x : int, gen_chance : float) -> None:
    """
    Spreads the continent noise value in the continent map based on the given parameters.

    This function modifies the continent map by setting the noise value of the specified tile 
    and potentially propagating that value to adjacent tiles based on the generation chance.

    Args:
        continent_resolution (float): The resolution for continent generation, affecting spread dynamics.
        y (float): The y-coordinate of the tile in the continent map.
        x (float): The x-coordinate of the tile in the continent map.
        gen_chance (float): The chance of generating a continent tile (range: 0 to 1). If this value is less than or equal to 0, the function exits early.
    """
    
    chunk_division = chunk_size/continent_resolution
    chunk_surface_percent = chunk_division/chunk_size * 100 # Ensures the continent is proportional with different chunk sizes
    gen_chance_divider = int(25/chunk_surface_percent) if chunk_surface_percent < 25 else 1
    world.continent_map[(y, x)].noise_val = 1

    if gen_chance <= 0: # Stops the function when the value of gen_chance is lower or equal to 0
        return
    
    adjacent_tiles = [(y-chunk_division, x), (y+chunk_division, x), (y, x-chunk_division), (y, x+chunk_division)]
    random.shuffle(adjacent_tiles)

    for new_y, new_x in adjacent_tiles:
        if world.continent_map.get((new_y, new_x)) and world.continent_map[(new_y, new_x)].noise_val != 1:
            if random.random() <= gen_chance:
                world.continent_map[(new_y, new_x)].noise_val = 1
                if random.random() >= 0.3: # Randomizes the value substracted from gen_chance
                    spread_continent(continent_resolution, new_y, new_x, gen_chance - 
                                    random.uniform((0.4/gen_chance_divider/continent_resolution), 
                                                    (0.6/gen_chance_divider/continent_resolution)))
                else: # Continues with the same gen_chance
                    spread_continent(continent_resolution, new_y, new_x, gen_chance)

def generate_world(screen : pygame.Surface, world_seed : int) -> None:
    """
    Generates a world based on the provided seed and initializes its properties.

    This function creates a grid of chunks, populates each chunk with tiles based on continent noise values,
    and applies noise smoothing. The world is then rendered and saved as an image for debugging purposes.

    Args:
        screen (pygame.Surface): The surface where the world is drawn and saved.
        world_seed (int): The seed used for random number generation to ensure consistent world generation.
    """
    
    random.seed(world_seed) # Seed is initialized here to keep consistency in the generated noise
    chunk_ID = 0 # Initializes the chunk ID value

    # How many times to smoothen the noise 
    # 0 = raw noise, 1 = coarse noise, 2+ = smoother noise, etc
    # The higher the number the less smoothing it can apply to the noise 
    # and higher performance costs
    noise_smoothing_repetition = 3 
    continent_resolution = 8 # how much to divide chunks for continent details

    # Fills the continent map with placeholder tiles
    world.continent_map = {(y, x) : Tile(x, y, 0, 0) 
                            for y in range(0, world_size_px, int(chunk_size/continent_resolution)) 
                            for x in range(0, world_size_px, int(chunk_size/continent_resolution))} 
    generate_continent_map(continent_resolution, 5)

    # Fills the chunks with tiles, 
    # this is raw noise at this point 
    # only the continents have effect on the result
    for chunk_y in range(0, world_size_px, chunk_size):
        for chunk_x in range(0, world_size_px, chunk_size):
            chunk_tiles = {}
            continent_val = world.continent_map[(0, 0)].noise_val
            for y in range(0, chunk_size, tile_size):
                for x in range(0, chunk_size, tile_size):
                    y_offset = int(chunk_size / continent_resolution) * int(y / (chunk_size / continent_resolution))
                    x_offset = int(chunk_size / continent_resolution) * int(x / (chunk_size / continent_resolution))
                    continent_val = world.continent_map[(chunk_y + y_offset, chunk_x + x_offset)].noise_val
                    if continent_val == 1 and random.random() >= 0.4:
                        noise_val = 1
                    elif continent_val == 0 and random.random() >= 0.76:
                        noise_val = 1
                    else:
                        noise_val = 0
                    chunk_tiles[(chunk_x + x, chunk_y + y)] = Tile(chunk_x + x, chunk_y + y, noise_val, continent_val)
            chunk = Chunk(chunk_x, chunk_y, chunk_ID, chunk_tiles)
            world.chunks[(chunk_x, chunk_y)] = chunk
            chunk_ID += 1

    # Smoothen the noise in every chunk
    for (x, y), chunk in world.chunks.items():
        chunk_smoothen_noise(chunk, noise_smoothing_repetition)
        chunk.load_chunk(screen)

    pygame.image.save(screen, 'world.png') # Saves an image of the world to see the details and help with debugging

def main() -> None:
    """
    Main function to initialize the noise generator, generates the world, and runs the loop.

    This function handles the noise generator initialization, including setting up graphics and
    generating the world with a random seed. It then enters an event loop to manage
    user inputs and updates the display.
    """
    
    seed = random.randint(0, 100000)
    screen = initialize_graphics()
    clock = pygame.time.Clock()

    generate_world(screen, seed)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        clock.tick(30)
        pygame.display.update()

if __name__ == "__main__":
    main()