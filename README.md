# Noise Generator for World Generation

This project implements a noise-based world generation system using Python and Pygame. It generates chunks of land with customizable noise, continents, and tile types, rendering them as graphical representations on a screen. The noise is generated using Python's random function without relying on external noise libraries.

This project was primarily created as a personal challenge to practice Python and experiment with procedural generation.

## Features

- **World Generation**: Generates a grid of tiles with noise-based values, creating land and water.
- **Continent Spreading**: Randomly spreads continents across the world based on specified resolutions.
- **Noise Smoothing**: Provides options to smooth the generated noise for a more natural look.
- **Tile Types**: Dynamically assigns different tile types and colors based on noise values.
- **Interactive Visualization**: Uses Pygame to render and visualize the generated world.

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Noise_Generator.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Noise_Generator
    ```
3. Install dependencies (if any):
    ```bash
    pip install pygame
    ```
## Usage

1. Run the `main.py` script to generate a world:
    ```bash
    python main.py
    ```
2. The world will be displayed using Pygame, and a screenshot of the generated world will be saved as `world.png`. I left an exemple of a generated world/map in the project files.

## Customization

- **Noise Smoothing**: Modify the `noise_smoothing_repetition` variable in `generate_world()` to control the smoothness of the noise.
- **Continent Resolution**: Adjust the `continent_resolution` to control the chunk subdivision for continent details which can also affect the continent sizes.

