import pygame
import random

pygame.init()

# Screen dimensions
screen_width = 600
screen_height = 400

# Color definitions
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

# Create the game window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

# Snake initial position and size
snake_block = 10
snake_speed = 15

# Food position
food_x = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
food_y = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0

# Snake initial position
x1 = screen_width / 2
y1 = screen_height / 2

# Function to draw the snake
def snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, green, [x[0], x[1], snake_block, snake_block])

# Game loop
game_over = False
game_close = False

while not game_over:
    while game_close == True:
        screen.fill(black)
        font_style = pygame.font.SysFont(None, 25)
        message = font_style.render("You Lost! Press Q-Quit or C-Play Again", True, red)
        screen.blit(message, [screen_width / 6, screen_height / 3])
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_over = True
                    game_close = False
                if event.key == pygame.K_c:
                    game_close = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x1_change = -snake_block
                y1_change = 0
            elif event.key == pygame.K_RIGHT:
                x1_change = snake_block
                y1_change = 0
            elif event.key == pygame.K_UP:
                y1_change = -snake_block
                x1_change = 0
            elif event.key == pygame.K_DOWN:
                y1_change = snake_block
                x1_change = 0

    if x1 >= screen_width or x1 < 0 or y1 >= screen_height or y1 < 0:
        game_close = True

    x1 += x1_change
    y1 += y1_change

    screen.fill(black)
    pygame.draw.rect(screen, red, [food_x, food_y, snake_block, snake_block])
    snake_Head = []
    snake_Head.append(x1)
    snake_Head.append(y1)
    snake_List.append(snake_Head)
    if len(snake_List) > length_of_snake:
        del snake_List[0]

    for x in snake_List[:-1]:
        if x == snake_Head:
            game_close = True

    snake(snake_block, snake_List)
    pygame.display.update()

    if x1 == food_x and y1 == food_y:
        food_x = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
        food_y = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0
        length_of_snake += 1

    pygame.display.update()

    pygame.time.Clock().tick(snake_speed)

pygame.quit()
quit()