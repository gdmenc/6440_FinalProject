import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys

from src.core.node import Node
from src.core.skeleton import Skeleton
from src.solver.jacobian_ik import JacobianIK
from src.input.commander import Commander

WINDOW_SIZE = (1024, 768)
FPS = 60

def draw_grid():
    """
    Draws a simple floor grid for spatial reference.
    """
    glBegin(GL_LINES)
    glColor3f(0.3, 0.3, 0.3)
    for i in range(-20, 21, 2):
        glVertex3f(i, 0, -20)
        glVertex3f(i, 0, 20)
        glVertex3f(-20, 0, i)
        glVertex3f(20, 0, i)
    glEnd()

def draw_target(pos):
    """
    Draws a green wireframe box at the target position.
    """
    if pos is None:
        return

    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(0.0, 1.0, 0.0)

    size = 0.5
    glBegin(GL_LINE_LOOP)
    glVertex3f(-size, 0, 0)
    glVertex3f(0, size, 0)
    glVertex3f(size, 0, 0)
    glVertex3f(0, -size, 0)
    glEnd()

    glBegin(GL_LINES)
    glVertex3f(0, 0, -size)
    glVertex3f(0, 0, size)
    glEnd()

    glPopMatrix()

def draw_node_recursive(node):
    """
    Recursively draws bones and joints using OpenGL.
    """
    glPushMatrix()
    glTranslatef(node.world_position[0], node.world_position[1], node.world_position[2])
    glColor3f(1.0, 0.2, 0.2)

    glPointSize(8)
    glBegin(GL_POINTS)
    glVertex3f(0, 0, 0)
    glEnd()
    glPopMatrix()

    if node.parent:
        glBegin(GL_LINES)
        glColor3f(0.9, 0.9, 0.9)
        glVertex3fv(node.parent.world_position)
        glVertex3fv(node.world_position)
        glEnd()
    
    for child in node.children:
        draw_node_recursive(child)


def setup_scene():
    """
    Builds the Humanoid Skeleton.
    """
    root = Node("Root", offset=(0, 0, 0))
    
    _spine = Node("Spine", offset=(0, 3.0, 0), parent=root,
                limits={'x': (-10, 10), 'y': (-20, 20), 'z': (-10, 10)})
    
    _r_shoulder = Node("R_Shoulder", offset=(1.0, 1.0, 0), parent=_spine,
                      limits={'x': (-180, 180), 'y': (-90, 90), 'z': (-90, 90)})
    
    _r_elbow = Node("R_Elbow", offset=(3.0, 0, 0), parent=_r_shoulder,
                   limits={'x': (-10, 10), 'y': (0, 160), 'z': (-10, 10)}) # Hinge-ish
    
    _r_hand = Node("R_Hand", offset=(2.5, 0, 0), parent=_r_elbow)

    _l_shoulder = Node("L_Shoulder", offset=(-1.0, 1.0, 0), parent=_spine,
                      limits={'x': (-180, 180), 'y': (-90, 90), 'z': (-90, 90)})
    
    _l_elbow = Node("L_Elbow", offset=(-3.0, 0, 0), parent=_l_shoulder,
                   limits={'x': (-10, 10), 'y': (-160, 0), 'z': (-10, 10)})
    
    _l_hand = Node("L_Hand", offset=(-2.5, 0, 0), parent=_l_elbow)

    skeleton = Skeleton(root)
    return skeleton


def render_text_overlay(screen, font, user_text, last_status):
    """
    Renders the 2D UI text on top of the 3D scene.
    """
    text_surface = font.render(f"> {user_text}_", True, (255, 255, 255))
    screen.blit(text_surface, (20, WINDOW_SIZE[1] - 40))

    status_surface = font.render(f"Status: {last_status}", True, (200, 200, 200))
    screen.blit(status_surface, (20, 20))

    instructions_surface = font.render("Examples: 'Raise right hand', 'Move left hand forward 2'", True, (100, 100, 100))
    screen.blit(instructions_surface, (20, 50))


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Humanoid IK Commander")

    gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 100.0)
    glTranslatef(0.0, -5.0, -25)
    glRotatef(15, 1, 0, 0)

    skeleton = setup_scene()
    solver = JacobianIK(damping=0.5, step_size=0.05, max_iterations=5, threshold=0.1)
    commander = Commander(skeleton)
    font = pygame.font.SysFont('Consolas', 24)

    running = True
    user_text = ""
    last_status = "Ready"

    active_effector = None
    active_target_pos = None

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    result = commander.parse(user_text)
                    last_status = result.message
                    user_text = ""

                    if result.success and result.effector_name:
                        active_effector = result.effector_name
                        active_target_pos = result.target_pos

                        if result.message == "Skeleton reset to default pose.":
                            active_effector = None
                            active_target_pos = None

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    user_text += event.unicode
        
        skeleton.update()

        if active_effector and active_target_pos is not None:
            solver.solve(skeleton, active_effector, active_target_pos)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_grid()
        draw_node_recursive(skeleton.root)
        draw_target(active_target_pos)

        pygame.display.set_caption(f"Input: {user_text} | {last_status}")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()