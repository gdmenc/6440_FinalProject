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
    Builds the Humanoid Skeleton with full body structure.
    """
    # Root/Pelvis - Lifted so feet are on the ground
    root = Node("Root", offset=(0, 5.7, 0))

    # Spine chain (multiple segments for flexibility)
    _spine1 = Node(
        "Spine1",
        offset=(0, 1.0, 0),
        parent=root,
        limits={"x": (-20, 20), "y": (-30, 30), "z": (-20, 20)},
    )

    _spine2 = Node(
        "Spine2",
        offset=(0, 1.0, 0),
        parent=_spine1,
        limits={"x": (-20, 20), "y": (-30, 30), "z": (-20, 20)},
    )

    _chest = Node(
        "Chest",
        offset=(0, 1.0, 0),
        parent=_spine2,
        limits={"x": (-15, 15), "y": (-30, 30), "z": (-15, 15)},
    )

    # Neck and Head
    _neck = Node(
        "Neck",
        offset=(0, 0.8, 0),
        parent=_chest,
        limits={"x": (-45, 45), "y": (-80, 80), "z": (-45, 45)},
    )

    _head = Node("Head", offset=(0, 0.6, 0), parent=_neck)

    # Right arm chain
    _r_shoulder = Node(
        "R_Shoulder",
        offset=(-1.2, 0.5, 0),
        parent=_chest,
        limits={"x": (-180, 180), "y": (-90, 90), "z": (-90, 90)},
    )

    _r_elbow = Node(
        "R_Elbow",
        offset=(-2.5, 0, 0),
        parent=_r_shoulder,
        limits={"x": (-10, 10), "y": (-150, 0), "z": (-10, 10)},
    )

    _r_wrist = Node(
        "R_Wrist",
        offset=(-2.0, 0, 0),
        parent=_r_elbow,
        limits={"x": (-70, 70), "y": (-90, 90), "z": (-20, 20)},
    )

    _r_hand = Node("R_Hand", offset=(-0.8, 0, 0), parent=_r_wrist)

    # Left arm chain
    _l_shoulder = Node(
        "L_Shoulder",
        offset=(1.2, 0.5, 0),
        parent=_chest,
        limits={"x": (-180, 180), "y": (-90, 90), "z": (-90, 90)},
    )

    _l_elbow = Node(
        "L_Elbow",
        offset=(2.5, 0, 0),
        parent=_l_shoulder,
        limits={"x": (-10, 10), "y": (0, 150), "z": (-10, 10)},
    )

    _l_wrist = Node(
        "L_Wrist",
        offset=(2.0, 0, 0),
        parent=_l_elbow,
        limits={"x": (-70, 70), "y": (-90, 90), "z": (-20, 20)},
    )

    _l_hand = Node("L_Hand", offset=(0.8, 0, 0), parent=_l_wrist)

    # Right leg chain
    _r_hip = Node(
        "R_Hip",
        offset=(-0.6, -0.2, 0),
        parent=root,
        limits={"x": (-120, 45), "y": (-45, 45), "z": (-90, 45)},
    )

    _r_knee = Node(
        "R_Knee",
        offset=(0, -2.5, 0),
        parent=_r_hip,
        limits={"x": (-150, 0), "y": (-10, 10), "z": (-10, 10)},
    )

    _r_ankle = Node(
        "R_Ankle",
        offset=(0, -2.5, 0),
        parent=_r_knee,
        limits={"x": (-45, 45), "y": (-20, 20), "z": (-30, 30)},
    )

    _r_foot = Node("R_Foot", offset=(0, -0.5, 0.8), parent=_r_ankle)

    # Left leg chain
    _l_hip = Node(
        "L_Hip",
        offset=(0.6, -0.2, 0),
        parent=root,
        limits={"x": (-120, 45), "y": (-45, 45), "z": (-45, 90)},
    )

    _l_knee = Node(
        "L_Knee",
        offset=(0, -2.5, 0),
        parent=_l_hip,
        limits={"x": (-150, 0), "y": (-10, 10), "z": (-10, 10)},
    )

    _l_ankle = Node(
        "L_Ankle",
        offset=(0, -2.5, 0),
        parent=_l_knee,
        limits={"x": (-45, 45), "y": (-20, 20), "z": (-30, 30)},
    )

    _l_foot = Node("L_Foot", offset=(0, -0.5, 0.8), parent=_l_ankle)

    skeleton = Skeleton(root)
    return skeleton


def main():
    """
    Main application loop for the Humanoid IK Commander.

    Controls:
    - Type commands and press Enter to move limbs
    - Left Mouse Drag: Rotate camera
    - Mouse Wheel: Zoom in/out
    - Arrow Keys: Pan camera
    - Ctrl+R: Reset camera
    """
    # Initialize Pygame and OpenGL
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Humanoid IK Commander")

    glEnable(GL_DEPTH_TEST)

    # Setup Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 100.0)
    
    # Switch back to ModelView for camera/object transforms
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Camera state (adjusted for full-body view)
    camera_distance = 30.0
    camera_rotation_x = 10.0
    camera_rotation_y = 0.0
    camera_pan_x = 0.0
    camera_pan_y = -5.0  # Look up at the torso (approx Y=5)

    # Mouse control state
    mouse_dragging = False
    last_mouse_pos = (0, 0)

    # Initialize scene components
    skeleton = setup_scene()
    skeleton.reset_pose()  # Set initial standing pose
    solver = JacobianIK(damping=0.5, step_size=0.05, max_iterations=5, threshold=0.1)
    commander = Commander(skeleton)
    font = pygame.font.SysFont("Consolas", 24)

    # Application state
    running = True
    user_text = ""
    last_status = "Ready - Use mouse/arrows to move camera"

    # IK solver state
    active_effector = None
    active_target_pos = None

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Process text command
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

                # Camera panning with arrow keys
                elif event.key == pygame.K_LEFT:
                    camera_pan_x -= 2.0
                elif event.key == pygame.K_RIGHT:
                    camera_pan_x += 2.0
                elif event.key == pygame.K_UP:
                    camera_pan_y += 2.0
                elif event.key == pygame.K_DOWN:
                    camera_pan_y -= 2.0

                # Camera reset with Ctrl+R
                elif (
                    event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    camera_distance = 30.0
                    camera_rotation_x = 10.0
                    camera_rotation_y = 0.0
                    camera_pan_x = 0.0
                    camera_pan_y = -5.0
                    last_status = "Camera reset"

                else:
                    user_text += event.unicode

            # Mouse button down - start dragging
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()

            # Mouse button up - stop dragging
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False

            # Mouse motion - rotate camera
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]

                    camera_rotation_y += dx * 0.5
                    camera_rotation_x += dy * 0.5

                    # Clamp vertical rotation to prevent flipping
                    camera_rotation_x = np.clip(camera_rotation_x, -89, 89)

                    last_mouse_pos = current_mouse_pos

            # Mouse wheel - zoom camera
            elif event.type == pygame.MOUSEWHEEL:
                camera_distance -= event.y * 2.0
                camera_distance = np.clip(camera_distance, 5.0, 100.0)

        # Update skeleton forward kinematics
        skeleton.update()

        # Run IK solver if active
        if active_effector and active_target_pos is not None:
            solver.solve(skeleton, active_effector, active_target_pos)

        # Clear buffers and reset transformations
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply camera transformations
        glTranslatef(camera_pan_x, camera_pan_y, -camera_distance)
        glRotatef(camera_rotation_x, 1, 0, 0)
        glRotatef(camera_rotation_y, 0, 1, 0)

        # Draw scene
        draw_grid()
        draw_node_recursive(skeleton.root)
        draw_target(active_target_pos)

        # Update display
        pygame.display.set_caption(f"Input: {user_text} | {last_status}")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
