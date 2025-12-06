import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.node import Node
from src.core.skeleton import Skeleton
from src.core.mesh_skeleton import MeshSkeleton, MeshNode
from src.solver.jacobian_ik import JacobianIK
from src.input.commander import Commander
from src.renderer import SkeletonRenderer
from src.mesh_renderer import MeshRenderer
from src.motion.controller import WalkMotion, DanceMotion
from src.motion.controller import WaveMotion

WINDOW_SIZE = (1024, 768)
FPS = 60

# Help text content
HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║                    HUMANOID IK COMMANDER                     ║
║               Press F1 or type "help" to close               ║
╠══════════════════════════════════════════════════════════════╣
║  MOTION COMMANDS (type and press Enter)                      ║
║  ─────────────────────────────────────                       ║
║  walk              Start walking animation                   ║
║  walk fast         Walk at 1.8x speed                        ║
║  walk slow         Walk at 0.5x speed                        ║
║  walk <number>     Walk at custom speed (e.g., walk 1.5)     ║
║  dance             Start dancing animation                   ║
║  dance fast        Dance at 1.8x speed                       ║
║  dance slow        Dance at 0.5x speed                       ║
║  dance <number>    Dance at custom speed (e.g., dance 1.5)   ║
║  wave l/r          Start hand waving animation for l/r hand  ║
║  wave l/r fast     Wave l/r hand at 1.8x speed               ║
║  wave l/r slow     Wave l/r hand at 0.5x speed               ║
║  stop              Stop current animation                    ║
║  reset             Reset skeleton to default pose            ║
╠══════════════════════════════════════════════════════════════╣
║  IK COMMANDS (type and press Enter)                          ║
║  ─────────────────────────────────────                       ║
║  <body part> <direction> [distance]                          ║
║                                                              ║
║  Body Parts:                                                 ║
║    right/left hand, wrist, elbow, shoulder                   ║
║    right/left foot, ankle, knee, hip                         ║
║    head, neck, chest, torso, spine                           ║
║                                                              ║
║  Directions:                                                 ║
║    up, down, left, right, forward, back                      ║
║    (also: raise, lift, lower, drop, backward)                ║
║                                                              ║
║  Examples:                                                   ║
║    right hand up 5                                           ║
║    left foot forward 3                                       ║
║    head left                                                 ║
╠══════════════════════════════════════════════════════════════╣
║  MESH COMMANDS (type and press Enter)                        ║
║  ─────────────────────────────────────                       ║
║  load model1       Load Model1 mesh (1-4 available)          ║
║  load model2       Load Model2 mesh                          ║
║  unload            Unload mesh, return to default skeleton   ║
╠══════════════════════════════════════════════════════════════╣
║  CAMERA CONTROLS                                             ║
║  ─────────────────────────────────────                       ║
║  Left Mouse Drag   Rotate camera                             ║
║  Mouse Wheel       Zoom in/out                               ║
║  Arrow Keys        Pan camera                                ║
║  Ctrl+R            Reset camera                              ║
║  Ctrl+V            Toggle wireframe/capsule view             ║
║  Ctrl+B            Toggle mesh/skeleton view                 ║
║  F1                Toggle this help screen                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def surface_to_texture(surface):
    """
    Convert a pygame surface to an OpenGL texture.
    Returns texture ID, width, height.
    """
    texture_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    
    return texture_id, width, height


def draw_textured_quad(texture_id, x, y, width, height):
    """
    Draw a textured quad at the given position.
    """
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y + height)
    glTexCoord2f(1, 0); glVertex2f(x + width, y + height)
    glTexCoord2f(1, 1); glVertex2f(x + width, y)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)


def draw_help_overlay():
    """
    Renders the help overlay on top of the OpenGL scene.
    """
    overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))  # Dark semi-transparent background
    
    lines = HELP_TEXT.strip().split('\n')
    y_offset = 60
    line_height = 15
    
    help_font = pygame.font.SysFont("Consolas", 14)
    
    for line in lines:
        if '═' in line or '║' in line or '╔' in line or '╗' in line or '╚' in line or '╝' in line or '╠' in line or '╣' in line:
            color = (100, 180, 255)  # Blue for borders
        elif 'HUMANOID IK COMMANDER' in line or 'MOTION COMMANDS' in line or 'IK COMMANDS' in line or 'CAMERA CONTROLS' in line:
            color = (255, 220, 100)  # Yellow for headers
        elif line.strip().startswith('─'):
            color = (100, 180, 255)  # Blue for dividers
        elif any(cmd in line for cmd in ['walk', 'wave', 'stop', 'reset', 'right', 'left', 'Ctrl+', 'F1', 'Mouse', 'Arrow']):
            color = (150, 255, 150)  # Green for commands
        else:
            color = (220, 220, 220)  # Light gray for regular text
        
        text_surface = help_font.render(line, True, color)
        x_offset = (WINDOW_SIZE[0] - text_surface.get_width()) // 2
        overlay.blit(text_surface, (x_offset, y_offset))
        y_offset += line_height
    
    hint_font = pygame.font.SysFont("Consolas", 12)
    hint = hint_font.render("Type 'help' or press F1 to toggle this screen", True, (150, 150, 150))
    overlay.blit(hint, ((WINDOW_SIZE[0] - hint.get_width()) // 2, WINDOW_SIZE[1] - 40))
    
    texture_id, width, height = surface_to_texture(overlay)
    draw_textured_quad(texture_id, 0, 0, width, height)
    glDeleteTextures([texture_id])


def draw_status_bar(font, input_text, status_text, show_help_hint=True):
    """
    Renders the status bar at the bottom of the screen.
    """
    bar_height = 30
    bar_surface = pygame.Surface((WINDOW_SIZE[0], bar_height), pygame.SRCALPHA)
    bar_surface.fill((30, 30, 40, 230))
    
    input_display = f"> {input_text}_"
    input_surface = font.render(input_display, True, (255, 255, 255))
    bar_surface.blit(input_surface, (10, 5))
    
    status_surface = font.render(status_text, True, (180, 180, 180))
    bar_surface.blit(status_surface, (300, 5))
    
    if show_help_hint:
        hint_font = pygame.font.SysFont("Consolas", 12)
        hint = hint_font.render("F1: Help", True, (100, 150, 200))
        bar_surface.blit(hint, (WINDOW_SIZE[0] - 80, 8))
    
    texture_id, width, height = surface_to_texture(bar_surface)
    draw_textured_quad(texture_id, 0, WINDOW_SIZE[1] - bar_height, width, height)
    glDeleteTextures([texture_id])


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

def draw_axis():
    """
    Draws a simple axis to represent world frame.
    """
    size: float = 10.0
    offset: float = 0.0

    glPushAttrib(GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)
    glBegin(GL_LINES)
    
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(-offset, 0.0, 0.0)
    glVertex3f(size, 0.0, 0.0)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-offset, 0.0, 0.0)
    glVertex3f(0.0, size, 0.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-offset, 0.0, 0.0)
    glVertex3f(0.0, 0.0, size)

    glEnd()
    glPopAttrib()

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


def draw_sphere(position, radius, color=(0.8, 0.6, 0.4)):
    """
    Draws a sphere at the given position.
    """
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3f(*color)

    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluSphere(quadric, radius, 16, 16)
    gluDeleteQuadric(quadric)

    glPopMatrix()


def draw_cylinder_between(start_pos, end_pos, radius, color=(0.8, 0.6, 0.4)):
    """
    Draws a cylinder between two points.
    """
    glPushMatrix()

    glTranslatef(start_pos[0], start_pos[1], start_pos[2])
    direction = end_pos - start_pos
    length = np.linalg.norm(direction)

    if length > 1e-3:
        direction = direction / length
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)

        if axis_len > 1e-3:
            axis /= axis_len
            angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction))))
            glRotatef(angle, axis[0], axis[1], axis[2])

        elif np.dot(z_axis, direction) < 0:
            glRotatef(180, 1, 0, 0)

    glColor3f(*color)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluCylinder(quadric, radius, radius, length, 16, 1)
    gluDeleteQuadric(quadric)

    glPopMatrix()


def get_body_part_color(node_name):
    name = node_name.lower()
    if "head" in name or "neck" in name:
        return (0.95, 0.8, 0.7)
    if "spine" in name or "chest" in name or "root" in name:
        return (0.85, 0.65, 0.5)
    if "r_" in name and any(part in name for part in ("shoulder", "elbow", "wrist", "hand")):
        return (0.7, 0.75, 0.9)
    if "l_" in name and any(part in name for part in ("shoulder", "elbow", "wrist", "hand")):
        return (0.9, 0.7, 0.75)
    if "r_" in name and any(part in name for part in ("hip", "knee", "ankle", "foot")):
        return (0.6, 0.65, 0.85)
    if "l_" in name and any(part in name for part in ("hip", "knee", "ankle", "foot")):
        return (0.85, 0.6, 0.65)
    return (0.8, 0.6, 0.4)


def get_joint_radius(node_name):
    name = node_name.lower()
    if "head" in name:
        return 0.7
    if "neck" in name:
        return 0.4
    if "chest" in name or "spine2" in name:
        return 0.6
    if "spine" in name or "root" in name:
        return 0.5
    if "shoulder" in name or "hip" in name:
        return 0.4
    if "elbow" in name or "knee" in name:
        return 0.3
    if "wrist" in name or "ankle" in name:
        return 0.25
    if "hand" in name or "foot" in name:
        return 0.2
    return 0.3


def draw_node_capsule(node):
    """
    Recursively draws bones as capsules.
    """
    color = get_body_part_color(node)
    joint_radius = get_joint_radius(node.name)

    draw_sphere(node.world_position, joint_radius, color)

    if node.parent:
        parent_radius = get_joint_radius(node.parent)
        bone_radius = (joint_radius + parent_radius) / 3.0
        draw_cylinder_between(node.parent.world_position, node.world_position, bone_radius, color)

    for child in node.children:
        draw_node_capsule(child)


def setup_scene():
    """
    Builds the Humanoid Skeleton with full body structure.
    """
    root = Node("Root", offset=(0, 5.7, 0))

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

    _neck = Node(
        "Neck",
        offset=(0, 0.8, 0),
        parent=_chest,
        limits={"x": (-45, 45), "y": (-80, 80), "z": (-45, 45)},
    )

    _head = Node("Head", offset=(0, 0.6, 0), parent=_neck)

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
        limits={"x": (-10, 10), "y": (0, 150), "z": (-10, 10)},
    )

    _r_wrist = Node(
        "R_Wrist",
        offset=(-2.0, 0, 0),
        parent=_r_elbow,
        limits={"x": (-70, 70), "y": (-90, 90), "z": (-20, 20)},
    )

    _r_hand = Node("R_Hand", offset=(-0.8, 0, 0), parent=_r_wrist)

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
        limits={"x": (-10, 10), "y": (-150, 0), "z": (-10, 10)},
    )

    _l_wrist = Node(
        "L_Wrist",
        offset=(2.0, 0, 0),
        parent=_l_elbow,
        limits={"x": (-70, 70), "y": (-90, 90), "z": (-20, 20)},
    )

    _l_hand = Node("L_Hand", offset=(0.8, 0, 0), parent=_l_wrist)

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
    
    Press F1 or type 'help' for the full command reference.
    """
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Humanoid IK Commander")

    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    light_position = [10.0, 10.0, 10.0, 1.0]
    light_ambient = [0.3, 0.3, 0.3, 1.0]
    light_diffuse = [0.8, 0.8, 0.8, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 100.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    camera_distance = 30.0
    camera_rotation_x = 10.0
    camera_rotation_y = 0.0
    camera_pan_x = 0.0
    camera_pan_y = -5.0

    mouse_dragging = False
    last_mouse_pos = (0, 0)

    skeleton = setup_scene()
    skeleton.reset_pose()  # Set initial standing pose
    solver = JacobianIK(damping=0.5, step_size=0.05, max_iterations=5, threshold=0.1)
    commander = Commander(skeleton)
    renderer = SkeletonRenderer()
    mesh_renderer = MeshRenderer()
    font = pygame.font.SysFont("Consolas", 24)
    
    walk_motion = WalkMotion(skeleton, speed=1.0)
    dance_motion = DanceMotion(skeleton, speed=1.0)
    wave_motion = WaveMotion(skeleton)
    
    # Mesh skeleton (loaded on demand)
    mesh_skeleton = None
    show_mesh = False  # Toggle between skeleton and mesh view

    running = True
    user_text = ""
    last_status = "Ready - Press F1 for help"
    show_help = False  # Help overlay toggle

    active_effector = None
    active_target_pos = None

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    cmd_lower = user_text.lower().strip()
                    
                    if cmd_lower == "help":
                        show_help = not show_help
                        user_text = ""
                        continue
                    
                    elif cmd_lower == "walk":
                        dance_motion.stop()
                        wave_motion.stop()
                        walk_motion.start()
                        active_effector = None
                        active_target_pos = None
                        last_status = "Walking animation started"
                    elif cmd_lower == "dance":
                        walk_motion.stop()
                        wave_motion.stop()
                        dance_motion.start()
                        active_effector = None
                        active_target_pos = None
                        last_status = "Dancing animation started"

                    elif cmd_lower == "stop":
                        walk_motion.stop()
                        dance_motion.stop()
                        wave_motion.stop()
                        last_status = "Animation stopped"
                    elif cmd_lower.startswith("dance "):
                        walk_motion.stop()
                        wave_motion.stop()
                        speed_str = cmd_lower[6:].strip()
                        if speed_str == "fast":
                            dance_motion.set_speed(1.8)
                        elif speed_str == "slow":
                            dance_motion.set_speed(0.5)
                        else:
                            try:
                                dance_motion.set_speed(float(speed_str))
                            except ValueError:
                                pass
                        dance_motion.start()
                        active_effector = None
                        active_target_pos = None
                        last_status = f"Dancing at speed {dance_motion.speed:.1f}x"

                    elif cmd_lower.startswith("walk "):
                        dance_motion.stop()
                        wave_motion.stop()
                        speed_str = cmd_lower[5:].strip()
                        if speed_str == "fast":
                            walk_motion.set_speed(1.8)
                        elif speed_str == "slow":
                            walk_motion.set_speed(0.5)
                        else:
                            try:
                                walk_motion.set_speed(float(speed_str))
                            except ValueError:
                                pass
                        walk_motion.start()
                        active_effector = None
                        active_target_pos = None
                        last_status = f"Walking at speed {walk_motion.speed:.1f}x"

                    elif cmd_lower.startswith("wave "):
                        walk_motion.stop()
                        dance_motion.stop()
                        if "fast" in cmd_lower:
                            wave_motion.set_speed(1.8)
                            speed_idx = cmd_lower.find("fast")
                            wave_motion.specified_arm = cmd_lower[5:speed_idx].strip()
                        elif "slow" in cmd_lower:
                            wave_motion.set_speed(0.5)
                            speed_idx = cmd_lower.find("slow")
                            wave_motion.specified_arm = cmd_lower[5:speed_idx].strip()
                        else:
                            wave_motion.specified_arm = cmd_lower[5:].strip()

                        wave_motion.start()
                        active_effector = None
                        active_target_pos = None
                        last_status = f"Hand waving at speed {wave_motion.speed:.1f}x"

                    elif cmd_lower.startswith("load "):
                        # Load mesh model command
                        model_name = cmd_lower[5:].strip()
                        model_map = {
                            "model1": "assests/Model1.skel",
                            "model2": "assests/Model2.skel",
                            "model3": "assests/Model3.skel",
                            "model4": "assests/Model4.skel",
                        }
                        if model_name in model_map:
                            try:
                                mesh_skeleton = MeshSkeleton.from_files(model_map[model_name])
                                mesh_skeleton.reset_pose()
                                show_mesh = True
                                last_status = f"Loaded {model_name.capitalize()} mesh"
                            except Exception as e:
                                last_status = f"Failed to load mesh: {str(e)[:30]}"
                        else:
                            last_status = f"Unknown model: {model_name} (try model1-model4)"
                    
                    elif cmd_lower == "unload":
                        mesh_skeleton = None
                        show_mesh = False
                        last_status = "Mesh unloaded, using default skeleton"

                    else:
                        result = commander.parse(user_text)
                        last_status = result.message

                        if result.success:
                            walk_motion.pause()
                            dance_motion.pause()
                            wave_motion.pause()
                            active_effector = result.effector_name
                            active_target_pos = result.target_pos

                            if result.message == "Skeleton reset to default pose.":
                                active_effector = None
                                active_target_pos = None
                            elif result.effector_name:
                                active_effector = result.effector_name
                                active_target_pos = result.target_pos
                    
                    user_text = ""

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

                # Toggle visualization mode with Ctrl+V
                elif (
                    event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    mode = renderer.toggle_mode()
                    last_status = f"Visualization: {mode.capitalize()}"

                # Toggle mesh/skeleton view with Ctrl+B
                elif (
                    event.key == pygame.K_b and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    if mesh_skeleton is not None:
                        show_mesh = not show_mesh
                        view_mode = "Mesh" if show_mesh else "Skeleton"
                        last_status = f"View mode: {view_mode}"
                    else:
                        last_status = "No mesh loaded. Use 'load model1' first."

                # Toggle help overlay with F1
                elif event.key == pygame.K_F1:
                    show_help = not show_help

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

        dt = clock.get_time() / 1000.0
        
        if walk_motion.is_active():
            walk_motion.update(dt)
        elif dance_motion.is_active():
            dance_motion.update(dt)
        elif wave_motion.is_active():
            wave_motion.update(dt)
        else:
            skeleton.update()

            if active_effector and active_target_pos is not None:
                solver.solve(skeleton, active_effector, active_target_pos)

        # Sync rotations from main skeleton to mesh skeleton
        if mesh_skeleton is not None:
            # Map joint names: our naming -> C++ mesh naming
            name_mapping = {
                "Root": "Root",
                "Spine1": "Chest",
                "Spine2": "Waist",
                "Chest": "Neck",  # Our Chest maps to their Neck position in hierarchy
                "Neck": "Neck",
                "R_Hip": "Right hip",
                "R_Knee": "Right knee",
                "R_Ankle": "Right foot",
                "L_Hip": "Left hip",
                "L_Knee": "Left knee",
                "L_Ankle": "Left foot",
                "R_Shoulder": "Right shoulder",
                "R_Elbow": "Right elbow",
                "L_Shoulder": "Left shoulder",
                "L_Elbow": "Left elbow",
            }
            
            for joint_name, joint in skeleton.joints_map.items():
                mesh_name = name_mapping.get(joint_name, joint_name)
                mesh_joint = mesh_skeleton.get_joint(mesh_name)
                if mesh_joint is not None:
                    rot = joint.rotation.copy()
                    if joint_name in ["R_Shoulder", "L_Shoulder", "R_Elbow", "L_Elbow"]:
                        orig_y = rot[1]
                        orig_z = rot[0]
                        rot[1] = orig_z
                        rot[0] = -orig_y
                    if joint_name in ["L_Shoulder", "L_Elbow"]:
                        rot[1] = -rot[1]
                        rot[0] = -rot[0]
                        rot[2] = rot[2]

                    if joint_name == "R_Shoulder":
                        rot[2] -= 80  # Rotate down (around Z)
                    elif joint_name == "L_Shoulder":
                        rot[2] += 80  # Rotate down (around Z)
                    
                    mesh_joint.rotation = rot
            mesh_skeleton.update()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(camera_pan_x, camera_pan_y, -camera_distance)
        glRotatef(camera_rotation_x, 1, 0, 0)
        glRotatef(camera_rotation_y, 0, 1, 0)

        glDisable(GL_LIGHTING)
        draw_grid()
        draw_target(active_target_pos)
        draw_axis()
        
        # Render mesh or skeleton based on current view mode
        if show_mesh and mesh_skeleton is not None:
            mesh_renderer.render(mesh_skeleton)
        else:
            renderer.render(skeleton.root)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_SIZE[0], WINDOW_SIZE[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        if show_help:
            draw_help_overlay()
        
        draw_status_bar(font, user_text, last_status, not show_help)
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        pygame.display.set_caption("Humanoid IK Commander")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
