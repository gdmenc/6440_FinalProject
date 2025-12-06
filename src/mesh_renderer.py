"""
Mesh renderer for displaying skinned 3D character meshes.
Supports rendering the mesh with skeletal deformation.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Optional

from src.core.mesh_skeleton import MeshSkeleton


class MeshRenderer:
    """
    Handles visualization of skinned meshes.
    Renders the deformed mesh with proper lighting and materials.
    """
    
    def __init__(self):
        self.show_wireframe = False
        self.show_skeleton_overlay = False
        self.mesh_color = (0.8, 0.65, 0.5)  # Skin-like color
        self.wireframe_color = (0.2, 0.2, 0.2)
    
    def set_wireframe(self, enabled: bool):
        """Toggle wireframe rendering."""
        self.show_wireframe = enabled
    
    def set_skeleton_overlay(self, enabled: bool):
        """Toggle skeleton overlay on mesh."""
        self.show_skeleton_overlay = enabled
    
    def render(self, mesh_skeleton: MeshSkeleton):
        """
        Render the skinned mesh using vertex arrays for performance.
        
        Args:
            mesh_skeleton: The MeshSkeleton containing deformed vertices
        """
        if mesh_skeleton.deformed_vertices is None or mesh_skeleton.mesh_faces is None:
            return
        
        vertices = mesh_skeleton.deformed_vertices
        faces = mesh_skeleton.mesh_faces
        normals = mesh_skeleton.deformed_normals
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.3, 0.25, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [*self.mesh_color, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30.0)
        
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(*self.wireframe_color)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glColor3f(*self.mesh_color)
        
        glShadeModel(GL_SMOOTH)
        
        # Use vertex arrays for faster rendering
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        
        if normals is not None and len(normals) == len(vertices):
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, normals)
        
        # Flatten face indices for glDrawElements
        indices = faces.flatten().astype(np.uint32)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        
        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_LIGHTING)
        
        # Optionally draw skeleton overlay
        if self.show_skeleton_overlay:
            self._render_skeleton_overlay(mesh_skeleton)
    
    def _render_skeleton_overlay(self, mesh_skeleton: MeshSkeleton):
        """Render skeleton bones as lines overlay on mesh."""
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        
        for node in mesh_skeleton.bone_nodes:
            # Draw joint as point
            glColor3f(1.0, 0.2, 0.2)
            glPointSize(6)
            glBegin(GL_POINTS)
            glVertex3fv(node.world_position)
            glEnd()
            
            # Draw bone as line to parent
            if node.parent:
                glColor3f(1.0, 1.0, 0.0)
                glBegin(GL_LINES)
                glVertex3fv(node.parent.world_position)
                glVertex3fv(node.world_position)
                glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
    
    def render_skeleton_only(self, mesh_skeleton: MeshSkeleton):
        """
        Render just the skeleton structure (no mesh).
        Useful for debugging bone positions.
        """
        glDisable(GL_LIGHTING)
        
        for node in mesh_skeleton.bone_nodes:
            # Draw joint as sphere
            glPushMatrix()
            glTranslatef(node.world_position[0], node.world_position[1], node.world_position[2])
            glColor3f(1.0, 0.3, 0.3)
            
            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, 0.15, 8, 8)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            
            # Draw bone as cylinder to parent
            if node.parent:
                self._draw_bone(node.parent.world_position, node.world_position)
    
    def _draw_bone(self, start_pos, end_pos, radius=0.08, color=(0.9, 0.9, 0.2)):
        """Draw a bone as a cylinder between two points."""
        glPushMatrix()
        glColor3f(*color)
        
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
                angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
                glRotatef(angle, axis[0], axis[1], axis[2])
            elif np.dot(z_axis, direction) < 0:
                glRotatef(180, 1, 0, 0)
        
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, radius, radius, length, 8, 1)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
