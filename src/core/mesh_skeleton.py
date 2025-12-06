"""
Mesh-based skeleton system for loading and rendering 3D character meshes.
Loads .skel, .attach, and .obj files to create a skinned mesh character.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from src.core.node import Node
from src.core.skeleton import Skeleton
from src.core.transform import (
    Mat4x4,
    get_translation_matrix,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
)


class MeshNode(Node):
    """
    Extension of Node that stores additional mesh-related data.
    Tracks bind pose and current pose for skinning calculations.
    """
    
    def __init__(
        self,
        name: str,
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent: Optional[MeshNode] = None,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
        index: int = 0,
    ) -> None:
        super().__init__(name, offset, parent, limits)
        self.index = index
        # Bind pose matrices (captured at rest pose)
        self.bind_world_matrix: Mat4x4 = np.eye(4, dtype=np.float32)
        self.bind_world_inverse: Mat4x4 = np.eye(4, dtype=np.float32)
    
    def capture_bind_pose(self) -> None:
        """
        Captures the current world matrix as the bind pose.
        Call this after the skeleton is set up in rest pose.
        """
        self.bind_world_matrix = self.world_matrix.copy()
        self.bind_world_inverse = np.linalg.inv(self.bind_world_matrix).astype(np.float32)
        
        for child in self.children:
            if isinstance(child, MeshNode):
                child.capture_bind_pose()
    
    def get_skinning_matrix(self) -> Mat4x4:
        """
        Returns the matrix to transform vertices from bind pose to current pose.
        This is: current_world * inverse_bind_world
        """
        return self.world_matrix @ self.bind_world_inverse


class MeshSkeleton(Skeleton):
    """
    Skeleton that can be loaded from .skel files and supports mesh skinning.
    """
    
    # Joint names matching the C++ implementation exactly
    DEFAULT_JOINT_NAMES = [
        "Root",             # 0
        "Chest",            # 1
        "Waist",            # 2
        "Neck",             # 3
        "Right hip",        # 4
        "Right leg",        # 5
        "Right knee",       # 6
        "Right foot",       # 7
        "Left hip",         # 8
        "Left leg",         # 9
        "Left knee",        # 10
        "Left foot",        # 11
        "Right collarbone", # 12
        "Right shoulder",   # 13
        "Right elbow",      # 14
        "Left collarbone",  # 15
        "Left shoulder",    # 16
        "Left elbow",       # 17
    ]
    
    def __init__(self, root: MeshNode):
        super().__init__(root)
        self.mesh_vertices: Optional[NDArray[np.float32]] = None
        self.mesh_faces: Optional[NDArray[np.int32]] = None
        self.mesh_normals: Optional[NDArray[np.float32]] = None
        self.skin_weights: Optional[NDArray[np.float32]] = None
        self.deformed_vertices: Optional[NDArray[np.float32]] = None
        self.deformed_normals: Optional[NDArray[np.float32]] = None
        self.bone_nodes: List[MeshNode] = []
        
    @classmethod
    def from_files(cls, skel_path: str, obj_path: str = None, attach_path: str = None) -> 'MeshSkeleton':
        """
        Load a mesh skeleton from .skel, .obj, and .attach files.
        
        Args:
            skel_path: Path to the .skel file
            obj_path: Optional path to .obj file (defaults to same name as skel)
            attach_path: Optional path to .attach file (defaults to same name as skel)
        """
        skel_path = Path(skel_path)
        base_path = skel_path.parent / skel_path.stem
        
        if obj_path is None:
            obj_path = str(base_path) + ".obj"
        if attach_path is None:
            attach_path = str(base_path) + ".attach"
        
        # Load skeleton structure
        skeleton = cls._load_skel(str(skel_path))
        
        # Load mesh if available
        if Path(obj_path).exists():
            skeleton._load_obj(obj_path)
        
        # Load skin weights if available
        if Path(attach_path).exists():
            skeleton._load_attach(attach_path)
            
        # IMPORTANT: Update skeleton matrices to ensure world_matrix is valid
        # BEFORE capturing the bind pose
        skeleton.update()
        
        # Capture bind pose (using the updated world matrices)
        skeleton.capture_bind_pose()
        
        return skeleton
    
    @classmethod
    def _load_skel(cls, skel_path: str) -> 'MeshSkeleton':
        """
        Load skeleton hierarchy from .skel file.
        The .skel file format: x y z parent_idx
        Where x,y,z is the LOCAL offset from parent (not world position).
        """
        bones = []
        
        with open(skel_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    parent_idx = int(parts[3])
                    bones.append({
                        'offset': (x, y, z),
                        'parent_idx': parent_idx
                    })
        
        # Create nodes - positions in .skel are LOCAL offsets from parent
        nodes: List[MeshNode] = []
        scale = 15.0  # Scale to match our scene units
        
        for i, bone in enumerate(bones):
            name = cls.DEFAULT_JOINT_NAMES[i] if i < len(cls.DEFAULT_JOINT_NAMES) else f"Bone_{i}"
            
            if i == 0:
                # Root node: center at origin (x=0, z=0), keep height
                root_height = bone['offset'][1] * scale  # Keep the Y height
                offset = (0.0, root_height, 0.0)
            else:
                # Child nodes: use offset directly
                offset = (
                    bone['offset'][0] * scale,
                    bone['offset'][1] * scale,
                    bone['offset'][2] * scale
                )
            
            parent = nodes[bone['parent_idx']] if bone['parent_idx'] >= 0 and bone['parent_idx'] < len(nodes) else None
            
            node = MeshNode(
                name=name,
                offset=offset,
                parent=parent,
                index=i
            )
            nodes.append(node)
        
        if not nodes:
            raise ValueError(f"No bones found in {skel_path}")
        
        skeleton = cls(nodes[0])
        skeleton.bone_nodes = nodes
        skeleton.update()
        
        return skeleton
    
    def _load_obj(self, obj_path: str) -> None:
        """Load mesh geometry from .obj file."""
        vertices = []
        faces = []
        normals = []
        
        with open(obj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                elif parts[0] == 'vn':
                    nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                    normals.append([nx, ny, nz])
                elif parts[0] == 'f':
                    # Handle face format: v, v/vt, v/vt/vn, or v//vn
                    face_indices = []
                    for p in parts[1:]:
                        idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                        face_indices.append(idx)
                    if len(face_indices) >= 3:
                        faces.append(face_indices[:3])
        
        self.mesh_vertices = np.array(vertices, dtype=np.float32)
        self.mesh_faces = np.array(faces, dtype=np.int32)
        
        if len(self.mesh_vertices) > 0:
            # Scale to match skeleton
            scale = 15.0
            self.mesh_vertices *= scale
            
            # Center mesh around origin
            center = (self.mesh_vertices.max(axis=0) + self.mesh_vertices.min(axis=0)) / 2
            self.mesh_vertices -= center
            
            # Put feet at y=0
            min_y = self.mesh_vertices[:, 1].min()
            self.mesh_vertices[:, 1] -= min_y
        
        # Always recompute normals after geometric transformations to ensure correctness
        self._compute_normals()
        
        # Initialize deformed vertices
        self.deformed_vertices = self.mesh_vertices.copy()
        if self.mesh_normals is not None:
            self.deformed_normals = self.mesh_normals.copy()
    
    def _compute_normals(self) -> None:
        """Compute vertex normals from faces."""
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
            
        normals = np.zeros_like(self.mesh_vertices)
        
        for face in self.mesh_faces:
            v0, v1, v2 = self.mesh_vertices[face[0]], self.mesh_vertices[face[1]], self.mesh_vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Add to each vertex's normal
            for idx in face:
                normals[idx] += face_normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.mesh_normals = (normals / norms).astype(np.float32)
        self.deformed_normals = self.mesh_normals.copy()
    
    def _load_attach(self, attach_path: str) -> None:
        """Load skin weights from .attach file."""
        weights = []
        
        with open(attach_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    row = [float(p) for p in parts]
                    weights.append(row)
        
        if weights:
            self.skin_weights = np.array(weights, dtype=np.float32)
    
    def capture_bind_pose(self) -> None:
        """
        Captures the inverse bind matrices for all bones.
        These are used to transform vertices from model space to bone local space.
        Matches C++: inverse_bind_matrices_.push_back(glm::inverse(joint_ptr->GetTransform().GetLocalToWorldMatrix()))
        """
        self.inv_bind_matrices = []
        for node in self.bone_nodes:
            # We need the inverse of the world matrix at bind time
            inv_bind = np.linalg.inv(node.world_matrix)
            self.inv_bind_matrices.append(inv_bind)

    def update_mesh(self) -> None:
        """
        Perform skeletal mesh deformation (Linear Blend Skinning).
        Matches C++ implementation: 
        final_pos += weight * (current_T[joint_idx] * inverse_bind_matrices_[joint_idx] * bind_pos)
        """
        if (self.mesh_vertices is None or 
            self.skin_weights is None or 
            not hasattr(self, 'inv_bind_matrices') or 
            not self.inv_bind_matrices):
            return

        # Pre-compute skinning matrices for all bones: M = current_world * inv_bind
        # This transforms a vertex from Bind Model Space -> Bone Local -> Current World Space
        skinning_matrices = []
        for i, node in enumerate(self.bone_nodes):
            # C++: current_T[joint_idx] * inverse_bind_matrices_[joint_idx]
            # Note: Matrix multiplication order in numpy is A @ B for A * B
            M = node.world_matrix @ self.inv_bind_matrices[i]
            skinning_matrices.append(M)
        
        # We need to apply this to all vertices. 
        # For performance valid computation, we can vectorize this.
        # However, to ensure EXACT correctness with C++ reference first, let's look at the math:
        # v_final = sum( weight_j * (M_j @ v_bind) )
        
        num_verts = len(self.mesh_vertices)
        
        # Convert vertices to homogeneous coordinates (Nx4)
        ones = np.ones((num_verts, 1), dtype=np.float32)
        v_bind_homo = np.hstack([self.mesh_vertices, ones]) # Shape (N, 4)
        
        # Initialize final positions and normals
        v_final = np.zeros_like(self.mesh_vertices)
        n_final = np.zeros_like(self.mesh_normals) if self.mesh_normals is not None else None
        
        # We will iterate through bones (columns of weights) to vectorize efficiently
        # C++: weight column j corresponds to bone index j+1
        # The root bone (index 0) gets the remainder weight: 1.0 - sum(weights)
        
        num_bones = len(self.bone_nodes)
        num_weight_cols = self.skin_weights.shape[1]
        
        # Track total weight for each vertex to calculate root weight later
        total_weights = np.zeros(num_verts, dtype=np.float32)
        
        # Accumulate deformations from bones 1..N
        for j in range(num_weight_cols):
            bone_idx = j + 1
            if bone_idx >= num_bones:
                continue
                
            weights = self.skin_weights[:, j] # Shape (N,)
            total_weights += weights
            
            # Skinning matrix for this bone
            M = skinning_matrices[bone_idx]
            
            # Transform all vertices by this bone's matrix
            # (N, 4) = (N, 4) @ (4, 4).T 
            v_transformed = v_bind_homo @ M.T
            
            # Add weighted contribution: separate x,y,z scalling
            # Shape (N, 3) += (N, 1) * (N, 3)
            # Take only xyz from transformed
            v_final += weights[:, np.newaxis] * v_transformed[:, :3]

            # Transform Normals if they exist
            # For rigid/uniform-scale transforms, we can use the upper 3x3 of M
            if n_final is not None:
                # Rotation part of M
                # Normal transform is technically (M^-1).T but for rotation M.R is fine
                M_rot = M[:3, :3]
                n_transformed = self.mesh_normals @ M_rot.T
                n_final += weights[:, np.newaxis] * n_transformed

        # Handle Root Bone (Index 0) - Remainder weights
        root_weights = np.maximum(0, 1.0 - total_weights)
        
        if np.any(root_weights > 0):
            M_root = skinning_matrices[0]
            
            v_root_transformed = v_bind_homo @ M_root.T
            v_final += root_weights[:, np.newaxis] * v_root_transformed[:, :3]
            
            if n_final is not None:
                M_root_rot = M_root[:3, :3]
                n_root_transformed = self.mesh_normals @ M_root_rot.T
                n_final += root_weights[:, np.newaxis] * n_root_transformed

        self.deformed_vertices = v_final
        
        # Normalize the final normals
        if n_final is not None:
            # Avoid division by zero
            norms = np.linalg.norm(n_final, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            self.deformed_normals = n_final / norms
        else:
             self._compute_normals()
             self.deformed_normals = self.mesh_normals.copy()
    
    def update(self) -> None:
        """Update skeleton and mesh."""
        super().update()
        self.update_mesh()
    
    def reset_pose(self) -> None:
        """
        Resets all joints in the mesh skeleton to their default state.
        """
        self.root.reset()
        self.update()
    
    def get_joint(self, name: str):
        """
        Retrieves a joint by name from the bone_nodes list.
        Returns None if the joint does not exist.
        """
        # First check the standard joints_map from parent class
        joint = self.joints_map.get(name)
        if joint:
            return joint
        
        # Also search in bone_nodes by name
        for node in self.bone_nodes:
            if node.name == name:
                return node
        return None
    
    def get_bone_by_index(self, index: int) -> Optional[MeshNode]:
        """Get a bone node by its index."""
        if 0 <= index < len(self.bone_nodes):
            return self.bone_nodes[index]
        return None
