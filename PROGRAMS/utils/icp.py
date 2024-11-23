import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from enum import Enum
from utils.meshgrid import Meshgrid
from utils.octree import Octree

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class Matching(Enum):
    SIMPLE_LINEAR = 1
    VECTORIZED_LINEAR = 2
    SIMPLE_OCTREE = 3
    VECTORIZED_OCTREE = 4

class IterativeClosestPoint():
    def __init__(
        self,
        max_iter: int = 1000,
        match_mode: Matching = Matching.VECTORIZED_LINEAR
    ) -> None:
        # Define maximum number of ICP iterations
        self.max_iter: int = max_iter

        # Define the algorithm used to find closest points
        self.match_mode: Matching = match_mode

    def __call__(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Runs the full ICP algorithm given a point cloud and meshgrid.
        """
        
        for i in self.max_iter:
            closest_pt, dist = self.match(pt_cloud, meshgrid)

    def match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Finds the closest point and distance from points on a point cloud 
        to a triangle meshgrid.
        """

        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')

        # Input meshgrid should be a Meshgrid object
        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')

        ####
        ## Search algorithms to find closest points
        ####

        # Performs a simple linear search for closest points
        if self.match_mode == Matching.SIMPLE_LINEAR:
            return self._simple_linear_match(pt_cloud, meshgrid)
        
        # Performs a faster vectorized linear search for closest points
        elif self.match_mode == Matching.VECTORIZED_LINEAR:
            return self._vectorized_linear_match(pt_cloud, meshgrid)
        
        # Performs a simple iterative search for closest points using Octrees
        elif self.match_mode == Matching.SIMPLE_OCTREE:
            return self._simple_octree_match(pt_cloud, meshgrid)
        
        # Performs a simple iterative search for closest points using Octrees
        elif self.match_mode == Matching.VECTORIZED_OCTREE:
            return self._vectorized_octree_match(pt_cloud, meshgrid)

    def _simple_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a simple linear search for the closest point 
        containing loops over all data points and Triangles. Finds the 
        closest point and distance to the meshgrid.
        """

        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Iterate through all the points and triangles in the meshgrid
        for i, point in enumerate(pt_cloud):
            for triangle in meshgrid:
                # Extract the bounding box of the triangle
                box = triangle.box()

                # Extend the bounding box by a margin determined by the
                # current minimum distance from each point
                box.enlarge(min_dist[i])

                # Check if there are any candidates to consider
                if box.contains(point[None,]):
                    # Compute closest distance on the triangle for all candidates
                    dist, pt = triangle.closest_distance_to(point[None,])

                    # Find candidates where distance to triangle is less than
                    # the previously recorded minimum distance
                    if dist[0] < min_dist[i]:
                        # Update the closest point and minimum distance
                        closest_pt[i] = pt[0]
                        min_dist[i] = dist[0]

        return closest_pt, min_dist
    
    def _vectorized_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a fast vectorized linear search for the closest point
        containing a only single loop over all Triangles in the meshgrid.
        Closest point and distance to the meshgrid for all data points are updated
        at once for each Triangle.
        """

        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Iterate through all the triangles in the meshgrid
        for triangle in meshgrid:
            # Extract the bounding box of the triangle
            box = triangle.box()

            # Extend the bounding box by a margin determined by the
            # current minimum distance from each point
            expanded_min = box.min_xyz.reshape(1, 3) - min_dist.reshape(-1, 1)
            expanded_max = box.max_xyz.reshape(1, 3) + min_dist.reshape(-1, 1)

            # Identify candidate points within the bounding box
            candidates = np.all((expanded_min <= pt_cloud) & \
                                (pt_cloud <= expanded_max), axis=1)

            # Check if there are any candidates to consider
            if candidates.any():
                # Compute closest distance on the triangle for all candidates
                candidate_points = pt_cloud[candidates]
                dist, pt = triangle.closest_distance_to(candidate_points)

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                closer_mask = dist < min_dist[candidates]

                # Select indices where new distances are closer from candidate indices
                indices = np.where(candidates)[0]
                closer_indices = indices[closer_mask]

                # Update the closest point and minimum distance
                min_dist[closer_indices] = dist[closer_mask]
                closest_pt[closer_indices] = pt[closer_mask]

        return closest_pt, min_dist
    
    def _simple_octree_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a simple iterative Octree search to find the
        closest point using a loop over all data points and elements of 
        the tree. Find the closest point and distance to the meshgrid for
        all data points.
        """

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)
        
        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Create a new tree containing triangles from the meshgrid
        tree = Octree(meshgrid.triangles)

        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist
        
        # Search Octree for every point
        for i, pt in enumerate(pt_cloud):
            closest_pt[i], min_dist[i] = self._simple_search_octree(
                pt, tree, closest_pt[i], min_dist[i]
            )

        return closest_pt, min_dist
    
    def _simple_search_octree(
        self,
        point: NDArray[np.float32],
        tree: Octree,
        closest_pt: NDArray[np.float32],
        min_dist: float
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Helper function that recursively searches an Octree to
        find the closest point and distance to the meshgrid, which
        are stored as elements of the Octree.
        """
        
        # Point should be a 3D vector
        if len(point.shape) != 1 or point.shape[0] != 3:
            raise ValueError(f'Point should be a 3D vector of (x, y, z) coordinates!')
        
        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist
        
        # Get the bounding box of the tree
        box = tree.box()
        box.enlarge(min_dist)

        # Stop if there are no candidates to consider
        if not box.contains(point[None,]):
            return closest_pt, min_dist
        
        # Iterate through all elements of the subtree if node is a child
        if not tree.have_subtrees:
            for triangle in tree.elements:
                # Compute closest distance on the triangle for all candidates
                dist, pt = triangle.closest_distance_to(point[None,])

                if dist[0] < min_dist:
                    # Update the closest point and minimum distance
                    closest_pt = pt[0]
                    min_dist = dist[0]

            return closest_pt, min_dist

        # Recursively process all subtrees
        for subtree in tree:
            closest_pt, min_dist = self._simple_search_octree(
                point, subtree, closest_pt, min_dist
            )

        return closest_pt, min_dist
    
    def _vectorized_octree_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        tree: Octree = None,
        closest_pt: NDArray[np.float32] = None,
        min_dist: NDArray[np.float32] = None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a vectorized Octree search to find the
        closest point with a single loop over the elements of 
        the tree. Find the closest point and distance to the meshgrid 
        for all data points.
        """

        # Populate a matrix of closest points on the meshgrid (if necessary)
        if closest_pt is None:
            closest_pt = np.zeros_like(pt_cloud)
        
        # Populate a matrix of closest distances to the meshgrid (if necessary)
        if min_dist is None:
            min_dist = np.empty(pt_cloud.shape[0])
            min_dist.fill(np.inf)

        # Create a new tree containing triangles from the meshgrid (if necessary)
        if tree is None:
            tree = Octree(meshgrid.triangles)

        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist
        
        # Get the bounding box of the tree
        box = tree.box()
        
        # Extend the bounding box by a margin determined by the
        # current minimum distance from each point
        expanded_min = box.min_xyz.reshape(1, 3) - min_dist.reshape(-1, 1)
        expanded_max = box.max_xyz.reshape(1, 3) + min_dist.reshape(-1, 1)

        # Identify candidate points within the bounding box
        candidates = np.all((expanded_min <= pt_cloud) & \
                            (pt_cloud <= expanded_max), axis=1)
        
        # Stop if there are no candidates to consider
        if not candidates.any():
            return closest_pt, min_dist

        closest_pt = closest_pt.copy()
        min_dist = min_dist.copy()
        
        # Iterate through all elements of the subtree if node is a child
        if not tree.have_subtrees:
            for triangle in tree.elements:
                # Compute closest distance on the triangle for all candidates
                candidate_points = pt_cloud[candidates]
                dist, pt = triangle.closest_distance_to(candidate_points)

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                closer_mask = dist < min_dist[candidates]

                # Select indices where new distances are closer from candidate indices
                indices = np.where(candidates)[0]
                closer_indices = indices[closer_mask]

                # Update the closest point and minimum distance
                closest_pt[closer_indices] = pt[closer_mask]
                min_dist[closer_indices] = dist[closer_mask]

            return closest_pt, min_dist
        
        # Recursively process all subtrees
        for subtree in tree:
            closest_pt, min_dist = self._vectorized_octree_match(
                pt_cloud, meshgrid, subtree, closest_pt, min_dist
            )

        return closest_pt, min_dist
    