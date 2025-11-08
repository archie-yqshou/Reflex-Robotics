"""
STL slicing and contour generation algorithms

This module handles:
- STL file parsing
- Slicing at specific Z heights
- Contour building from segments
- Nesting level classification (islands, holes, inner solids)
"""

import numpy as np


def parse_stl_ascii(filename):
    """
    Parse ASCII STL file and return triangles categorized by orientation.
    
    Args:
        filename: Path to ASCII STL file
    
    Returns:
        Dictionary with 'regular', 'bottom', and 'top' triangle lists
    """
    regular_triangles = []
    bottom_triangles = []  # Faces pointing down (normal z = -1)
    top_triangles = []     # Faces pointing up (normal z = 1)
    
    with open(filename, 'r') as f:
        current_triangle = {'normal': None, 'vertices': []}
        
        for line in f:
            line = line.strip()
            
            if line.startswith('facet normal'):
                parts = line.split()
                current_triangle['normal'] = [float(parts[2]), float(parts[3]), float(parts[4])]
            
            elif line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_triangle['vertices'].append(vertex)
            
            elif line.startswith('endfacet'):
                if len(current_triangle['vertices']) == 3:
                    normal_z = current_triangle['normal'][2]
                    
                    # Categorize based on normal direction
                    if abs(normal_z - 1.0) < 1e-6:  # Pointing up (top face)
                        top_triangles.append(current_triangle)
                    elif abs(normal_z + 1.0) < 1e-6:  # Pointing down (bottom face)
                        bottom_triangles.append(current_triangle)
                    else:  # Angled or vertical faces
                        regular_triangles.append(current_triangle)
                
                current_triangle = {'normal': None, 'vertices': []}
    
    return {
        'regular': regular_triangles,
        'bottom': bottom_triangles,
        'top': top_triangles
    }


def get_model_bounds(triangles_dict):
    """
    Get the Z bounds (min and max) of the model.
    
    Args:
        triangles_dict: Dictionary from parse_stl_ascii
    
    Returns:
        Tuple of (z_min, z_max)
    """
    all_triangles = triangles_dict['regular'] + triangles_dict['bottom'] + triangles_dict['top']
    all_z = [v[2] for tri in all_triangles for v in tri['vertices']]
    return min(all_z), max(all_z)


def slice_at_z(triangles_dict, z_height, z_min, z_max, tolerance=1e-3):
    """
    Slice triangles at a given z height.
    
    Args:
        triangles_dict: Dict with 'regular', 'bottom', and 'top' triangle lists
        z_height: The Z plane to slice at
        z_min: Minimum Z value in the model
        z_max: Maximum Z value in the model
        tolerance: Tolerance for comparing Z values
    
    Returns:
        Tuple of (segments, has_horizontal_faces)
        - segments: List of line segments [[x1, y1], [x2, y2]]
        - has_horizontal_faces: Boolean indicating if this layer has horizontal surfaces
    """
    segments = []
    has_horizontal_faces = False
    
    # Determine if this is the first or last layer
    is_first_layer = abs(z_height - z_min) < tolerance
    is_last_layer = abs(z_height - z_max) < tolerance
    
    # Check for horizontal faces at this Z height (bottom-facing triangles)
    edge_count = {}
    for tri in triangles_dict['bottom']:
        v1, v2, v3 = tri['vertices']
        # Check if triangle is at this Z height
        tri_z = (v1[2] + v2[2] + v3[2]) / 3
        if abs(tri_z - z_height) < tolerance:
            has_horizontal_faces = True
            edges = [
                (tuple(v1[:2]), tuple(v2[:2])),
                (tuple(v2[:2]), tuple(v3[:2])),
                (tuple(v3[:2]), tuple(v1[:2]))
            ]
            for edge in edges:
                edge_key = tuple(sorted([edge[0], edge[1]]))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    
    # Check for horizontal faces at this Z height (top-facing triangles)
    for tri in triangles_dict['top']:
        v1, v2, v3 = tri['vertices']
        # Check if triangle is at this Z height
        tri_z = (v1[2] + v2[2] + v3[2]) / 3
        if abs(tri_z - z_height) < tolerance:
            has_horizontal_faces = True
            edges = [
                (tuple(v1[:2]), tuple(v2[:2])),
                (tuple(v2[:2]), tuple(v3[:2])),
                (tuple(v3[:2]), tuple(v1[:2]))
            ]
            for edge in edges:
                edge_key = tuple(sorted([edge[0], edge[1]]))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    
    # Only add perimeter edges from horizontal faces (edges that appear once)
    for edge_key, count in edge_count.items():
        if count == 1:
            segments.append([list(edge_key[0]), list(edge_key[1])])
    


    # Regular layers
    for tri in triangles_dict['regular']:
        v1, v2, v3 = tri['vertices']
        intersections = []
        
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        
        for p1, p2 in edges:
            z1, z2 = p1[2], p2[2]
            
            # Check if edge crosses the z plane
            if (z1 <= z_height <= z2) or (z2 <= z_height <= z1):
                if abs(z2 - z1) < 1e-10:  # Edge is parallel to plane
                    continue
                
                # Linear interpolation to find intersection point
                t = (z_height - z1) / (z2 - z1)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                
                intersections.append([x, y])
        
        # Each triangle should intersect at exactly 2 points (or 0)
        if len(intersections) == 2:
            segments.append(intersections)
    
    return segments, has_horizontal_faces


def deduplicate_segments_keep_one(segments, epsilon=1e-4):
    """
    Remove duplicate segments, keeping ONE copy of each unique edge.
    
    Args:
        segments: List of line segments [[x1, y1], [x2, y2]]
        epsilon: Tolerance for comparing coordinates (not used with rounding)
    
    Returns:
        List of unique segments (first occurrence kept)
    """
    if not segments:
        return []
    
    unique_segments = []
    seen_edges = set()
    
    for seg in segments:
        # Normalize edge (round and sort so direction/precision doesn't matter)
        p1 = tuple(np.round(seg[0], decimals=4))
        p2 = tuple(np.round(seg[1], decimals=4))
        edge_key = tuple(sorted([p1, p2]))
        
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_segments.append(seg)
    
    return unique_segments


def build_contours_nearest_neighbor(segments, epsilon=0.01):
    """
    Build contours from segments using nearest-neighbor chaining.
    
    Args:
        segments: List of line segments [[x1, y1], [x2, y2]]
        epsilon: Distance tolerance for connecting segments (mm)
    
    Returns:
        List of contours, where each contour is a list of points
    """
    if not segments:
        return []
    
    # Convert segments to a list we can modify
    remaining_segments = [seg.copy() for seg in segments]
    contours = []
    
    while remaining_segments:
        # Start a new contour with the first remaining segment
        current_segment = remaining_segments.pop(0)
        contour = [current_segment[0], current_segment[1]]
        
        # Keep chaining until we can't find a match or close the loop
        max_iterations = len(segments) * 2  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            current_end = contour[-1]
            found_match = False
            
            # Search for a segment whose start or end is close to current_end
            for i, seg in enumerate(remaining_segments):
                dist_to_start = np.sqrt((seg[0][0] - current_end[0])**2 + 
                                       (seg[0][1] - current_end[1])**2)
                dist_to_end = np.sqrt((seg[1][0] - current_end[0])**2 + 
                                     (seg[1][1] - current_end[1])**2)
                
                if dist_to_start < epsilon:
                    # Connect to start of segment
                    contour.append(seg[1])
                    remaining_segments.pop(i)
                    found_match = True
                    break
                elif dist_to_end < epsilon:
                    # Connect to end of segment (reversed)
                    contour.append(seg[0])
                    remaining_segments.pop(i)
                    found_match = True
                    break
            
            # Check if contour is closed (end connects to start)
            if len(contour) > 2:
                dist_to_start = np.sqrt((contour[-1][0] - contour[0][0])**2 + 
                                       (contour[-1][1] - contour[0][1])**2)
                if dist_to_start < epsilon:
                    # Contour is closed, remove duplicate end point
                    contour.pop()
                    break
            
            # If no match found, this contour is done (might be unclosed)
            if not found_match:
                break
        
        if len(contour) >= 3:  # Only keep contours with at least 3 points
            contours.append(contour)
    
    return contours


def calculate_signed_area(contour):
    """
    Does not need to be signed. We absolute value it. 
    
    Args:
        contour: List of [x, y] points
    
    Returns:
        Positive area = counter-clockwise winding
        Negative area = clockwise winding
        Zero = degenerate (line or point)
    """
    if len(contour) < 3:
        return 0.0
    
    area = 0.0
    n = len(contour)
    
    for i in range(n):
        j = (i + 1) % n  # Next vertex (wraps around to start)
        area += contour[i][0] * contour[j][1]  # x_i * y_j
        area -= contour[j][0] * contour[i][1]  # x_j * y_i
    
    return area / 2.0


def point_in_polygon(point, polygon):
    """
    Check if points are inside. For nested shapes
    
    Args:
        point: [x, y] coordinates
        polygon: List of [x, y] points forming a closed polygon
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        # Check if ray from point crosses this edge
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def organize_by_nesting_levels(contours):
    """
    Classify contours by nesting level using geometric containment.
    Even levels (0, 2, 4...) = solid regions
    Odd levels (1, 3, 5...) = holes
    
    Args:
        contours: List of contours (each contour is a list of points)
    
    Returns:
        Tuple of (islands, contours_with_data)
        - islands: List of dicts with 'outer', 'holes', 'inner_solids'
        - contours_with_data: List of dicts with contour info and nesting level
    """
    if not contours:
        return [], []
    
    # Step 1: Calculate absolute area and sort (largest first)
    contours_with_data = []
    for c in contours:
        area = calculate_signed_area(c)
        if abs(area) < 1e-6:  # Skip degenerate contours
            continue
        contours_with_data.append({
            'contour': c,
            'area': abs(area),
            'level': None
        })
    
    contours_with_data.sort(key=lambda x: x['area'], reverse=True)
    
    # Step 2: Determine nesting level for each contour
    for i, item in enumerate(contours_with_data):
        # Count how many larger contours contain this one
        nesting_level = 0
        test_point = item['contour'][0]
        
        for j in range(i):  # Only check larger contours
            larger = contours_with_data[j]
            is_inside = point_in_polygon(test_point, larger['contour'])
            
            if is_inside:
                nesting_level += 1
        
        item['level'] = nesting_level
        item['is_hole'] = (nesting_level % 2 == 1)  # Odd = hole
    
    # Step 3: Group by top-level islands (level 0)
    islands = []
    
    for item in contours_with_data:
        if item['level'] == 0:
            islands.append({
                'outer': item['contour'],
                'area': item['area'],
                'holes': [],
                'inner_solids': []
            })
    
    # Step 4: Assign holes and inner solids to their parent islands
    for item in contours_with_data:
        if item['level'] > 0:
            # Find which level-0 island it belongs to
            for island in islands:
                if point_in_polygon(item['contour'][0], island['outer']):
                    if item['is_hole']:
                        island['holes'].append(item['contour'])
                    else:
                        island['inner_solids'].append(item['contour'])
                    break
    
    return islands, contours_with_data


def generate_crosshatch_infill(island, line_spacing=0.4):
    """
    Generate cross-hatch (solid) infill for top/bottom layers and horizontal surfaces.
    Combines horizontal lines with perpendicular (vertical) lines for 100% fill.
    
    Args:
        island: Dict with 'outer', 'holes', 'inner_solids' keys
        line_spacing: Spacing between infill lines in mm
    
    Returns:
        List of line segments [[x1, y1], [x2, y2]] representing infill
    """
    infill_lines = []
    
    # Generate horizontal lines
    horizontal_lines = generate_horizontal_infill(island, line_spacing)
    infill_lines.extend(horizontal_lines)
    
    # Generate vertical lines (perpendicular to horizontal)
    vertical_lines = generate_vertical_infill(island, line_spacing)
    infill_lines.extend(vertical_lines)
    
    return infill_lines


def generate_vertical_infill(island, line_spacing=0.4):
    """
    Generate vertical infill lines for an island (solid region with holes).
    
    Args:
        island: Dict with 'outer', 'holes', 'inner_solids' keys
        line_spacing: Spacing between infill lines in mm
    
    Returns:
        List of line segments [[x1, y1], [x2, y2]] representing vertical infill
    """
    infill_lines = []
    
    outer_contour = island['outer']
    holes = island.get('holes', [])
    inner_solids = island.get('inner_solids', [])
    
    # Get X bounds of the outer contour
    x_coords = [p[0] for p in outer_contour]
    x_min, x_max = min(x_coords), max(x_coords)
    
    # Generate vertical lines at regular intervals
    current_x = x_min + line_spacing
    
    while current_x < x_max:
        # Find all intersections of this vertical line with contours
        intersections = []
        
        # Intersections with outer boundary
        outer_intersections = find_vertical_contour_intersections(outer_contour, current_x)
        intersections.extend([(y, 'outer') for y in outer_intersections])
        
        # Intersections with holes
        for hole in holes:
            hole_intersections = find_vertical_contour_intersections(hole, current_x)
            intersections.extend([(y, 'hole') for y in hole_intersections])
        
        # Sort intersections by Y coordinate
        intersections.sort(key=lambda p: p[0])
        
        # Build line segments: inside outer, outside holes, outside inner solids
        if len(intersections) >= 2:
            i = 0
            while i < len(intersections) - 1:
                y1, type1 = intersections[i]
                y2, type2 = intersections[i + 1]
                
                # Check if midpoint is in a valid region
                mid_y = (y1 + y2) / 2
                mid_point = [current_x, mid_y]
                
                # Must be inside outer and outside all holes
                in_outer = point_in_polygon(mid_point, outer_contour)
                in_hole = any(point_in_polygon(mid_point, hole) for hole in holes)
                
                # Don't fill inner solids (they get their own infill)
                in_inner_solid = any(point_in_polygon(mid_point, solid) for solid in inner_solids)
                
                if in_outer and not in_hole and not in_inner_solid:
                    infill_lines.append([[current_x, y1], [current_x, y2]])
                
                i += 1
        
        current_x += line_spacing
    
    # Generate infill for inner solids too
    for inner_solid in inner_solids:
        mini_island = {
            'outer': inner_solid,
            'holes': [],
            'inner_solids': []
        }
        inner_infill = generate_vertical_infill(mini_island, line_spacing)
        infill_lines.extend(inner_infill)
    
    return infill_lines


def generate_horizontal_infill(island, line_spacing=0.4):
    """
    Generate horizontal infill lines for an island (solid region with holes).
    
    Args:
        island: Dict with 'outer', 'holes', 'inner_solids' keys
        line_spacing: Spacing between infill lines in mm
    
    Returns:
        List of line segments [[x1, y1], [x2, y2]] representing infill
    """
    infill_lines = []
    
    outer_contour = island['outer']
    holes = island.get('holes', [])
    inner_solids = island.get('inner_solids', [])
    
    # Get Y bounds of the outer contour
    y_coords = [p[1] for p in outer_contour]
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Generate horizontal lines at regular intervals
    current_y = y_min + line_spacing
    
    while current_y < y_max:
        # Find all intersections of this horizontal line with contours
        intersections = []
        
        # Intersections with outer boundary
        outer_intersections = find_contour_intersections(outer_contour, current_y)
        intersections.extend([(x, 'outer') for x in outer_intersections])
        
        # Intersections with holes
        for hole in holes:
            hole_intersections = find_contour_intersections(hole, current_y)
            intersections.extend([(x, 'hole') for x in hole_intersections])
        
        # Sort intersections by X coordinate
        intersections.sort(key=lambda p: p[0])
        
        # Build line segments: inside outer, outside holes, outside inner solids
        if len(intersections) >= 2:
            i = 0
            while i < len(intersections) - 1:
                x1, type1 = intersections[i]
                x2, type2 = intersections[i + 1]
                
                # Check if midpoint is in a valid region
                mid_x = (x1 + x2) / 2
                mid_point = [mid_x, current_y]
                
                # Must be inside outer and outside all holes
                in_outer = point_in_polygon(mid_point, outer_contour)
                in_hole = any(point_in_polygon(mid_point, hole) for hole in holes)
                
                # Infill solid regions but NOT inner solids (those get their own infill)
                in_inner_solid = any(point_in_polygon(mid_point, solid) for solid in inner_solids)
                
                if in_outer and not in_hole and not in_inner_solid:
                    infill_lines.append([[x1, current_y], [x2, current_y]])
                
                i += 1
        
        current_y += line_spacing
    
    for inner_solid in inner_solids:
        mini_island = {
            'outer': inner_solid,
            'holes': [],
            'inner_solids': []
        }
        inner_infill = generate_horizontal_infill(mini_island, line_spacing)
        infill_lines.extend(inner_infill)
    
    return infill_lines


def find_contour_intersections(contour, y_line):
    """
    Find X coordinates where a horizontal line intersects a contour.
    
    Args:
        contour: List of [x, y] points forming a polygon
        y_line: Y coordinate of the horizontal line
    
    Returns:
        List of X coordinates where the line intersects the contour
    """
    intersections = []
    n = len(contour)
    
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]  # Next point (wraps around)
        
        y1, y2 = p1[1], p2[1]
        
        # Check if edge crosses the horizontal line
        if (y1 <= y_line <= y2) or (y2 <= y_line <= y1):
            if abs(y2 - y1) < 1e-10:  # Edge is horizontal
                # Edge is on the line, add both endpoints
                intersections.append(p1[0])
                intersections.append(p2[0])
            else:
                # Calculate intersection X coordinate
                t = (y_line - y1) / (y2 - y1)
                x_intersect = p1[0] + t * (p2[0] - p1[0])
                intersections.append(x_intersect)
    
    # Remove duplicates and sort
    intersections = sorted(set(np.round(intersections, decimals=4)))
    
    return intersections


def find_vertical_contour_intersections(contour, x_line):
    """
    Find Y coordinates where a vertical line intersects a contour.
    
    Args:
        contour: List of [x, y] points forming a polygon
        x_line: X coordinate of the vertical line
    
    Returns:
        List of Y coordinates where the line intersects the contour
    """
    intersections = []
    n = len(contour)
    
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]  # Next point (wraps around)
        
        x1, x2 = p1[0], p2[0]
        
        # Check if edge crosses the vertical line
        if (x1 <= x_line <= x2) or (x2 <= x_line <= x1):
            if abs(x2 - x1) < 1e-10:  # Edge is vertical
                # Edge is on the line, add both endpoints
                intersections.append(p1[1])
                intersections.append(p2[1])
            else:
                # Calculate intersection Y coordinate
                t = (x_line - x1) / (x2 - x1)
                y_intersect = p1[1] + t * (p2[1] - p1[1])
                intersections.append(y_intersect)
    
    # Remove duplicates and sort
    intersections = sorted(set(np.round(intersections, decimals=4)))
    
    return intersections

