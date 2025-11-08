import sys
import json

import numpy as np
import matplotlib.pyplot as plt


# Build a slicer, sigle wall, no support, horizontal stripes

nozzle_diameter = 0.2
layer_height = 0.05
num_walls = 1
wall_thickness = 0.2 # one layer of wall therefore same as nozzle diameter


# STL file, read it in some way 
# slice by layer height
# starting position. Gcode value to tell it to move to the corner of the first layer of the STL. 
# some sort of algo to follow along the wall of the STL
# fill in space between walls with horizontal infill
# move up by layer height, update starting position, repeat for each layer.


## STL Parser

def parse_stl_ascii(filename):
    """Parse ASCII STL file and return triangles categorized by orientation"""
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

#NOTE: What if its an inverted cone? How do we handle that?
#NOTE: What about an L shaped. How do you determine when to print the L as a top layer, and split up the other contour as a infill layer

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
        List of line segments representing the slice contour
    """
    segments = []
    
    # Determine if this is the first or last layer
    is_first_layer = abs(z_height - z_min) < tolerance
    is_last_layer = abs(z_height - z_max) < tolerance
    
    # Handle bottom layer - add ONLY perimeter edges (not internal diagonals)
    if is_first_layer:
        # Collect all edges and count occurrences
        edge_count = {}
        for tri in triangles_dict['bottom']:
            v1, v2, v3 = tri['vertices']
            edges = [
                (tuple(v1[:2]), tuple(v2[:2])),
                (tuple(v2[:2]), tuple(v3[:2])),
                (tuple(v3[:2]), tuple(v1[:2]))
            ]
            for edge in edges:
                # Normalize edge (sorted so direction doesn't matter)
                edge_key = tuple(sorted([edge[0], edge[1]]))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
        
        # Only add edges that appear once (perimeter edges, not shared diagonals)
        for edge_key, count in edge_count.items():
            if count == 1:
                segments.append([list(edge_key[0]), list(edge_key[1])])
    
    # Handle top layer - add ONLY perimeter edges (not internal diagonals)
    if is_last_layer:
        # Collect all edges and count occurrences
        edge_count = {}
        for tri in triangles_dict['top']:
            v1, v2, v3 = tri['vertices']
            edges = [
                (tuple(v1[:2]), tuple(v2[:2])),
                (tuple(v2[:2]), tuple(v3[:2])),
                (tuple(v3[:2]), tuple(v1[:2]))
            ]
            for edge in edges:
                # Normalize edge (sorted so direction doesn't matter)
                edge_key = tuple(sorted([edge[0], edge[1]]))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
        
        # Only add edges that appear once (perimeter edges, not shared diagonals)
        for edge_key, count in edge_count.items():
            if count == 1:
                segments.append([list(edge_key[0]), list(edge_key[1])])
    
    # Process regular triangles (walls and angled surfaces)
    for tri in triangles_dict['regular']:
        v1, v2, v3 = tri['vertices']
        intersections = []
        
        # Check each edge of the triangle
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        
        for p1, p2 in edges:
            z1, z2 = p1[2], p2[2]
            
            # Check if edge crosses the z plane
            if (z1 <= z_height <= z2) or (z2 <= z_height <= z1):
                if abs(z2 - z1) < 1e-10:  # ARCHIE NOTE: Edge is parallel to plane, need to ignore it for now, as either another triangle will share that edge or the entire 2D triangle will have to be included in this slice. 
                    continue
                
                # Linear interpolation to find intersection point
                t = (z_height - z1) / (z2 - z1)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                
                intersections.append([x, y])
        
        # Each triangle should intersect at exactly 2 points (or 0)
        if len(intersections) == 2:
            segments.append(intersections)
    
    return segments


def deduplicate_segments_keep_one(segments, epsilon=1e-4):
    """
    Remove duplicate segments, keeping ONE copy of each unique edge.
    
    Unlike removing all duplicates, this keeps the first occurrence.
    This solves the issue where wall triangles and horizontal face triangles
    create the same edges, causing duplicate contours.
    
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
            unique_segments.append(seg)  # Keep first occurrence
    
    removed_count = len(segments) - len(unique_segments)
    if removed_count > 0:
        print(f"    Deduplication: {len(segments)} -> {len(unique_segments)} segments")
        print(f"    Removed {removed_count} duplicate segments")
    
    return unique_segments


def calculate_signed_area(contour):
    """
    Calculate signed area of a contour using the Shoelace formula.
    
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
    Ray casting algorithm to check if point is inside polygon.
    
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
    Classify contours by nesting level using containment.
    Even levels (0, 2, 4...) = solid regions
    Odd levels (1, 3, 5...) = holes
    
    Args:
        contours: List of contours (each contour is a list of points)
    
    Returns:
        List of islands, each with:
        {
            'outer': outer_contour,
            'level': 0,
            'children': [list of contours at deeper levels]
        }
    """
    if not contours:
        return []
    
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
            
            # Debug output for troubleshooting
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


def build_contours_nearest_neighbor(segments, epsilon=1e-4):
    """
    Build contours from segments using naive nearest-neighbor chaining.
    
    Args:
        segments: List of line segments [[x1, y1], [x2, y2]]
        epsilon: Distance tolerance for connecting segments
    
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
# NOTE: Contours is fine. How do you pick between 1-3 contours for base layer. If its a donut, there are 2 contours. What happens then? How do we account for it.
















def plot_layer(segments, contours, z_height, title="Layer"):
    """
    Plot the segments and contours for a layer.
    
    Args:
        segments: Raw segments from slicing
        contours: Built contours from segments
        z_height: Z height of the layer
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot raw segments
    ax1.set_title(f"{title} - Raw Segments (n={len(segments)})")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    for seg in segments:
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.5)
        # Mark start points
        ax1.plot(seg[0][0], seg[0][1], 'go', markersize=3)
        # Mark end points
        ax1.plot(seg[1][0], seg[1][1], 'ro', markersize=3)
    
    # Plot contours with nesting classification
    ax2.set_title(f"{title} - Contours with Classification (n={len(contours)})")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Try to classify contours for visualization
    try:
        if contours:
            islands, contours_data = organize_by_nesting_levels(contours)
            
            # Plot solids in blue, holes in red
            for item in contours_data:
                contour = item['contour']
                contour_closed = contour + [contour[0]]
                x_coords = [p[0] for p in contour_closed]
                y_coords = [p[1] for p in contour_closed]
                
                if item['is_hole']:
                    # Holes in red, dashed
                    ax2.plot(x_coords, y_coords, 'r--', linewidth=2, 
                            label=f"HOLE L{item['level']} ({len(contour)} pts)")
                    ax2.plot(x_coords, y_coords, 'ro', markersize=4)
                else:
                    # Solids in blue, solid line
                    ax2.plot(x_coords, y_coords, 'b-', linewidth=2, 
                            label=f"SOLID L{item['level']} ({len(contour)} pts)")
                    ax2.plot(x_coords, y_coords, 'bo', markersize=4)
            
            ax2.legend(fontsize=8, loc='best')
    except:
        # Fallback to simple plotting if classification fails
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(contours), 1)))
        
        for idx, contour in enumerate(contours):
            if not contour:
                continue
            
            contour_closed = contour + [contour[0]]
            x_coords = [p[0] for p in contour_closed]
            y_coords = [p[1] for p in contour_closed]
            
            ax2.plot(x_coords, y_coords, '-', color=colors[idx], linewidth=2, 
                    label=f"Contour {idx+1} ({len(contour)} pts)")
            ax2.plot(x_coords, y_coords, 'o', color=colors[idx], markersize=4)
        
        if contours:
            ax2.legend(fontsize=8)
    
    plt.suptitle(f"Layer at Z = {z_height} mm", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# --- TEST CODE ---
if __name__ == "__main__":
    stl_file = "Hole with Rod.STL"
    
    print("=" * 60)
    print("STL SLICER - CONTOUR BUILDING TEST")
    print("=" * 60)
    
    print("\nParsing STL...")
    triangles_dict = parse_stl_ascii(stl_file)
    
    print(f"\nTriangle categorization:")
    print(f"  Regular triangles (walls/angled): {len(triangles_dict['regular'])}")
    print(f"  Bottom triangles (normal z=-1): {len(triangles_dict['bottom'])}")
    print(f"  Top triangles (normal z=1): {len(triangles_dict['top'])}")
    print(f"  Total: {len(triangles_dict['regular']) + len(triangles_dict['bottom']) + len(triangles_dict['top'])}")
    
    # Find Z bounds
    all_triangles = triangles_dict['regular'] + triangles_dict['bottom'] + triangles_dict['top']
    all_z = [v[2] for tri in all_triangles for v in tri['vertices']]
    z_min, z_max = min(all_z), max(all_z)
    
    print(f"\nModel bounds: Z from {z_min} to {z_max}")
    
    # Test slicing at bottom, near-bottom, middle, and top
    test_layers = [
        (z_min, "Bottom Layer (z=zmin)"),
        (z_min + 1.5, "Second Layer (z=0.1)"),
        ((z_max-z_min)/2 + z_min, "Middle Layer"),
        (z_max, "Top Layer (z=zmax)")
    ]
    
    print("\n" + "=" * 60)
    print("SLICING AND BUILDING CONTOURS")
    print("=" * 60)
    
    for z, label in test_layers:
        print(f"\n{label} (z={z}):")
        
        # Slice at this Z height
        segments = slice_at_z(triangles_dict, z, z_min, z_max)
        print(f"  Raw segments: {len(segments)}")
        
        # Deduplicate segments (keep one copy of each edge)
        segments_unique = deduplicate_segments_keep_one(segments)

        # Build contours using nearest-neighbor chaining
        contours = build_contours_nearest_neighbor(segments_unique, epsilon=0.01)
        print(f"  Contours found: {len(contours)}")
        
        # Classify contours by nesting level
        if contours:
            islands, contours_data = organize_by_nesting_levels(contours)
            
            print(f"\n  Nesting Classification:")
            print(f"    Islands (level 0): {len(islands)}")
            
            for i, island in enumerate(islands):
                print(f"\n    Island {i+1}:")
                print(f"      Outer contour: {len(island['outer'])} points, area={island['area']:.2f}")
                print(f"      Holes: {len(island['holes'])}")
                for j, hole in enumerate(island['holes']):
                    hole_area = abs(calculate_signed_area(hole))
                    print(f"        Hole {j+1}: {len(hole)} points, area={hole_area:.2f}")
                print(f"      Inner solids: {len(island['inner_solids'])}")
                for j, solid in enumerate(island['inner_solids']):
                    solid_area = abs(calculate_signed_area(solid))
                    print(f"        Solid {j+1}: {len(solid)} points, area={solid_area:.2f}")
            
            print(f"\n  All contours by nesting level:")
            for idx, item in enumerate(contours_data):
                level_type = "HOLE" if item['is_hole'] else "SOLID"
                test_pt = item['contour'][0]
                print(f"    #{idx+1} Level {item['level']} ({level_type}): {len(item['contour'])} pts, area={item['area']:.2f}, test_pt=({test_pt[0]:.1f},{test_pt[1]:.1f})")
        
        # Plot the layer (show deduplicated segments and classified contours)
        plot_layer(segments_unique, contours, z, label)
    
    print("\n" + "=" * 60)
    print("Displaying plots...")
    plt.show()

