import numpy as np
import matplotlib.pyplot as plt
from geometry import calculate_signed_area, organize_by_nesting_levels, generate_horizontal_infill, generate_crosshatch_infill


def plot_layer(segments, contours, z_height, title="Layer", infill_spacing=0.4, has_horizontal_faces=False):
    """
    Plot the segments, contours, and infill for a layer.
    
    Args:
        segments: Raw segments from slicing
        contours: Built contours from segments
        z_height: Z height of the layer
        title: Title for the plot
        infill_spacing: Spacing between infill lines in mm
    
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
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
        # Mark start points (green)
        ax1.plot(seg[0][0], seg[0][1], 'go', markersize=3)
        # Mark end points (red)
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
    
    # Plot infill pattern (third subplot)
    infill_type = "Cross-Hatch (Solid)" if has_horizontal_faces else "Sparse"
    ax3.set_title(f"{title} - {infill_type} Infill (spacing={infill_spacing}mm)")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Generate and plot infill
    try:
        if contours:
            islands, contours_data = organize_by_nesting_levels(contours)
            
            total_infill_lines = 0
            
            # Generate infill for each island
            for island in islands:
                # Use cross-hatch for solid layers, horizontal for sparse layers
                if has_horizontal_faces:
                    infill_lines = generate_crosshatch_infill(island, line_spacing=infill_spacing)
                else:
                    infill_lines = generate_horizontal_infill(island, line_spacing=infill_spacing)
                
                # Plot infill lines in blue
                for line in infill_lines:
                    x_coords = [line[0][0], line[1][0]]
                    y_coords = [line[0][1], line[1][1]]
                    ax3.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.6)
                
                total_infill_lines += len(infill_lines)
                
                # Plot outer contour as reference (light gray)
                outer_closed = island['outer'] + [island['outer'][0]]
                x_coords = [p[0] for p in outer_closed]
                y_coords = [p[1] for p in outer_closed]
                ax3.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.3)
                
                # Plot holes as reference (light red)
                for hole in island['holes']:
                    hole_closed = hole + [hole[0]]
                    x_coords = [p[0] for p in hole_closed]
                    y_coords = [p[1] for p in hole_closed]
                    ax3.plot(x_coords, y_coords, 'r-', linewidth=1, alpha=0.3)
                
                # Plot inner solids outline (light green)
                for solid in island['inner_solids']:
                    solid_closed = solid + [solid[0]]
                    x_coords = [p[0] for p in solid_closed]
                    y_coords = [p[1] for p in solid_closed]
                    ax3.plot(x_coords, y_coords, 'g-', linewidth=1, alpha=0.3)
            
            ax3.text(0.02, 0.98, f"Infill lines: {total_infill_lines}", 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        ax3.text(0.5, 0.5, f"Infill generation failed:\n{str(e)}", 
                transform=ax3.transAxes, fontsize=10, ha='center', va='center')
    
    plt.suptitle(f"Layer at Z = {z_height} mm", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_layer_info(z, label, segments, contours, z_min, z_max):
    """
    Print detailed information about a sliced layer.
    
    Args:
        z: Z height
        label: Layer label
        segments: Raw segments
        contours: Built contours
        z_min: Minimum Z of model
        z_max: Maximum Z of model
    """
    print(f"\n{label} (z={z:.2f}):")
    print(f"  Raw segments: {len(segments)}")
    
    # Show deduplication info if applicable
    is_first_or_last = abs(z - z_min) < 0.001 or abs(z - z_max) < 0.001
    if is_first_or_last:
        print(f"  (First/last layer - includes horizontal faces)")
    
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
            print(f"    #{idx+1} Level {item['level']} ({level_type}): {len(item['contour'])} pts, "
                  f"area={item['area']:.2f}, test_pt=({test_pt[0]:.1f},{test_pt[1]:.1f})")

