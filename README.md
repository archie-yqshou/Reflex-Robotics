1. **Set your STL file** in `run.py`:
   ```python
   STL_FILE = "STLs/YourFile.stl"
   ```

2. **Run the slicer**:
   ```bash
   python run.py
   ```

3. **View the results**: The program will display visualizations for each test layer showing:
   - Raw segments from slicing
   - Classified contours (outer perimeters, holes, inner solids)
   - Infill patterns (cross-hatch for solid layers, sparse for regular layers)

## Choosing Layers to Slice

Edit the `test_layers` list in `run.py` to select which Z-heights to visualize:

```python
test_layers = [
    (z_min, "Bottom Layer"),           # Bottom of model
    (25, "25mm"),                      # Specific height (25mm)
    ((z_max - z_min) / 2 + z_min, "Middle Layer"),  # Middle of model
    (z_max, "Top Layer")               # Top of model
]
```
## Requirements

- Python 3.x
- NumPy
- Matplotlib

## File Structure

- `run.py` - Main script (configure and run here)
- `geometry.py` - Slicing algorithms
- `visualization.py` - Plotting functions
- `STLs/` - Place your STL files here

