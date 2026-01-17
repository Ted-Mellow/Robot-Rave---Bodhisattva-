# CSV Trajectories

Place your trajectory CSV files here.

## CSV Format

CSV files should have the following columns:
- `time`: Time in seconds
- `joint1` to `joint6`: Joint angles in radians
- `description`: (Optional) Description of the waypoint

### Example

```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,0.0,0.0,0.0,0.0,0.0,Home position
1.0,0.5,1.0,-1.0,0.5,0.5,0.5,Target position
2.0,0.0,0.0,0.0,0.0,0.0,0.0,Return home
```

## Joint Limits (Radians)

| Joint | Min    | Max   |
|-------|--------|-------|
| J1    | -2.62  | 2.62  |
| J2    | 0.00   | 3.14  |
| J3    | -2.97  | 0.00  |
| J4    | -1.75  | 1.75  |
| J5    | -1.22  | 1.22  |
| J6    | -2.09  | 2.09  |

## Running Trajectories

```bash
python run_csv_trajectory.py csv_trajectories/example_wave.csv
```
