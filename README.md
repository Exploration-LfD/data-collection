# data-collection
Collect human trajectories

### Usage
To record images, states, and actions, use `record_human_trajectories.py` as shown below:
`python record_human_trajectories.py --env ENV --save_dir DIR`

You can load a particular state that you've saved using the modified `manual_control.py`. Ex:
`python manual_control.py --env ENV --load DIR/00002/00016`