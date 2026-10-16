The ecmp-sim directory contains the Python code for our simulation, to explore various questions as described in the proposal document.
The ns-3-sim directory contains the relevant scratch code for the ns-3 simulation component.

To run the ns-3 related experiments, download ns-3 version 3.49, and cd into ns-3.49
Then run rm -rf scratch to remove the default provided scratch directory, then symlink the ns-3-sim directory provided in this repo to a symlink called scratch inside your ns3 directory.

For example:
cd /path/to/ns-3.49
rm -rf scratch
ln -s ~/path/to/cs3821-assignment-code/ns-3-sim scratch

Instructions to run the ns-3 simulation coming soon.
