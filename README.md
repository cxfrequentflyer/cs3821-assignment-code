# COMP3821 26T3 - Assignment Code

## Information:
The ```ecmp-sim``` directory contains the Python code for our simulation, to explore various questions as described in the proposal document.
The ```dynamic-sim``` directory contains the C++ discrete-event (queueing) simulator.
The ```ns-3-sim``` directory contains the ns-3 simulation component: the `lb-fabric` program and its experiment scripts.

## ns-3 Simulations:
To run the ns-3 related experiments, download ns-3 version ```3.49``` and ```cd``` into ```ns-3.49```.
Build it like this:

```bash
./ns3 configure --enable-examples --enable-tests
./ns3 build
./ns3 run first # to check that the core simulator is working correctly
```
Then run ```rm -rf scratch``` to remove the default provided ```scratch``` directory, then symlink the ```ns-3-sim``` directory provided in this repo to a symlink called ```scratch``` inside your ```ns-3.49``` directory.

For example:
```bash
cd /path/to/ns-3.49
rm -rf scratch
ln -s ~/path/to/cs3821-assignment-code/ns-3-sim scratch
```

Then build and run the load-balancing simulation:
```bash
./ns3 configure -d optimized
./ns3 build lb-fabric
./ns3 run "lb-fabric --scheme=drill --load=0.6 --flows=1000"
```
`ns-3-sim/README.md` documents `lb-fabric` (a leaf-spine fabric with ECMP, WCMP,
flowlets, CONGA, Presto, per-packet and DRILL load balancing) and the
`ns-3-sim/experiments/` scripts (ns01-ns09) that rerun the `ecmp-sim` and
`dynamic-sim` experiments at packet level.
