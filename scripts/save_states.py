# save_states.py ─ save every state of a multi-state object as
#                  <object>_state<N>.pdb in the current directory
#
# ── From a terminal  (batch):
#     pymol -cq save_states.py -- your_multistate.pdb
#
# ── From an interactive PyMOL session:
#     PyMOL> run save_states.py
#     PyMOL> load your_multistate.pdb
#     PyMOL> save_states            # uses the first object
#     PyMOL> save_states myobj      # or name object explicitly
#

from __future__ import print_function
from pymol import cmd
import os, sys

def save_states(obj_name=None, prefix=None):
    """
    Save every state of *obj_name* as an individual PDB.

    Parameters
    ----------
    obj_name : str, optional
        The object whose states will be saved.  Defaults to first object.
    prefix : str, optional
        Filename prefix (defaults to obj_name).
    """
    # Default to first loaded object
    if obj_name is None:
        objects = cmd.get_object_list()
        if not objects:
            raise RuntimeError("No objects are loaded.")
        obj_name = objects[0]

    if prefix is None:
        prefix = obj_name

    n_states = cmd.count_states(obj_name)
    if n_states == 0:
        raise RuntimeError("Object '%s' has no states." % obj_name)

    for state in range(1, n_states + 1):
        fname = "%s_state%d.pdb" % (prefix, state)
        cmd.save(fname, obj_name, state=state)
        print("Saved", fname)

# expose as PyMOL command
cmd.extend("save_states", save_states)

# ---------- batch use ----------
if __name__ == "__main__":
    # run via:  pymol -cq save_states.py -- multistate.pdb
    # (PyMOL passes everything after '--' to the script)
    args = [a for a in sys.argv[1:] if a != "--"]
    if args:
        pdb_path = args[0]
        obj = os.path.splitext(os.path.basename(pdb_path))[0]
        cmd.load(pdb_path, obj)
        save_states(obj)
        cmd.quit()
