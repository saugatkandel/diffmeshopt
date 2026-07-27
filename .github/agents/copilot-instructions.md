# Repository instructions

This is a PyTorch scientific computing project.

Important conventions:
- Use torch tensors rather than numpy inside optimization loops. Otherwise, make functions usable with both by converting to torch tensors at the start of the function.
- Preserve CPU/GPU compatibility, and MPS compatibility when possible.
- Avoid hidden global state.
- Prefer explicit tensor shapes.

Before modifying:
- inspect related modules
- understand tensor shapes
- check existing tests