from torch.utils.cpp_extension import load

match_propagate = load(name='match_propagate', sources=['match_propagate.cpp', 'match_propagate_kernel.cu'])
