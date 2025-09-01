# This registry maps model name strings to their class constructors for easy instantiation.
from models.graphaf.model import GraphAFGenerator
from models.graphvae.model import GraphVAE
from models.cvae3d.model import LigandGenerator3D
from models.equifm.model import EquivariantFlowMatchingModel
# RGA is algorithmic, not a typical model class, so it may not be needed here, but define a dummy if needed
# (We will handle RGA generation separately without a model class instance.)

model_registry = {
    'graphaf': GraphAFGenerator,
    'graphvae': GraphVAE,
    'cvae3d': LigandGenerator3D,
    'equifm': EquivariantFlowMatchingModel
    # 'rga': None or a dummy placeholder if needed
}

def get_model_class(name: str):
    return model_registry.get(name, None)
