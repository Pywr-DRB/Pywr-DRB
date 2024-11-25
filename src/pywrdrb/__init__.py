import sys
import os

from pywr.model import Model
from pywr.recorders import *

sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../"))

from pywrdrb.model_builder import ModelBuilder
from pywrdrb.post.output_loader import Output

# Not sure why this is needed, but it is.
from .parameters.ffmp import *
VolBalanceNYCDemand.register()