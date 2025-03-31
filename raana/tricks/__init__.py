from .base       import Trick
from .centralize import TrickCentralize
from .principle  import TrickPCA
from .norm_row   import TrickNormRow
from .norm_col   import TrickNormCol

trick_centralize = lambda: TrickCentralize()
trick_pca        = lambda: TrickPCA()
trick_norm_row   = lambda: TrickNormRow()
trick_norm_col   = lambda: TrickNormCol()



