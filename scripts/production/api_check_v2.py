"""Verifica que la versión de u25 soporte el evaluador primario y las variantes de v2_api.json (run_2b_v2.sh).
Sale con 1 si el primario no está soportado; una variante no soportada sólo se avisa."""
import sys

import u25.objectives as O
from f2b_model_v2 import EV, primary_kw, variant_kw

try:
    O.validate_eval_kw(primary_kw())
except Exception as e:  # noqa: BLE001
    print(f"ERROR: evaluador primario {primary_kw()} no soportado por u25: {e}")
    sys.exit(1)
print(f"evaluador primario: {primary_kw()}")
for v in EV.get("variantes", {}):
    kw = {k: x for k, x in variant_kw(v).items() if not k.startswith("_")}
    try:
        O.validate_eval_kw(kw)
        print(f"variante {v}: {variant_kw(v)}")
    except Exception as e:  # noqa: BLE001
        print(f"AVISO: variante {v} no soportada ({e}); sus columnas quedarán como ERR")
